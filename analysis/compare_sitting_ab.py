#!/usr/bin/env python
"""Compare sitting-only A/B posture analysis outputs.

The script compares a default cfg session against a static-retention cfg session.
It only consumes offline analysis CSVs; it does not change runtime posture logic,
thresholds, renderer behavior, RGB code, or cfg files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INPUTS = {
    "posture": "posture_verdict_by_segment.csv",
    "prob": "stand_sit_probability_by_segment.csv",
    "no_points": "no_points_effect_by_pose.csv",
    "combined": "combined_diagnostics_by_segment.csv",
    "tracking": "tracking_metrics_by_segment.csv",
}

SUMMARY_COLUMNS = [
    "segment_id",
    "expected_distance_m",
    "default_posture_accuracy",
    "static_posture_accuracy",
    "delta_posture_accuracy",
    "default_display_sitting_rate",
    "static_display_sitting_rate",
    "delta_display_sitting_rate",
    "default_display_standing_rate",
    "static_display_standing_rate",
    "delta_display_standing_rate",
    "default_mean_stand_prob",
    "static_mean_stand_prob",
    "delta_mean_stand_prob",
    "default_mean_sit_prob",
    "static_mean_sit_prob",
    "delta_mean_sit_prob",
    "default_stand_minus_sit_margin",
    "static_stand_minus_sit_margin",
    "delta_margin",
    "default_NO_POINTS_rate",
    "static_NO_POINTS_rate",
    "delta_NO_POINTS_rate",
    "default_mean_geom_pts",
    "static_mean_geom_pts",
    "delta_mean_geom_pts",
    "default_geom_pts_ge_3_rate",
    "static_geom_pts_ge_3_rate",
    "delta_geom_pts_ge_3_rate",
    "default_range_mae",
    "static_range_mae",
    "delta_range_mae",
    "verdict",
]


def read_required_tables(folder: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    missing: list[Path] = []
    for key, filename in REQUIRED_INPUTS.items():
        path = folder / filename
        if not path.exists():
            missing.append(path)
            continue
        df = pd.read_csv(path)
        df.columns = [str(col).strip() for col in df.columns]
        tables[key] = df
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required analysis CSVs:\n{formatted}")
    optional = folder / "no_points_stand_sit_diagnosis.csv"
    if optional.exists():
        df = pd.read_csv(optional)
        df.columns = [str(col).strip() for col in df.columns]
        tables["no_points_segment"] = df
    return tables


def number(value: Any) -> float | None:
    if value is None:
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def get_first(row: pd.Series | None, names: list[str]) -> Any:
    if row is None:
        return None
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if pd.isna(value):
            continue
        return value
    return None


def row_by_segment(df: pd.DataFrame, segment_id: str) -> pd.Series | None:
    if "segment_id" not in df.columns:
        return None
    matches = df[df["segment_id"].astype(str) == segment_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def no_points_rate_from_buckets(no_points_df: pd.DataFrame) -> dict[float, float]:
    required = {"expected_pose", "expected_distance_m", "quality_bucket", "frames"}
    if not required.issubset(no_points_df.columns):
        return {}
    df = no_points_df.copy()
    df = df[df["expected_pose"].astype(str).str.upper() == "SITTING"]
    df["expected_distance_m"] = pd.to_numeric(df["expected_distance_m"], errors="coerce")
    df["frames"] = pd.to_numeric(df["frames"], errors="coerce").fillna(0.0)
    result: dict[float, float] = {}
    for distance, group in df.groupby("expected_distance_m"):
        total = float(group["frames"].sum())
        if total <= 0:
            continue
        no_points_frames = float(
            group[group["quality_bucket"].astype(str).str.upper() == "NO_POINTS"]["frames"].sum()
        )
        result[float(distance)] = no_points_frames / total
    return result


def geom_ge3_from_segment_buckets(df: pd.DataFrame) -> dict[str, float]:
    required = {"segment_id", "frames", "geom_pts_ge_3_rate"}
    if not required.issubset(df.columns):
        return {}
    work = df.copy()
    work["frames"] = pd.to_numeric(work["frames"], errors="coerce").fillna(0.0)
    work["geom_pts_ge_3_rate"] = pd.to_numeric(work["geom_pts_ge_3_rate"], errors="coerce")
    result: dict[str, float] = {}
    for segment_id, group in work.groupby("segment_id"):
        valid = group.dropna(subset=["geom_pts_ge_3_rate"])
        total = float(valid["frames"].sum())
        if total <= 0:
            continue
        weighted = float((valid["frames"] * valid["geom_pts_ge_3_rate"]).sum() / total)
        result[str(segment_id)] = weighted
    return result


def collect_session_metrics(folder: Path) -> pd.DataFrame:
    tables = read_required_tables(folder)
    posture = tables["posture"]
    prob = tables["prob"]
    combined = tables["combined"]
    tracking = tables["tracking"]
    no_points_rates = no_points_rate_from_buckets(tables["no_points"])
    geom_ge3_rates = geom_ge3_from_segment_buckets(tables.get("no_points_segment", pd.DataFrame()))

    segment_ids: set[str] = set()
    for df in (posture, prob, combined, tracking):
        if "segment_id" in df.columns:
            segment_ids.update(df["segment_id"].dropna().astype(str).tolist())

    rows: list[dict[str, Any]] = []
    for segment_id in sorted(segment_ids):
        p_row = row_by_segment(posture, segment_id)
        prob_row = row_by_segment(prob, segment_id)
        c_row = row_by_segment(combined, segment_id)
        t_row = row_by_segment(tracking, segment_id)

        expected_pose = first_present(
            get_first(p_row, ["expected_pose"]),
            get_first(c_row, ["expected_pose"]),
            get_first(t_row, ["expected_pose"]),
        )
        if str(expected_pose).upper() != "SITTING":
            continue

        expected_distance = number(
            first_present(
                get_first(p_row, ["expected_distance_m"]),
                get_first(c_row, ["expected_distance_m"]),
                get_first(t_row, ["expected_distance_m"]),
                get_first(prob_row, ["expected_distance_m"]),
            )
        )

        no_points_rate = number(
            first_present(
                get_first(p_row, ["quality_NO_POINTS_rate", "NO_POINTS_rate"]),
                get_first(c_row, ["quality_NO_POINTS_rate", "NO_POINTS_rate"]),
            )
        )
        if no_points_rate is None and expected_distance is not None:
            no_points_rate = no_points_rates.get(expected_distance)

        mean_geom_pts = number(
            first_present(get_first(p_row, ["mean_geom_pts"]), get_first(c_row, ["mean_geom_pts"]))
        )
        geom_pts_ge_3_rate = number(
            first_present(
                get_first(p_row, ["geom_pts_ge_3_rate"]),
                get_first(c_row, ["geom_pts_ge_3_rate"]),
                get_first(t_row, ["geom_pts_ge_3_rate"]),
            )
        )
        if geom_pts_ge_3_rate is None:
            geom_pts_ge_3_rate = geom_ge3_rates.get(segment_id)

        mean_stand_prob = number(
            first_present(get_first(prob_row, ["mean_stand_prob"]), get_first(p_row, ["mean_stand_prob"]))
        )
        mean_sit_prob = number(
            first_present(get_first(prob_row, ["mean_sit_prob"]), get_first(p_row, ["mean_sit_prob"]))
        )
        margin = number(
            get_first(prob_row, ["mean_margin_stand_minus_sit", "stand_minus_sit_margin"])
        )
        if margin is None and mean_stand_prob is not None and mean_sit_prob is not None:
            margin = mean_stand_prob - mean_sit_prob

        rows.append(
            {
                "segment_id": segment_id,
                "expected_pose": "SITTING",
                "expected_distance_m": expected_distance,
                "posture_accuracy": number(get_first(p_row, ["accuracy", "posture_accuracy"])),
                "display_sitting_rate": number(get_first(p_row, ["sitting_rate", "display_sitting_rate"])),
                "display_standing_rate": number(
                    get_first(p_row, ["standing_rate", "display_standing_rate"])
                ),
                "mean_stand_prob": mean_stand_prob,
                "mean_sit_prob": mean_sit_prob,
                "stand_minus_sit_margin": margin,
                "NO_POINTS_rate": no_points_rate,
                "mean_geom_pts": mean_geom_pts,
                "geom_pts_ge_3_rate": geom_pts_ge_3_rate,
                "range_mae": number(
                    first_present(get_first(c_row, ["range_mae_m"]), get_first(t_row, ["mae_range_m"]))
                ),
                "range_jitter_p95_m": number(get_first(t_row, ["range_jitter_p95_m"])),
                "tracking_presence_rate": number(get_first(t_row, ["tracking_presence_rate"])),
                "extra_track_rate": number(get_first(t_row, ["extra_track_rate"])),
                "tid_switch_count": number(get_first(t_row, ["tid_switch_count"])),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["expected_distance_m", "segment_id"], na_position="last")
    return out


def prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {col: f"{prefix}_{col}" for col in df.columns if col != "segment_id"}
    return df.rename(columns=rename)


def delta(row: pd.Series, static_col: str, default_col: str) -> float | None:
    static_value = number(row.get(static_col))
    default_value = number(row.get(default_col))
    if static_value is None or default_value is None:
        return None
    return static_value - default_value


def improves(value: float | None, threshold: float = 0.10) -> bool:
    return value is not None and value > threshold


def worsens_negative(value: float | None, threshold: float = -0.10) -> bool:
    return value is not None and value < threshold


def compute_verdict(row: pd.Series) -> str:
    d_accuracy = number(row.get("delta_posture_accuracy"))
    d_geom = number(row.get("delta_mean_geom_pts"))
    d_no_points = number(row.get("delta_NO_POINTS_rate"))
    d_ge3 = number(row.get("delta_geom_pts_ge_3_rate"))
    d_range = number(row.get("delta_range_mae"))
    d_tracking_presence = delta(row, "static_tracking_presence_rate", "default_tracking_presence_rate")
    d_extra = delta(row, "static_extra_track_rate", "default_extra_track_rate")
    d_tid_switch = delta(row, "static_tid_switch_count", "default_tid_switch_count")
    d_sit_prob = number(row.get("delta_mean_sit_prob"))

    geometry_improved = improves(d_geom) or worsens_negative(d_no_points) or improves(d_ge3)
    accuracy_improved = improves(d_accuracy)
    static_stand_prob = number(row.get("static_mean_stand_prob"))
    static_sit_prob = number(row.get("static_mean_sit_prob"))
    stand_prob_still_dominates = (
        static_stand_prob is not None
        and static_sit_prob is not None
        and static_stand_prob > static_sit_prob
    )
    static_display_standing = number(row.get("static_display_standing_rate"))
    static_display_sitting = number(row.get("static_display_sitting_rate"))
    display_still_standing = (
        static_display_standing is not None
        and static_display_sitting is not None
        and static_display_standing > static_display_sitting
        and static_display_standing > 0.50
    )

    tracking_regression = (
        (d_range is not None and d_range > 0.20)
        or (d_tracking_presence is not None and d_tracking_presence < -0.05)
        or (d_extra is not None and d_extra > 0.10)
        or (d_tid_switch is not None and d_tid_switch > 1.0)
    )
    if tracking_regression:
        return "STATIC_RETENTION_TRACKING_REGRESSION"
    if geometry_improved and accuracy_improved:
        return "STATIC_RETENTION_HELPED_GEOMETRY_AND_POSTURE"
    if geometry_improved and not accuracy_improved and stand_prob_still_dominates:
        return "GEOMETRY_IMPROVED_MODEL_STILL_WRONG"
    if improves(d_sit_prob) and display_still_standing:
        return "GATING_DECISION_REMAINS_PROBLEM"
    if not geometry_improved:
        return "STATIC_RETENTION_DID_NOT_IMPROVE_GEOMETRY"
    return "MIXED_OR_INCONCLUSIVE"


def build_summary(default_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    merged = prefixed(default_df, "default").merge(
        prefixed(static_df, "static"), on="segment_id", how="outer"
    )
    merged["expected_distance_m"] = merged["default_expected_distance_m"].combine_first(
        merged["static_expected_distance_m"]
    )

    metric_pairs = {
        "posture_accuracy": "posture_accuracy",
        "display_sitting_rate": "display_sitting_rate",
        "display_standing_rate": "display_standing_rate",
        "mean_stand_prob": "mean_stand_prob",
        "mean_sit_prob": "mean_sit_prob",
        "stand_minus_sit_margin": "margin",
        "NO_POINTS_rate": "NO_POINTS_rate",
        "mean_geom_pts": "mean_geom_pts",
        "geom_pts_ge_3_rate": "geom_pts_ge_3_rate",
        "range_mae": "range_mae",
    }
    for metric, delta_name in metric_pairs.items():
        static_col = f"static_{metric}"
        default_col = f"default_{metric}"
        merged[f"delta_{delta_name}"] = merged.apply(lambda row: delta(row, static_col, default_col), axis=1)

    merged["verdict"] = merged.apply(compute_verdict, axis=1)

    summary = pd.DataFrame(
        {
            "segment_id": merged["segment_id"],
            "expected_distance_m": merged["expected_distance_m"],
            "default_posture_accuracy": merged.get("default_posture_accuracy"),
            "static_posture_accuracy": merged.get("static_posture_accuracy"),
            "delta_posture_accuracy": merged.get("delta_posture_accuracy"),
            "default_display_sitting_rate": merged.get("default_display_sitting_rate"),
            "static_display_sitting_rate": merged.get("static_display_sitting_rate"),
            "delta_display_sitting_rate": merged.get("delta_display_sitting_rate"),
            "default_display_standing_rate": merged.get("default_display_standing_rate"),
            "static_display_standing_rate": merged.get("static_display_standing_rate"),
            "delta_display_standing_rate": merged.get("delta_display_standing_rate"),
            "default_mean_stand_prob": merged.get("default_mean_stand_prob"),
            "static_mean_stand_prob": merged.get("static_mean_stand_prob"),
            "delta_mean_stand_prob": merged.get("delta_mean_stand_prob"),
            "default_mean_sit_prob": merged.get("default_mean_sit_prob"),
            "static_mean_sit_prob": merged.get("static_mean_sit_prob"),
            "delta_mean_sit_prob": merged.get("delta_mean_sit_prob"),
            "default_stand_minus_sit_margin": merged.get("default_stand_minus_sit_margin"),
            "static_stand_minus_sit_margin": merged.get("static_stand_minus_sit_margin"),
            "delta_margin": merged.get("delta_margin"),
            "default_NO_POINTS_rate": merged.get("default_NO_POINTS_rate"),
            "static_NO_POINTS_rate": merged.get("static_NO_POINTS_rate"),
            "delta_NO_POINTS_rate": merged.get("delta_NO_POINTS_rate"),
            "default_mean_geom_pts": merged.get("default_mean_geom_pts"),
            "static_mean_geom_pts": merged.get("static_mean_geom_pts"),
            "delta_mean_geom_pts": merged.get("delta_mean_geom_pts"),
            "default_geom_pts_ge_3_rate": merged.get("default_geom_pts_ge_3_rate"),
            "static_geom_pts_ge_3_rate": merged.get("static_geom_pts_ge_3_rate"),
            "delta_geom_pts_ge_3_rate": merged.get("delta_geom_pts_ge_3_rate"),
            "default_range_mae": merged.get("default_range_mae"),
            "static_range_mae": merged.get("static_range_mae"),
            "delta_range_mae": merged.get("delta_range_mae"),
            "verdict": merged["verdict"],
        }
    )
    return summary[SUMMARY_COLUMNS].sort_values(["expected_distance_m", "segment_id"], na_position="last")


def comparison_tables(summary: pd.DataFrame, default_df: pd.DataFrame, static_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tracking = prefixed(default_df, "default").merge(
        prefixed(static_df, "static"), on="segment_id", how="outer"
    )
    tracking["expected_distance_m"] = tracking["default_expected_distance_m"].combine_first(
        tracking["static_expected_distance_m"]
    )
    for metric in ["tracking_presence_rate", "extra_track_rate", "tid_switch_count", "range_mae", "range_jitter_p95_m"]:
        tracking[f"delta_{metric}"] = tracking.apply(
            lambda row, name=metric: delta(row, f"static_{name}", f"default_{name}"), axis=1
        )

    probability_cols = [
        "segment_id",
        "expected_distance_m",
        "default_mean_stand_prob",
        "static_mean_stand_prob",
        "delta_mean_stand_prob",
        "default_mean_sit_prob",
        "static_mean_sit_prob",
        "delta_mean_sit_prob",
        "default_stand_minus_sit_margin",
        "static_stand_minus_sit_margin",
        "delta_margin",
        "verdict",
    ]
    geometry_cols = [
        "segment_id",
        "expected_distance_m",
        "default_NO_POINTS_rate",
        "static_NO_POINTS_rate",
        "delta_NO_POINTS_rate",
        "default_mean_geom_pts",
        "static_mean_geom_pts",
        "delta_mean_geom_pts",
        "default_geom_pts_ge_3_rate",
        "static_geom_pts_ge_3_rate",
        "delta_geom_pts_ge_3_rate",
        "verdict",
    ]
    tracking_cols = [
        "segment_id",
        "expected_distance_m",
        "default_tracking_presence_rate",
        "static_tracking_presence_rate",
        "delta_tracking_presence_rate",
        "default_extra_track_rate",
        "static_extra_track_rate",
        "delta_extra_track_rate",
        "default_tid_switch_count",
        "static_tid_switch_count",
        "delta_tid_switch_count",
        "default_range_mae",
        "static_range_mae",
        "delta_range_mae",
        "default_range_jitter_p95_m",
        "static_range_jitter_p95_m",
        "delta_range_jitter_p95_m",
    ]
    return {
        "sitting_ab_probability_comparison.csv": summary[probability_cols],
        "sitting_ab_geometry_comparison.csv": summary[geometry_cols],
        "sitting_ab_tracking_comparison.csv": tracking[tracking_cols],
    }


def yes_no_mixed(values: list[bool | None]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return "unknown"
    if all(known):
        return "yes"
    if not any(known):
        return "no"
    return "mixed by distance"


def fmt(value: Any) -> str:
    parsed = number(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.3f}"


def segment_line(summary: pd.DataFrame, segment_id: str) -> str:
    row_df = summary[summary["segment_id"] == segment_id]
    if row_df.empty:
        return f"{segment_id}: no comparison row was available."
    row = row_df.iloc[0]
    return (
        f"{segment_id}: accuracy {fmt(row['default_posture_accuracy'])} -> "
        f"{fmt(row['static_posture_accuracy'])}, sit_prob {fmt(row['default_mean_sit_prob'])} -> "
        f"{fmt(row['static_mean_sit_prob'])}, stand_prob {fmt(row['default_mean_stand_prob'])} -> "
        f"{fmt(row['static_mean_stand_prob'])}, mean_geom_pts {fmt(row['default_mean_geom_pts'])} -> "
        f"{fmt(row['static_mean_geom_pts'])}, verdict {row['verdict']}."
    )


def final_verdict_text(summary: pd.DataFrame) -> str:
    geometry_answers: list[bool | None] = []
    accuracy_answers: list[bool | None] = []
    for _, row in summary.iterrows():
        d_geom = number(row.get("delta_mean_geom_pts"))
        d_no_points = number(row.get("delta_NO_POINTS_rate"))
        d_ge3 = number(row.get("delta_geom_pts_ge_3_rate"))
        known = [v is not None for v in (d_geom, d_no_points, d_ge3)]
        if any(known):
            geometry_answers.append(improves(d_geom) or worsens_negative(d_no_points) or improves(d_ge3))
        else:
            geometry_answers.append(None)
        d_accuracy = number(row.get("delta_posture_accuracy"))
        accuracy_answers.append(None if d_accuracy is None else improves(d_accuracy))

    sitting_4m = summary[summary["segment_id"] == "sitting_4m"]
    if sitting_4m.empty:
        four_m_answer = "unknown"
    else:
        row = sitting_4m.iloc[0]
        stand = number(row.get("static_mean_stand_prob"))
        sit = number(row.get("static_mean_sit_prob"))
        four_m_answer = "unknown" if stand is None or sit is None else ("yes" if stand > sit else "no")

    sitting_3m = summary[summary["segment_id"] == "sitting_3m"]
    if sitting_3m.empty:
        three_m_answer = "unknown"
    else:
        row = sitting_3m.iloc[0]
        stand = number(row.get("static_mean_stand_prob"))
        sit = number(row.get("static_mean_sit_prob"))
        display_stand = number(row.get("static_display_standing_rate"))
        display_sit = number(row.get("static_display_sitting_rate"))
        mismatch = (
            stand is not None
            and sit is not None
            and sit > stand
            and display_stand is not None
            and display_sit is not None
            and display_stand > display_sit
        )
        three_m_answer = "yes" if mismatch else "no"

    dominant_verdict = "MIXED_OR_INCONCLUSIVE"
    if not summary.empty and "verdict" in summary.columns:
        dominant_verdict = str(summary["verdict"].mode().iloc[0])

    if dominant_verdict == "STATIC_RETENTION_HELPED_GEOMETRY_AND_POSTURE":
        next_path = "continue cfg/static seated point extraction validation before changing posture logic"
    elif dominant_verdict == "GEOMETRY_IMPROVED_MODEL_STILL_WRONG":
        next_path = "inspect posture features/model behavior after confirming geometry improved"
    elif dominant_verdict == "GATING_DECISION_REMAINS_PROBLEM":
        next_path = "inspect display/gating logic after confirming sit probability dominates"
    elif dominant_verdict == "STATIC_RETENTION_TRACKING_REGRESSION":
        next_path = "resolve tracking/range regression before posture-specific changes"
    elif dominant_verdict == "STATIC_RETENTION_DID_NOT_IMPROVE_GEOMETRY":
        next_path = "do not pursue static retention as the primary fix path without additional evidence"
    else:
        next_path = "collect cleaner A/B data or inspect per-frame traces before selecting a fix path"

    return "\n".join(
        [
            f"- Did static retention improve seated point geometry? {yes_no_mixed(geometry_answers)}.",
            f"- Did static retention improve sitting posture accuracy? {yes_no_mixed(accuracy_answers)}.",
            f"- Did sitting_4m still favor STANDING? {four_m_answer}.",
            f"- Did sitting_3m still suffer from gating/display mismatch? {three_m_answer}.",
            f"- What should we fix next? {next_path}.",
        ]
    )


def write_report(out_dir: Path, default_path: Path, static_path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Sitting A/B Comparison Report",
        "",
        "## 1. Purpose",
        "Compare the current default cfg against TI static-retention cfg for sitting-only posture behavior.",
        "",
        "## 2. Protocol",
        "- Test A: sitting at 2m, 3m, and 4m for 60 seconds each with the default cfg.",
        "- Test B: sitting at 2m, 3m, and 4m for 60 seconds each with the static-retention cfg.",
        "",
        "## 3. Sessions compared",
        f"- Default analysis folder: `{default_path}`",
        f"- Static-retention analysis folder: `{static_path}`",
        "",
        "## 4. Tracking comparison",
        "See `sitting_ab_tracking_comparison.csv` for range, jitter, extra-track, presence, and TID-switch deltas.",
        "",
        "## 5. Geometry/point evidence comparison",
        "See `sitting_ab_geometry_comparison.csv` for NO_POINTS, mean_geom_pts, and geom_pts_ge_3 deltas.",
        "",
        "## 6. Stand-vs-sit probability comparison",
        "See `sitting_ab_probability_comparison.csv` for mean stand/sit probabilities and margins.",
        "",
        "## 7. Posture accuracy comparison",
        "See `sitting_ab_summary.csv` for posture accuracy and display-rate deltas.",
        "",
        "## 8. Per-distance result: 2m",
        segment_line(summary, "sitting_2m"),
        "",
        "## 9. Per-distance result: 3m",
        segment_line(summary, "sitting_3m"),
        "",
        "## 10. Per-distance result: 4m",
        segment_line(summary, "sitting_4m"),
        "",
        "## 11. Final verdict",
        final_verdict_text(summary),
        "",
        "## 12. Recommended next engineering path",
        "Use the final verdict above to choose the next engineering path. Do not change posture thresholds, renderer logic, RGB code, or model behavior based on this report until the live A/B sessions are recorded and analyzed.",
        "",
    ]
    (out_dir / "SITTING_AB_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", required=True, type=Path, help="Default-cfg analysis output folder.")
    parser.add_argument("--static", required=True, type=Path, help="Static-retention analysis output folder.")
    parser.add_argument("--out", required=True, type=Path, help="Comparison output folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_df = collect_session_metrics(args.default)
    static_df = collect_session_metrics(args.static)
    if default_df.empty:
        raise ValueError(f"No SITTING segment rows found in {args.default}")
    if static_df.empty:
        raise ValueError(f"No SITTING segment rows found in {args.static}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(default_df, static_df)
    summary.to_csv(args.out / "sitting_ab_summary.csv", index=False)
    for filename, table in comparison_tables(summary, default_df, static_df).items():
        table.to_csv(args.out / filename, index=False)
    write_report(args.out, args.default, args.static, summary)
    print(f"Wrote sitting A/B comparison outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
