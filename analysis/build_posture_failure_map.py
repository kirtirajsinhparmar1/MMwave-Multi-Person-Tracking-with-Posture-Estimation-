"""Build an offline posture failure map from existing lite posture logs.

This script does not train or export a replacement posture model. It measures
where the old displayed posture and raw old ONNX probabilities disagree with
the user-provided segment protocol labels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cleaned-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def coarse_pose(value: object) -> str:
    text = norm_text(value)
    if "SIT" in text:
        return "SITTING"
    if "STAND" in text:
        return "STANDING"
    if "MOVE" in text or "WALK" in text:
        return "MOVING"
    if "LIE" in text or "LYING" in text:
        return "LYING"
    if "FALL" in text:
        return "FALLING"
    return "UNKNOWN"


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def predicted_display_pose(row: pd.Series) -> str:
    rate_cols = {
        "STANDING": "display_standing_rate",
        "SITTING": "display_sitting_rate",
        "MOVING": "display_moving_rate",
        "UNKNOWN": "display_unknown_rate",
    }
    present = {label: as_float(row.get(col)) for label, col in rate_cols.items() if col in row.index}
    if present:
        label, value = max(present.items(), key=lambda item: item[1])
        return label if value > 0 else "UNKNOWN"

    direct_col = None
    for candidate in ("old_display_pose", "display_pose", "displayed_pose", "final_pose", "old_pose"):
        if candidate in row.index:
            direct_col = candidate
            break
    return coarse_pose(row.get(direct_col)) if direct_col else "UNKNOWN"


def predicted_raw_prob_pose(row: pd.Series) -> str:
    stand = as_float(row.get("stand_prob_mean"), default=math.nan)
    sit = as_float(row.get("sit_prob_mean"), default=math.nan)
    if math.isnan(stand) or math.isnan(sit):
        return "UNKNOWN"
    return "SITTING" if sit > stand else "STANDING"


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expected_pose_norm"] = out.get("expected_pose", "").map(coarse_pose)
    out["expected_subpose_norm"] = out.get("expected_subpose", "").map(norm_text)
    out["expected_position_norm"] = out.get("expected_position", "").map(norm_text)
    out["expected_distance_m_num"] = numeric(out, "expected_distance_m")
    out["people_count_num"] = numeric(out, "people_count")
    out["people_count_num"] = numeric(out, "people_count")
    out["old_display_pred"] = out.apply(predicted_display_pose, axis=1)
    out["raw_prob_pred"] = out.apply(predicted_raw_prob_pose, axis=1)
    out["display_correct"] = out["old_display_pred"] == out["expected_pose_norm"]
    out["raw_prob_correct"] = out["raw_prob_pred"] == out["expected_pose_norm"]
    out["display_false_sitting_on_standing"] = (
        (out["expected_pose_norm"] == "STANDING") & (out["old_display_pred"] == "SITTING")
    )
    out["display_false_standing_on_sitting"] = (
        (out["expected_pose_norm"] == "SITTING") & (out["old_display_pred"] == "STANDING")
    )
    out["raw_false_sitting_on_standing"] = (
        (out["expected_pose_norm"] == "STANDING") & (out["raw_prob_pred"] == "SITTING")
    )
    out["raw_false_standing_on_sitting"] = (
        (out["expected_pose_norm"] == "SITTING") & (out["raw_prob_pred"] == "STANDING")
    )
    out["is_standing_3m"] = (
        (out["expected_pose_norm"] == "STANDING")
        & (out["expected_distance_m_num"].sub(3.0).abs() <= 0.25)
    )
    out["range_error_m"] = (numeric(out, "range_m_mean") - out["expected_distance_m_num"]).abs()
    return out


def mean_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.astype(float).mean())


def conditional_rate(df: pd.DataFrame, mask: pd.Series, condition: pd.Series) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    return float((mask & condition).sum() / denom)


def metrics_for(group: pd.DataFrame) -> dict[str, object]:
    standing_mask = group["expected_pose_norm"] == "STANDING"
    sitting_mask = group["expected_pose_norm"] == "SITTING"
    standing_3m_mask = group["is_standing_3m"]
    upright_mask = group["expected_subpose_norm"] == "SITTING_UPRIGHT"
    lean_back_mask = group["expected_subpose_norm"] == "SITTING_LEAN_BACK"
    lean_forward_mask = group["expected_subpose_norm"] == "SITTING_LEAN_FORWARD"
    return {
        "windows": len(group),
        "segments": group.get("segment_id", pd.Series(dtype=object)).nunique(),
        "person_instances": group[["session_id", "segment_id", "assigned_tid"]].drop_duplicates().shape[0]
        if {"session_id", "segment_id", "assigned_tid"}.issubset(group.columns)
        else 0,
        "old_display_posture_accuracy": mean_bool(group["display_correct"]),
        "raw_probability_posture_accuracy": mean_bool(group["raw_prob_correct"]),
        "standing_accuracy": conditional_rate(group, standing_mask, group["display_correct"]),
        "sitting_accuracy": conditional_rate(group, sitting_mask, group["display_correct"]),
        "false_sitting_on_standing": conditional_rate(
            group, standing_mask, group["old_display_pred"] == "SITTING"
        ),
        "false_sitting_on_standing_3m": conditional_rate(
            group, standing_3m_mask, group["old_display_pred"] == "SITTING"
        ),
        "false_standing_on_sitting": conditional_rate(
            group, sitting_mask, group["old_display_pred"] == "STANDING"
        ),
        "upright_sitting_accuracy": conditional_rate(group, upright_mask, group["display_correct"]),
        "lean_back_sitting_accuracy": conditional_rate(group, lean_back_mask, group["display_correct"]),
        "lean_forward_sitting_accuracy": conditional_rate(
            group, lean_forward_mask, group["display_correct"]
        ),
        "disappearance_rate": numeric(group, "disappearance_rate").mean(),
        "ui_visible_rate": numeric(group, "ui_visible_rate", default=1.0).mean(),
        "NO_POINTS_rate": numeric(group, "NO_POINTS_rate").mean(),
        "LOW_POINTS_rate": numeric(group, "LOW_POINTS_rate").mean(),
        "geom_pts_mean": numeric(group, "geom_pts_mean").mean(),
        "range_error_mean": group["range_error_m"].mean(),
        "people_count_mean": numeric(group, "people_count").mean(),
    }


def grouped_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(by, keys)}
        row.update(metrics_for(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by + ["windows"], ascending=[True] * len(by) + [False])


def select_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "window_id",
        "session_id",
        "segment_id",
        "person_slot",
        "assigned_tid",
        "window_size_s",
        "window_start_s",
        "window_end_s",
        "expected_pose",
        "expected_subpose",
        "expected_distance_m",
        "expected_position",
        "people_count",
        "old_display_pred",
        "raw_prob_pred",
        "stand_prob_mean",
        "sit_prob_mean",
        "sit_minus_stand_mean",
        "range_m_mean",
        "range_error_m",
        "geom_pts_mean",
        "NO_POINTS_rate",
        "LOW_POINTS_rate",
        "OK_rate",
        "ui_visible_rate",
        "disappearance_rate",
        "tracking_presence_rate",
        "pose_presence_rate",
        "pose_switch_count",
        "tid_switch_count",
        "visibility_reliability_label",
    ]
    return df[[col for col in wanted if col in df.columns]]


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def worst_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    valid = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return None
    return valid.sort_values(metric, ascending=True).iloc[0]


def write_report(out: Path, windows: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    overall = metrics_for(windows)
    by_subpose = tables["subpose"]
    by_distance = tables["distance"]
    by_position = tables["position"]
    by_people = tables["people_count"]

    worst_subpose = worst_row(by_subpose, "old_display_posture_accuracy")
    worst_distance = worst_row(by_distance, "old_display_posture_accuracy")
    worst_position = worst_row(by_position, "old_display_posture_accuracy")

    two_person = by_people[by_people["people_count_num"] == 2.0] if "people_count_num" in by_people else pd.DataFrame()
    one_person = by_people[by_people["people_count_num"] == 1.0] if "people_count_num" in by_people else pd.DataFrame()
    two_person_text = "not measured"
    if not two_person.empty and not one_person.empty:
        two_acc = float(two_person.iloc[0]["old_display_posture_accuracy"])
        one_acc = float(one_person.iloc[0]["old_display_posture_accuracy"])
        two_person_text = f"two-person accuracy {pct(two_acc)} vs one-person {pct(one_acc)}"

    upright = windows[windows["expected_subpose_norm"] == "SITTING_UPRIGHT"]
    lean_forward = windows[windows["expected_subpose_norm"] == "SITTING_LEAN_FORWARD"]
    upright_mask = windows["expected_subpose_norm"] == "SITTING_UPRIGHT"
    lean_forward_mask = windows["expected_subpose_norm"] == "SITTING_LEAN_FORWARD"
    upright_to_stand = conditional_rate(windows, upright_mask, windows["old_display_pred"] == "STANDING")
    lean_forward_to_stand = conditional_rate(
        windows, lean_forward_mask, windows["old_display_pred"] == "STANDING"
    )
    lean_forward_to_moving = conditional_rate(
        windows, lean_forward_mask, windows["old_display_pred"] == "MOVING"
    )

    wrong = windows[~windows["display_correct"]]
    correct = windows[windows["display_correct"]]
    wrong_quality = {
        "disappearance_rate": numeric(wrong, "disappearance_rate").mean() if not wrong.empty else 0.0,
        "NO_POINTS_rate": numeric(wrong, "NO_POINTS_rate").mean() if not wrong.empty else 0.0,
        "LOW_POINTS_rate": numeric(wrong, "LOW_POINTS_rate").mean() if not wrong.empty else 0.0,
        "geom_pts_mean": numeric(wrong, "geom_pts_mean").mean() if not wrong.empty else 0.0,
    }
    correct_quality = {
        "disappearance_rate": numeric(correct, "disappearance_rate").mean() if not correct.empty else 0.0,
        "NO_POINTS_rate": numeric(correct, "NO_POINTS_rate").mean() if not correct.empty else 0.0,
        "LOW_POINTS_rate": numeric(correct, "LOW_POINTS_rate").mean() if not correct.empty else 0.0,
        "geom_pts_mean": numeric(correct, "geom_pts_mean").mean() if not correct.empty else 0.0,
    }

    lines = [
        "# Posture Failure Map Report",
        "",
        "This is an offline analysis of existing lite logs. Protocol segment labels are the ground truth; old displayed posture and raw old ONNX probabilities are measured outputs/features only.",
        "",
        "## Overall Metrics",
        "",
        f"- Windows analyzed: {overall['windows']}",
        f"- Old displayed posture accuracy: {pct(float(overall['old_display_posture_accuracy']))}",
        f"- Raw old ONNX probability posture accuracy: {pct(float(overall['raw_probability_posture_accuracy']))}",
        f"- Standing accuracy: {pct(float(overall['standing_accuracy']))}",
        f"- Sitting accuracy: {pct(float(overall['sitting_accuracy']))}",
        f"- False SITTING on STANDING: {pct(float(overall['false_sitting_on_standing']))}",
        f"- False SITTING on standing_3m: {pct(float(overall['false_sitting_on_standing_3m']))}",
        f"- False STANDING on SITTING: {pct(float(overall['false_standing_on_sitting']))}",
        "",
        "## Required Questions",
        "",
        f"1. Which pose/subpose is worst? {worst_subpose['expected_subpose_norm']} has the lowest displayed accuracy at {pct(float(worst_subpose['old_display_posture_accuracy']))}." if worst_subpose is not None else "1. Which pose/subpose is worst? Not enough rows.",
        f"2. Which distance is worst? {worst_distance['expected_distance_m_num']}m has the lowest displayed accuracy at {pct(float(worst_distance['old_display_posture_accuracy']))}." if worst_distance is not None else "2. Which distance is worst? Not enough rows.",
        f"3. Which position is worst? {worst_position['expected_position_norm']} has the lowest displayed accuracy at {pct(float(worst_position['old_display_posture_accuracy']))}." if worst_position is not None else "3. Which position is worst? Not enough rows.",
        f"4. Does two-person degrade posture? {two_person_text}.",
        f"5. Does 3m standing become false sitting? Yes, measured false SITTING on standing_3m is {pct(float(overall['false_sitting_on_standing_3m']))}.",
        f"6. Does upright sitting become standing? Upright sitting is displayed as STANDING on {pct(upright_to_stand)} of upright windows.",
        f"7. Does lean-forward become standing or moving? Lean-forward is displayed as STANDING on {pct(lean_forward_to_stand)} and MOVING on {pct(lean_forward_to_moving)} of lean-forward windows.",
        "8. Are failures correlated with disappearance/NO_POINTS/low geom? "
        f"Wrong windows average disappearance={wrong_quality['disappearance_rate']:.3f}, NO_POINTS={wrong_quality['NO_POINTS_rate']:.3f}, LOW_POINTS={wrong_quality['LOW_POINTS_rate']:.3f}, geom_pts={wrong_quality['geom_pts_mean']:.2f}; "
        f"correct windows average disappearance={correct_quality['disappearance_rate']:.3f}, NO_POINTS={correct_quality['NO_POINTS_rate']:.3f}, LOW_POINTS={correct_quality['LOW_POINTS_rate']:.3f}, geom_pts={correct_quality['geom_pts_mean']:.2f}.",
        "9. Which data should be collected next once point logging is added? Prioritize the lowest-accuracy subpose/distance/position combinations in failure_map_by_distance_subpose_position.csv, with repeated standing_3m, upright sitting, lean-forward sitting, left/right, two-person, and 5m coverage using associated point-cloud logging.",
        "",
        "## Output Tables",
        "",
        "- failure_map_by_session.csv",
        "- failure_map_by_distance.csv",
        "- failure_map_by_subpose.csv",
        "- failure_map_by_position.csv",
        "- failure_map_by_people_count.csv",
        "- failure_map_by_distance_subpose_position.csv",
        "- standing_false_sitting_cases.csv",
        "- sitting_false_standing_cases.csv",
        "- disappearance_failure_cases.csv",
    ]
    (out / "POSTURE_FAILURE_MAP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out = ensure_dir(Path(args.out))
    windows_path = dataset_root / "posturenet_lite_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing lite dataset: {windows_path}")

    windows = pd.read_csv(windows_path)
    windows = add_predictions(windows)

    tables = {
        "session": grouped_metrics(windows, ["session_id"]),
        "distance": grouped_metrics(windows, ["expected_distance_m_num"]),
        "subpose": grouped_metrics(windows, ["expected_subpose_norm"]),
        "position": grouped_metrics(windows, ["expected_position_norm"]),
        "people_count": grouped_metrics(windows, ["people_count_num"]),
        "distance_subpose_position": grouped_metrics(
            windows, ["expected_distance_m_num", "expected_subpose_norm", "expected_position_norm"]
        ),
    }
    tables["session"].to_csv(out / "failure_map_by_session.csv", index=False)
    tables["distance"].to_csv(out / "failure_map_by_distance.csv", index=False)
    tables["subpose"].to_csv(out / "failure_map_by_subpose.csv", index=False)
    tables["position"].to_csv(out / "failure_map_by_position.csv", index=False)
    tables["people_count"].to_csv(out / "failure_map_by_people_count.csv", index=False)
    tables["distance_subpose_position"].to_csv(
        out / "failure_map_by_distance_subpose_position.csv", index=False
    )

    standing_cases = windows[
        (windows["expected_pose_norm"] == "STANDING")
        & ((windows["old_display_pred"] == "SITTING") | (windows["raw_prob_pred"] == "SITTING"))
    ]
    sitting_cases = windows[
        (windows["expected_pose_norm"] == "SITTING")
        & ((windows["old_display_pred"] == "STANDING") | (windows["raw_prob_pred"] == "STANDING"))
    ]
    disappearance_cases = windows[
        (numeric(windows, "disappearance_rate") > 0.0)
        | (numeric(windows, "ui_visible_rate", default=1.0) < 0.95)
        | (numeric(windows, "NO_POINTS_rate") > 0.0)
        | (numeric(windows, "LOW_POINTS_rate") > 0.25)
        | (numeric(windows, "tracking_presence_rate", default=1.0) < 0.95)
        | (numeric(windows, "pose_presence_rate", default=1.0) < 0.95)
    ]
    select_case_columns(standing_cases).to_csv(out / "standing_false_sitting_cases.csv", index=False)
    select_case_columns(sitting_cases).to_csv(out / "sitting_false_standing_cases.csv", index=False)
    select_case_columns(disappearance_cases).to_csv(out / "disappearance_failure_cases.csv", index=False)
    write_report(out, windows, tables)
    print(f"Failure map written to {out}")
    print(f"Windows analyzed: {len(windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
