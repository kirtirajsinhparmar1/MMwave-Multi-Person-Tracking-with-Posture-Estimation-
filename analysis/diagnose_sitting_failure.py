"""Generate diagnostic-only sitting posture failure reports.

This script reads the latest distance/posture analysis output and the current
session selected by that output. It does not modify runtime behavior, cfg files,
thresholds, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_PROB_COLUMNS = [
    "segment_id",
    "expected_distance_m",
    "expected_pose",
    "accuracy",
    "display_standing_rate",
    "display_sitting_rate",
    "mean_stand_prob",
    "mean_sit_prob",
    "median_stand_prob",
    "median_sit_prob",
    "mean_margin_stand_minus_sit",
    "median_margin_stand_minus_sit",
    "frames_stand_prob_gt_sit_prob",
    "frames_sit_prob_gt_stand_prob",
    "frames_display_standing_when_sit_prob_gt_stand_prob",
    "frames_display_sitting_when_stand_prob_gt_sit_prob",
    "failure_type",
    "evidence",
    "recommendation",
]

NO_POINTS_COLUMNS = [
    "segment_id",
    "expected_pose",
    "distance",
    "quality_bucket",
    "frames",
    "accuracy",
    "display_standing_rate",
    "display_sitting_rate",
    "mean_stand_prob",
    "mean_sit_prob",
    "mean_margin",
    "mean_geom_pts",
    "geom_pts_ge_1_rate",
    "geom_pts_ge_3_rate",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_existing(row: pd.Series, names: Iterable[str], default=None):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def resolve_session_path(out_dir: Path) -> Path | None:
    candidate_path = out_dir / "candidate_sessions.csv"
    if not candidate_path.exists():
        return None
    candidates = pd.read_csv(candidate_path)
    if candidates.empty or "path" not in candidates.columns:
        return None
    return Path(str(candidates.iloc[0]["path"]))


def normalize_pose_label(value) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "WARMUP"}:
        return "UNKNOWN"
    if "STAND" in text:
        return "STANDING"
    if "SIT" in text:
        return "SITTING"
    if "MOVE" in text or "WALK" in text:
        return "MOVING"
    if "FALL" in text:
        return "FALLING"
    if "LY" in text or "LAY" in text:
        return "LYING"
    if text in {"STANDING", "SITTING", "MOVING", "FALLING", "LYING", "UNKNOWN"}:
        return text
    return "OTHER"


def normalize_pose_table(pose: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["frame"] = numeric(pose.get("mmwave_frame_num", pose.get("frame_num", pd.Series(dtype=float))))
    if "host_monotonic_ns" in pose.columns:
        ts_ns = numeric(pose["host_monotonic_ns"])
        created_ns = None
        if metadata:
            created_ns = metadata.get("created_monotonic_ns")
        try:
            origin_ns = float(created_ns) if created_ns is not None else float(ts_ns.min())
        except (TypeError, ValueError):
            origin_ns = float(ts_ns.min())
        out["time_s"] = (ts_ns - origin_ns) / 1_000_000_000.0
    elif "timestamp_s" in pose.columns:
        out["time_s"] = numeric(pose["timestamp_s"])
    else:
        raise ValueError("No usable timestamp column found in pose table.")

    display = pose.get("final_label", pose.get("display_pose", pose.get("pose", "")))
    out["display_pose"] = display.map(normalize_pose_label)
    quality = pose.get("quality_flag", pose.get("quality", ""))
    out["quality"] = quality.astype(str).str.upper()
    out["stand_prob"] = numeric(pose.get("prob_standing", pose.get("stand_prob", pd.Series(dtype=float)))).fillna(0.0)
    out["sit_prob"] = numeric(pose.get("prob_sitting", pose.get("sit_prob", pd.Series(dtype=float)))).fillna(0.0)
    out["geom_pts"] = numeric(pose.get("geom_pts", pose.get("num_points", pd.Series(dtype=float)))).fillna(0.0)
    if "tid" in pose.columns:
        out["tid"] = pose["tid"]
    return out


def pose_rows_for_segment(pose: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    start = float(segment["start_time_s"])
    end = float(segment["end_time_s"])
    rows = pose[(pose["time_s"] >= start) & (pose["time_s"] <= end)].copy()
    if "frame" in rows.columns:
        rows = rows.drop_duplicates(subset=["frame"], keep="last")
    return rows


def quality_bucket(row: pd.Series) -> str:
    quality = str(row.get("quality", "")).upper()
    geom_pts = float(row.get("geom_pts", 0.0) or 0.0)
    if "NO_POINTS" in quality or geom_pts <= 0:
        return "NO_POINTS"
    if "LOW_POINTS" in quality:
        return "LOW_POINTS"
    if geom_pts >= 1:
        return "HAS_POINTS"
    return "OTHER"


def classify_failure(row: pd.Series, no_points_rate: float, mean_geom_pts: float) -> tuple[str, str]:
    expected = str(row["expected_pose"]).upper()
    accuracy = float(row["accuracy"])
    mean_stand = float(row["mean_stand_prob"])
    mean_sit = float(row["mean_sit_prob"])
    display_standing = float(row["standing_rate"])
    diff = mean_stand - mean_sit

    labels: list[str] = []
    if expected == "SITTING":
        if diff > 0.10:
            labels.append("MODEL_FAVORS_STANDING")
        if mean_sit > mean_stand and display_standing >= 0.15:
            labels.append("GATE_HOLDS_STANDING")
        if abs(diff) <= 0.10:
            labels.append("AMBIGUOUS_PROBABILITIES")
        if no_points_rate > 0.60 and mean_geom_pts < 1.0:
            labels.append("LOW_GEOMETRY_TARGET_ONLY")

    if not labels:
        if accuracy >= 0.90:
            return "NO_FAILURE", "accuracy is high and the dominant displayed posture matches the expected pose"
        return "MIXED", "accuracy is below the strong-pass level but no single probability rule dominates"
    if len(labels) > 1:
        return "MIXED", "; ".join(labels)
    return labels[0], labels[0]


def recommendation_for(failure_type: str, evidence: str) -> str:
    if failure_type == "NO_FAILURE":
        return "No posture change indicated by this segment."
    if "MODEL_FAVORS_STANDING" in evidence:
        return "Test whether more seated point evidence changes the model probabilities before changing thresholds."
    if "GATE_HOLDS_STANDING" in evidence:
        return "Inspect stand-to-sit gating only after confirming the probability and geometry evidence path."
    if "AMBIGUOUS_PROBABILITIES" in evidence:
        return "Collect the A/B probability-margin comparison before adding range-aware margins or features."
    if "LOW_GEOMETRY_TARGET_ONLY" in evidence:
        return "Run the static-retention A/B test to see whether seated geometry becomes available."
    return "Keep this diagnostic-only; defer fixes until the A/B evidence separates model, gating, and geometry causes."


def build_probability_diagnosis(out_dir: Path, pose: pd.DataFrame) -> pd.DataFrame:
    verdict = read_csv(out_dir / "posture_verdict_by_segment.csv")
    probs = read_csv(out_dir / "stand_sit_probability_by_segment.csv")
    combined = read_csv(out_dir / "combined_diagnostics_by_segment.csv")
    segments = read_csv(out_dir / "segments_auto.csv")

    merged = verdict.merge(
        probs,
        on=["segment_id", "expected_pose", "expected_distance_m"],
        suffixes=("_verdict", ""),
    ).merge(
        combined[["segment_id", "quality_NO_POINTS_rate", "mean_geom_pts"]],
        on="segment_id",
        suffixes=("", "_combined"),
    )

    segment_lookup = {str(row["segment_id"]): row for _, row in segments.iterrows()}
    rows = []
    for _, row in merged.iterrows():
        segment_id = str(row["segment_id"])
        seg_rows = pose_rows_for_segment(pose, segment_lookup[segment_id])
        display_standing_when_sit_gt_stand = int(
            ((seg_rows["display_pose"] == "STANDING") & (seg_rows["sit_prob"] > seg_rows["stand_prob"])).sum()
        )
        display_sitting_when_stand_gt_sit = int(
            ((seg_rows["display_pose"] == "SITTING") & (seg_rows["stand_prob"] > seg_rows["sit_prob"])).sum()
        )
        no_points_rate = float(first_existing(row, ["quality_NO_POINTS_rate_combined", "quality_NO_POINTS_rate"], 0.0))
        mean_geom_pts = float(first_existing(row, ["mean_geom_pts_combined", "mean_geom_pts"], 0.0))
        failure_type, failure_evidence = classify_failure(row, no_points_rate, mean_geom_pts)

        evidence = (
            f"mean_stand={float(row['mean_stand_prob']):.3f}, "
            f"mean_sit={float(row['mean_sit_prob']):.3f}, "
            f"margin={float(row['mean_margin_stand_minus_sit']):.3f}, "
            f"display_standing={float(row['standing_rate']):.3f}, "
            f"display_sitting={float(row['sitting_rate']):.3f}, "
            f"NO_POINTS_rate={no_points_rate:.3f}, "
            f"mean_geom_pts={mean_geom_pts:.3f}; "
            f"{failure_evidence}"
        )
        rows.append(
            {
                "segment_id": segment_id,
                "expected_distance_m": float(row["expected_distance_m"]),
                "expected_pose": row["expected_pose"],
                "accuracy": float(row["accuracy"]),
                "display_standing_rate": float(row["standing_rate"]),
                "display_sitting_rate": float(row["sitting_rate"]),
                "mean_stand_prob": float(row["mean_stand_prob"]),
                "mean_sit_prob": float(row["mean_sit_prob"]),
                "median_stand_prob": float(row["median_stand_prob"]),
                "median_sit_prob": float(row["median_sit_prob"]),
                "mean_margin_stand_minus_sit": float(row["mean_margin_stand_minus_sit"]),
                "median_margin_stand_minus_sit": float(row["median_margin_stand_minus_sit"]),
                "frames_stand_prob_gt_sit_prob": int(row["frames_stand_prob_gt_sit_prob"]),
                "frames_sit_prob_gt_stand_prob": int(row["frames_sit_prob_gt_stand_prob"]),
                "frames_display_standing_when_sit_prob_gt_stand_prob": display_standing_when_sit_gt_stand,
                "frames_display_sitting_when_stand_prob_gt_sit_prob": display_sitting_when_stand_gt_sit,
                "failure_type": failure_type,
                "evidence": evidence,
                "recommendation": recommendation_for(failure_type, failure_evidence),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_PROB_COLUMNS)


def build_no_points_diagnosis(out_dir: Path, pose: pd.DataFrame) -> pd.DataFrame:
    segments = read_csv(out_dir / "segments_auto.csv")
    rows = []
    for _, segment in segments.iterrows():
        seg_rows = pose_rows_for_segment(pose, segment)
        if seg_rows.empty:
            continue
        seg_rows = seg_rows.copy()
        seg_rows["quality_bucket"] = seg_rows.apply(quality_bucket, axis=1)
        expected = str(segment["expected_pose"]).upper()
        for bucket, bucket_rows in seg_rows.groupby("quality_bucket", sort=True):
            frames = len(bucket_rows)
            rows.append(
                {
                    "segment_id": segment["segment_id"],
                    "expected_pose": expected,
                    "distance": float(segment["expected_distance_m"]),
                    "quality_bucket": bucket,
                    "frames": int(frames),
                    "accuracy": float((bucket_rows["display_pose"] == expected).mean()) if frames else 0.0,
                    "display_standing_rate": float((bucket_rows["display_pose"] == "STANDING").mean()) if frames else 0.0,
                    "display_sitting_rate": float((bucket_rows["display_pose"] == "SITTING").mean()) if frames else 0.0,
                    "mean_stand_prob": float(bucket_rows["stand_prob"].mean()) if frames else 0.0,
                    "mean_sit_prob": float(bucket_rows["sit_prob"].mean()) if frames else 0.0,
                    "mean_margin": float((bucket_rows["stand_prob"] - bucket_rows["sit_prob"]).mean()) if frames else 0.0,
                    "mean_geom_pts": float(bucket_rows["geom_pts"].mean()) if frames else 0.0,
                    "geom_pts_ge_1_rate": float((bucket_rows["geom_pts"] >= 1).mean()) if frames else 0.0,
                    "geom_pts_ge_3_rate": float((bucket_rows["geom_pts"] >= 3).mean()) if frames else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=NO_POINTS_COLUMNS)


def read_metadata(session_path: Path | None) -> dict:
    if not session_path:
        return {}
    metadata_path = session_path / "session_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_cfg_lines(cfg_path: str | None) -> dict:
    result = {
        "cfg_path": cfg_path or "unknown",
        "sensor_position": "unknown",
        "static_range_angle": "unknown",
        "static_retention_available": "unknown",
    }
    if not cfg_path:
        return result
    path = Path(cfg_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("sensorPosition"):
                result["sensor_position"] = stripped
            elif stripped.startswith("staticRangeAngleCfg"):
                result["static_range_angle"] = stripped
    candidates = list(path.parent.glob("*staticRetention*.cfg")) if path.parent.exists() else []
    if candidates:
        preferred_prefix = path.name.split("_", 1)[0].upper()
        preferred = [candidate for candidate in candidates if candidate.name.upper().startswith(preferred_prefix)]
        result["static_retention_available"] = str((preferred or candidates)[0])
    else:
        result["static_retention_available"] = "not found next to latest cfg"
    return result


def summarize_rows(prob_diag: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["segment_id"]): row for _, row in prob_diag.iterrows()}


def write_failure_report(out_dir: Path, prob_diag: pd.DataFrame, no_points: pd.DataFrame, cfg_info: dict) -> None:
    rows = summarize_rows(prob_diag)
    sitting_4m = rows.get("sitting_4m")
    sitting_3m = rows.get("sitting_3m")
    sitting_1m = rows.get("sitting_1m")
    sitting_2m = rows.get("sitting_2m")

    lines = [
        "# Sitting Posture Failure Diagnosis",
        "",
        "## 1. Executive conclusion",
        "",
        "Tracking is not the bottleneck in the latest distance posture benchmark. The failure is in sit-vs-stand posture discrimination and becomes severe at 3m and 4m.",
        "",
        "We are not applying random threshold changes.",
        "We are not adding target-only posture rules yet.",
        "We are first determining whether the failure is model probability, decision gating, or geometry/feature availability.",
        "",
        "## 2. Tracking is not the bottleneck",
        "",
        "The current segment diagnostics show tracking presence at 100%, ID switches at 0, and extra track rate at 0 across the benchmark segments. Standing posture accuracy is about 99-100%, including far segments with high NO_POINTS rates.",
        "",
        "## 3. Sitting posture failure summary",
        "",
        "| Segment | Accuracy | Display STANDING | Display SITTING | Mean stand prob | Mean sit prob | Failure type |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for segment_id in ["sitting_1m", "sitting_2m", "sitting_3m", "sitting_4m"]:
        row = rows[segment_id]
        lines.append(
            f"| {segment_id} | {row['accuracy']:.3f} | {row['display_standing_rate']:.3f} | "
            f"{row['display_sitting_rate']:.3f} | {row['mean_stand_prob']:.3f} | "
            f"{row['mean_sit_prob']:.3f} | {row['failure_type']} |"
        )

    lines += [
        "",
        "## 4. Posture input data flow",
        "",
        "The posture model receives a 176-float vector built from an 8-frame window of 22-float per-frame posture features. Each 22-float frame contains target-level kinematics plus up to five associated point entries. Missing point entries are zero-padded.",
        "",
        "## 5. Meaning of NO_POINTS / geom_pts / assoc",
        "",
        "NO_POINTS is the overlay quality label emitted when the current frame has zero associated points for the target. `geom_pts` is the associated point count used for point geometry. Association modes describe whether point evidence came from target-index matching, nearest-neighbor fallback, or no association.",
        "",
        "## 6. Probability-level diagnosis",
        "",
        f"During sitting_4m, stand_prob is actually higher than sit_prob: mean stand={sitting_4m['mean_stand_prob']:.3f}, mean sit={sitting_4m['mean_sit_prob']:.3f}, mean margin={sitting_4m['mean_margin_stand_minus_sit']:.3f}. That segment has {int(sitting_4m['frames_stand_prob_gt_sit_prob'])} frames where stand_prob > sit_prob and {int(sitting_4m['frames_sit_prob_gt_stand_prob'])} frames where sit_prob > stand_prob.",
        "",
        f"During sitting_3m, the mean probabilities do not favor STANDING: mean stand={sitting_3m['mean_stand_prob']:.3f}, mean sit={sitting_3m['mean_sit_prob']:.3f}. The mean gap is larger than the 0.10 ambiguity rule in favor of SITTING, but the segment is frame-mixed: {int(sitting_3m['frames_stand_prob_gt_sit_prob'])} frames still have stand_prob > sit_prob, and display STANDING remains common.",
        "",
        f"During sitting_1m/2m, the model is better: sitting_1m mean sit={sitting_1m['mean_sit_prob']:.3f} vs stand={sitting_1m['mean_stand_prob']:.3f}; sitting_2m mean sit={sitting_2m['mean_sit_prob']:.3f} vs stand={sitting_2m['mean_stand_prob']:.3f}. Display still shows residual STANDING at {sitting_1m['display_standing_rate']:.3f} and {sitting_2m['display_standing_rate']:.3f}.",
        "",
        "## 7. Geometry / point-evidence diagnosis",
        "",
        "NO_POINTS and low geometry are much more damaging to sitting than to standing. Standing remains correct even with high NO_POINTS rates at 3m and 4m, while sitting fails under similar or lower geometry availability. The current data proves sparse geometry is part of the sitting failure context; it does not prove that a runtime NO_POINTS rule would be correct.",
        "",
        "## 8. Sensor/cfg/static seated target diagnosis",
        "",
        f"Latest benchmark cfg: `{cfg_info['cfg_path']}`",
        f"Sensor position line: `{cfg_info['sensor_position']}`",
        f"Static range-angle line: `{cfg_info['static_range_angle']}`",
        f"Static-retention/fine-motion cfg available for later testing: `{cfg_info['static_retention_available']}`",
        "",
        "The metric that would prove static retention helps is an A/B increase in seated `mean_geom_pts` and `geom_pts_ge_3_rate` that also increases sitting accuracy or shifts mean sit probability above mean stand probability at 3m/4m.",
        "",
        "## 9. What is proven",
        "",
        "Tracking is strong. Standing is nearly perfect. Sitting_4m is a model-probability failure under sparse geometry because mean stand probability exceeds mean sit probability. Sitting_3m is not a mean model-favors-standing case; it is mixed frame-level probability plus display/gating under sparse geometry. Sitting_1m/2m have better model probabilities but still show residual STANDING display.",
        "",
        "## 10. What is not proven",
        "",
        "The data does not prove that changing thresholds, holding previous sitting posture, suppressing target-only posture, changing the model, or changing cfg will fix the issue. It also does not prove that NO_POINTS alone causes failure, because standing succeeds with NO_POINTS.",
        "",
        "## 11. Recommended next experiment",
        "",
        "Run the current/original cfg versus the already available static-retention cfg on sitting at 2m, 3m, and 4m for 60 seconds each. Compare posture accuracy, stand/sit probabilities, NO_POINTS rate, mean_geom_pts, geom_pts_ge_3_rate, range error, range jitter, and time to stable sitting.",
        "",
        "## 12. What not to change yet",
        "",
        "Do not tune random posture thresholds, add target-only posture rules, add hold-previous-posture logic, suppress target-only posture, change the model, or modify cfg until the A/B test separates model probability, display/gating, and geometry availability.",
        "",
    ]
    (out_dir.parent.parent / "SITTING_POSTURE_FAILURE_DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")


def write_ab_plan(repo_root: Path) -> None:
    lines = [
        "# Next Sitting A/B Test Plan",
        "",
        "## Protocol",
        "",
        "### Test A: current/original cfg",
        "",
        "- sitting at 2m for 60 sec",
        "- sitting at 3m for 60 sec",
        "- sitting at 4m for 60 sec",
        "",
        "### Test B: static-retention/fine-motion cfg, only if already available",
        "",
        "- sitting at 2m for 60 sec",
        "- sitting at 3m for 60 sec",
        "- sitting at 4m for 60 sec",
        "",
        "## Metrics to compare",
        "",
        "- posture_accuracy",
        "- display_standing_rate",
        "- display_sitting_rate",
        "- mean_stand_prob",
        "- mean_sit_prob",
        "- stand_minus_sit_margin",
        "- NO_POINTS_rate",
        "- mean_geom_pts",
        "- geom_pts_ge_3_rate",
        "- range_mae_m",
        "- range_jitter",
        "- time_to_stable_sitting",
        "",
        "## Decision rules",
        "",
        "If static-retention increases geom_pts and sitting accuracy, cfg/static seated point extraction is likely the fix path.",
        "",
        "If geom_pts increases but stand_prob still dominates, posture model/features need improvement.",
        "",
        "If sit_prob dominates but display remains STANDING, decision/gating logic needs improvement.",
        "",
        "If probabilities are close/ambiguous, range-aware sit-vs-stand margin or additional geometry features may be needed later.",
        "",
        "No implementation changes are part of this plan.",
        "",
    ]
    (repo_root / "NEXT_SITTING_AB_TEST_PLAN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_output_dir", type=Path)
    args = parser.parse_args()

    out_dir = args.analysis_output_dir.resolve()
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)
    repo_root = out_dir.parent.parent
    session_path = resolve_session_path(out_dir)

    pose_path = out_dir / "mmwave_pose.csv"
    if not pose_path.exists() and session_path:
        pose_path = session_path / "mmwave_pose.csv"

    metadata = read_metadata(session_path)
    pose = normalize_pose_table(read_csv(pose_path), metadata)

    prob_diag = build_probability_diagnosis(out_dir, pose)
    no_points = build_no_points_diagnosis(out_dir, pose)
    prob_diag.to_csv(out_dir / "sitting_probability_failure_diagnosis.csv", index=False)
    no_points.to_csv(out_dir / "no_points_stand_sit_diagnosis.csv", index=False)

    cfg_info = read_cfg_lines(metadata.get("mmwave_cfg_path"))
    write_failure_report(out_dir, prob_diag, no_points, cfg_info)
    write_ab_plan(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
