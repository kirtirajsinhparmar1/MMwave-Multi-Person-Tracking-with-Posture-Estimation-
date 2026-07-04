from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from replay_posture_decision_fix import (
    _load_pose,
    _load_segments,
    _load_tracks,
    _norm_label,
    _pick_primary_tid,
    _safe_float,
    _switch_count,
    replay_decisions,
)


RANGE_MIN_VALUES = [0.0, 2.0, 2.5, 2.75, 3.0]
MIN_PROB_VALUES = [0.50, 0.52, 0.55, 0.58, 0.60]
MARGIN_VALUES = [0.12, 0.15, 0.18, 0.20, 0.25]
FRAMES_VALUES = [8, 10, 12, 16]
STANDING_VETO_PROB_VALUES = [0.50, 0.55, 0.60, 0.65]
STANDING_VETO_MARGIN_VALUES = [0.05, 0.08, 0.10, 0.12]


@dataclass
class Dataset:
    name: str
    session: Path
    rows: pd.DataFrame
    segments: pd.DataFrame
    segment_indices: dict[str, list[int]]


def _load_dataset(name: str, session: Path, segment_path: Path | None) -> Dataset:
    pose = _load_pose(session)
    pose = _load_tracks(session, pose)
    segments = _load_segments(segment_path, session, pose)
    base = replay_decisions(
        pose,
        range_min_m=0.0,
        min_prob=0.50,
        margin=0.12,
        frames=8,
        standing_veto_prob=1.01,
        standing_veto_margin=999.0,
        moving_guard=True,
    )
    base = base.reset_index(drop=True)
    base["row_id"] = base.index
    segment_indices: dict[str, list[int]] = {}
    for seg in segments.to_dict("records"):
        start = _safe_float(seg.get("start_time_s"), float("nan"))
        end = _safe_float(seg.get("end_time_s"), float("nan"))
        if math.isnan(start) or math.isnan(end):
            continue
        expected_distance = seg.get("expected_distance_m")
        window = base[(base["time_s"] >= start) & (base["time_s"] <= end)]
        primary_tid = _pick_primary_tid(window, expected_distance)
        if primary_tid is not None:
            window = window[window["tid"] == primary_tid]
        segment_indices[str(seg.get("segment_id", ""))] = window["row_id"].astype(int).tolist()
    return Dataset(name=name, session=session, rows=base, segments=segments, segment_indices=segment_indices)


def _stable_gate_mask(rows: pd.DataFrame, evidence: pd.Series, frames: int) -> list[bool]:
    counts: dict[int, int] = {}
    passed: list[bool] = []
    tids = rows["tid"].astype(int).tolist()
    evidence_values = evidence.astype(bool).tolist()
    for tid, ok in zip(tids, evidence_values):
        counts[tid] = counts.get(tid, 0) + 1 if ok else 0
        passed.append(counts[tid] >= frames)
    return passed


def _candidate_labels(rows: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    stand_prob = pd.to_numeric(rows["stand_prob"], errors="coerce").fillna(0.0)
    sit_prob = pd.to_numeric(rows["sit_prob"], errors="coerce").fillna(0.0)
    falling_prob = pd.to_numeric(rows["falling_prob"], errors="coerce").fillna(0.0)
    lying_prob = pd.to_numeric(rows["lying_prob"], errors="coerce").fillna(0.0)
    range_m = pd.to_numeric(rows["range_m"], errors="coerce")
    sit_minus_stand = sit_prob - stand_prob
    falling_lying_dominant = pd.concat([falling_prob, lying_prob], axis=1).max(axis=1) > pd.concat(
        [stand_prob, sit_prob], axis=1
    ).max(axis=1)
    evidence = (
        (range_m >= float(params["range_min_for_relative_gate_m"]))
        & (sit_prob >= float(params["soft_sitting_min_prob"]))
        & (sit_minus_stand >= float(params["relative_sitting_margin"]))
        & (stand_prob < float(params["standing_veto_prob"]))
        & ((stand_prob - sit_prob) < float(params["standing_veto_margin"]))
        & (~rows["moving_translation_confirmed"].astype(bool))
        & (~falling_lying_dominant)
    )
    gate_passed = _stable_gate_mask(rows, evidence, int(params["relative_sitting_frames"]))
    labels = rows["old_display_pose"].map(_norm_label).copy()
    gate = pd.Series(gate_passed, index=rows.index)
    labels[(labels == "STANDING") & gate] = "SITTING"
    labels[(labels == "MOVING") & gate & (~rows["moving_translation_confirmed"].astype(bool))] = "SITTING"
    return labels


def _accuracy(labels: pd.Series, expected_pose: str) -> float:
    expected = _norm_label(expected_pose)
    if labels.empty or expected not in {"STANDING", "SITTING", "MOVING", "LYING", "FALLING"}:
        return float("nan")
    return float((labels.map(_norm_label) == expected).mean())


def _rate(labels: pd.Series, target: str) -> float:
    if labels.empty:
        return float("nan")
    return float((labels.map(_norm_label) == target).mean())


def _segment_metrics(dataset: Dataset, candidate_labels: pd.Series) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    standing_old_frames: list[pd.Series] = []
    standing_new_frames: list[pd.Series] = []
    sitting_old_frames: list[pd.Series] = []
    sitting_new_frames: list[pd.Series] = []
    old_switches = 0
    new_switches = 0
    for seg in dataset.segments.to_dict("records"):
        segment_id = str(seg.get("segment_id", ""))
        idx = dataset.segment_indices.get(segment_id, [])
        if not idx:
            continue
        expected_pose = _norm_label(seg.get("expected_pose"))
        rows = dataset.rows.loc[idx]
        old_labels = rows["old_display_pose"].map(_norm_label)
        new_labels = candidate_labels.loc[idx].map(_norm_label)
        prefix = f"{dataset.name}_{segment_id}"
        metrics[f"{prefix}_old"] = _accuracy(old_labels, expected_pose)
        metrics[f"{prefix}_candidate"] = _accuracy(new_labels, expected_pose)
        old_switches += _switch_count(old_labels)
        new_switches += _switch_count(new_labels)
        if expected_pose == "STANDING":
            standing_old_frames.append(old_labels)
            standing_new_frames.append(new_labels)
        elif expected_pose == "SITTING":
            sitting_old_frames.append(old_labels)
            sitting_new_frames.append(new_labels)
    old_standing = pd.concat(standing_old_frames, ignore_index=True) if standing_old_frames else pd.Series(dtype=object)
    new_standing = pd.concat(standing_new_frames, ignore_index=True) if standing_new_frames else pd.Series(dtype=object)
    old_sitting = pd.concat(sitting_old_frames, ignore_index=True) if sitting_old_frames else pd.Series(dtype=object)
    new_sitting = pd.concat(sitting_new_frames, ignore_index=True) if sitting_new_frames else pd.Series(dtype=object)
    metrics[f"{dataset.name}_standing_false_sitting_rate_old"] = _rate(old_standing, "SITTING")
    metrics[f"{dataset.name}_standing_false_sitting_rate_candidate"] = _rate(new_standing, "SITTING")
    metrics[f"{dataset.name}_sitting_false_standing_rate_old"] = _rate(old_sitting, "STANDING")
    metrics[f"{dataset.name}_sitting_false_standing_rate_candidate"] = _rate(new_sitting, "STANDING")
    metrics[f"{dataset.name}_pose_switch_count_old"] = old_switches
    metrics[f"{dataset.name}_pose_switch_count_candidate"] = new_switches
    return metrics


def _acceptable(row: dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    for seg in ["standing_1m", "standing_2m", "standing_3m", "standing_4m"]:
        old = row.get(f"full_{seg}_old")
        new = row.get(f"full_{seg}_candidate")
        if pd.notna(old) and pd.notna(new) and (float(old) - float(new)) > 0.005:
            failures.append(f"{seg}_standing_drop")
    old_false = row.get("full_standing_false_sitting_rate_old")
    new_false = row.get("full_standing_false_sitting_rate_candidate")
    if pd.notna(old_false) and pd.notna(new_false) and (float(new_false) - float(old_false)) > 0.005:
        failures.append("standing_false_sitting_increase")
    for key in ["full_sitting_2m", "default_ab_sitting_2m"]:
        old = row.get(f"{key}_old")
        new = row.get(f"{key}_candidate")
        if pd.notna(old) and pd.notna(new) and (float(old) - float(new)) > 0.01:
            failures.append(f"{key}_drop")
    improvements = []
    for key in ["full_sitting_3m", "full_sitting_4m", "default_ab_sitting_3m", "default_ab_sitting_4m"]:
        old = row.get(f"{key}_old")
        new = row.get(f"{key}_candidate")
        if pd.notna(old) and pd.notna(new):
            improvements.append(float(new) - float(old))
    if not improvements or max(improvements) < 0.03:
        failures.append("no_3pt_sitting_3m_or_4m_gain")
    old_switch = row.get("full_pose_switch_count_old", 0) + row.get("default_ab_pose_switch_count_old", 0)
    new_switch = row.get("full_pose_switch_count_candidate", 0) + row.get("default_ab_pose_switch_count_candidate", 0)
    if old_switch and (float(new_switch) - float(old_switch)) / float(old_switch) > 0.10:
        failures.append("switch_count_increase")
    return not failures, ";".join(failures)


def _mine_errors(full: Dataset, out_dir: Path) -> pd.DataFrame:
    rows = []
    for seg in full.segments.to_dict("records"):
        segment_id = str(seg.get("segment_id", ""))
        idx = full.segment_indices.get(segment_id, [])
        if not idx:
            continue
        expected_pose = _norm_label(seg.get("expected_pose"))
        expected_distance = _safe_float(seg.get("expected_distance_m"), float("nan"))
        segment_rows = full.rows.loc[idx].copy()
        changed = segment_rows[segment_rows["old_display_pose"].map(_norm_label) != segment_rows["new_display_pose"].map(_norm_label)]
        for row in changed.to_dict("records"):
            old_pose = _norm_label(row.get("old_display_pose"))
            new_pose = _norm_label(row.get("new_display_pose"))
            range_m = _safe_float(row.get("range_m"), float("nan"))
            rows.append(
                {
                    "segment_id": segment_id,
                    "expected_pose": expected_pose,
                    "expected_distance_m": expected_distance,
                    "frame": row.get("frame", ""),
                    "timestamp_s": row.get("time_s", ""),
                    "tid": row.get("tid", ""),
                    "old_display_pose": old_pose,
                    "new_display_pose": new_pose,
                    "old_correct": old_pose == expected_pose,
                    "new_correct": new_pose == expected_pose,
                    "stand_prob": row.get("stand_prob", ""),
                    "sit_prob": row.get("sit_prob", ""),
                    "sit_minus_stand_margin": row.get("sit_minus_stand_margin", ""),
                    "quality": row.get("quality", ""),
                    "geom_pts": row.get("num_associated_points", row.get("geom_pts", "")),
                    "range_m": range_m,
                    "range_error_m": abs(range_m - expected_distance) if not math.isnan(range_m) and not math.isnan(expected_distance) else "",
                    "velocity_x": row.get("vx_mps", ""),
                    "velocity_y": row.get("vy_mps", ""),
                    "velocity_z": row.get("vz_mps", ""),
                    "speed_mps": row.get("horizontal_speed", ""),
                    "body_translation_if_available": row.get("moving_translation_displacement_m", ""),
                    "moving_override_state": row.get("moving_override_state", ""),
                    "moving_override_reason": row.get("replay_reason", ""),
                    "sitting_relative_gate_state": row.get("sitting_relative_gate_state", ""),
                    "sitting_relative_gate_stable_count": row.get("sitting_relative_gate_stable_count", ""),
                    "reason": row.get("replay_reason", ""),
                }
            )
    mined = pd.DataFrame(rows)
    mined.to_csv(out_dir / "relative_gate_error_mining.csv", index=False)
    standing_false = mined[
        (mined["expected_pose"] == "STANDING")
        & (mined["new_display_pose"] == "SITTING")
        & (mined["old_display_pose"] != "SITTING")
    ].copy()
    sitting_corrected = mined[
        (mined["expected_pose"] == "SITTING")
        & (mined["old_display_pose"] != "SITTING")
        & (mined["new_display_pose"] == "SITTING")
    ].copy()
    standing_false.to_csv(out_dir / "standing_false_sitting_frames.csv", index=False)
    sitting_corrected.to_csv(out_dir / "sitting_corrected_frames.csv", index=False)
    return mined


def _mine_summary(mined: pd.DataFrame) -> dict[str, Any]:
    false = mined[
        (mined["expected_pose"] == "STANDING")
        & (mined["new_display_pose"] == "SITTING")
        & (mined["old_display_pose"] != "SITTING")
    ]
    if false.empty:
        return {"standing_false_sitting_frames": 0}
    quality = false["quality"].astype(str).str.upper()
    return {
        "standing_false_sitting_frames": len(false),
        "standing_false_sitting_near_lt_2p5_rate": float((pd.to_numeric(false["range_m"], errors="coerce") < 2.5).mean()),
        "standing_false_sitting_no_points_rate": float(quality.eq("NO_POINTS").mean()),
        "standing_false_sitting_mean_stand_prob": pd.to_numeric(false["stand_prob"], errors="coerce").mean(),
        "standing_false_sitting_mean_sit_prob": pd.to_numeric(false["sit_prob"], errors="coerce").mean(),
        "standing_false_sitting_mean_margin": pd.to_numeric(false["sit_minus_stand_margin"], errors="coerce").mean(),
        "standing_false_sitting_moving_guard_rate": float(false["reason"].astype(str).str.contains("moving_override", case=False, na=False).mean()),
    }


def _write_report(out_dir: Path, results: pd.DataFrame, pareto: pd.DataFrame, mined: pd.DataFrame) -> None:
    mine = _mine_summary(mined)
    best = pareto.head(1)
    lines = [
        "# Sitting Gate Parameter Sweep Report",
        "",
        "## Regression Frame Mining",
        "",
        f"Standing false-SITTING changed frames: {mine.get('standing_false_sitting_frames', 0)}",
        f"Near-range (<2.5m) share: {mine.get('standing_false_sitting_near_lt_2p5_rate', float('nan')):.3f}",
        f"NO_POINTS share: {mine.get('standing_false_sitting_no_points_rate', float('nan')):.3f}",
        f"Mean stand_prob: {mine.get('standing_false_sitting_mean_stand_prob', float('nan')):.3f}",
        f"Mean sit_prob: {mine.get('standing_false_sitting_mean_sit_prob', float('nan')):.3f}",
        f"Mean sit-minus-stand margin: {mine.get('standing_false_sitting_mean_margin', float('nan')):.3f}",
        f"MOVING-guard reason share: {mine.get('standing_false_sitting_moving_guard_rate', float('nan')):.3f}",
        "",
        "## Sweep",
        "",
        f"Candidates evaluated: {len(results)}",
        f"Acceptable candidates: {int(results['acceptable'].sum()) if 'acceptable' in results else 0}",
        "",
    ]
    if best.empty:
        lines.extend(
            [
                "## Best Safe Candidate",
                "",
                "No candidate passed all acceptance criteria.",
            ]
        )
    else:
        best_lines = [
            "| " + " | ".join(str(col) for col in best.columns) + " |",
            "| " + " | ".join(["---"] * len(best.columns)) + " |",
        ]
        for row in best.to_dict("records"):
            best_lines.append("| " + " | ".join(str(row.get(col, "")) for col in best.columns) + " |")
        lines.extend(
            [
                "## Best Safe Candidate",
                "",
                "\n".join(best_lines),
            ]
        )
    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "A candidate must keep every standing segment within 0.5 percentage points of old accuracy, keep standing false-SITTING increase within 0.5 percentage points, avoid sitting_2m regression over 1 point, improve sitting_3m or sitting_4m by at least 3 points, and keep switch-count growth within 10%.",
        ]
    )
    (out_dir / "SITTING_GATE_PARAM_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep standing-protected sitting relative gate parameters offline.")
    parser.add_argument("--full-benchmark-session", required=True)
    parser.add_argument("--default-ab-session", required=True)
    parser.add_argument("--default-ab-segments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    full = _load_dataset("full", Path(args.full_benchmark_session).expanduser().resolve(), None)
    default_ab = _load_dataset(
        "default_ab",
        Path(args.default_ab_session).expanduser().resolve(),
        Path(args.default_ab_segments).expanduser().resolve(),
    )
    mined = _mine_errors(full, out_dir)

    param_grid = itertools.product(
        RANGE_MIN_VALUES,
        MIN_PROB_VALUES,
        MARGIN_VALUES,
        FRAMES_VALUES,
        STANDING_VETO_PROB_VALUES,
        STANDING_VETO_MARGIN_VALUES,
    )
    results: list[dict[str, Any]] = []
    for range_min, min_prob, margin, frames, veto_prob, veto_margin in param_grid:
        params = {
            "range_min_for_relative_gate_m": range_min,
            "soft_sitting_min_prob": min_prob,
            "relative_sitting_margin": margin,
            "relative_sitting_frames": frames,
            "standing_veto_prob": veto_prob,
            "standing_veto_margin": veto_margin,
        }
        full_labels = _candidate_labels(full.rows, params)
        ab_labels = _candidate_labels(default_ab.rows, params)
        row = dict(params)
        row.update(_segment_metrics(full, full_labels))
        row.update(_segment_metrics(default_ab, ab_labels))
        acceptable, failures = _acceptable(row)
        row["acceptable"] = acceptable
        row["failure_reasons"] = failures
        row["max_default_ab_sitting_3m_4m_gain"] = max(
            row.get("default_ab_sitting_3m_candidate", float("nan")) - row.get("default_ab_sitting_3m_old", float("nan")),
            row.get("default_ab_sitting_4m_candidate", float("nan")) - row.get("default_ab_sitting_4m_old", float("nan")),
        )
        row["max_full_sitting_3m_4m_gain"] = max(
            row.get("full_sitting_3m_candidate", float("nan")) - row.get("full_sitting_3m_old", float("nan")),
            row.get("full_sitting_4m_candidate", float("nan")) - row.get("full_sitting_4m_old", float("nan")),
        )
        row["total_switch_increase_rate"] = (
            (
                row.get("full_pose_switch_count_candidate", 0)
                + row.get("default_ab_pose_switch_count_candidate", 0)
                - row.get("full_pose_switch_count_old", 0)
                - row.get("default_ab_pose_switch_count_old", 0)
            )
            / max(1, row.get("full_pose_switch_count_old", 0) + row.get("default_ab_pose_switch_count_old", 0))
        )
        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["acceptable", "max_default_ab_sitting_3m_4m_gain", "max_full_sitting_3m_4m_gain", "total_switch_increase_rate"],
        ascending=[False, False, False, True],
    )
    result_df.to_csv(out_dir / "sweep_results.csv", index=False)
    acceptable_count = int(result_df["acceptable"].sum())
    pareto = result_df[result_df["acceptable"]].copy()
    if not pareto.empty:
        pareto = pareto.sort_values(
            ["max_default_ab_sitting_3m_4m_gain", "max_full_sitting_3m_4m_gain", "total_switch_increase_rate"],
            ascending=[False, False, True],
        ).head(25)
    pareto.to_csv(out_dir / "sweep_pareto_candidates.csv", index=False)
    _write_report(out_dir, result_df, pareto, mined)

    print(f"Candidates evaluated: {len(result_df)}")
    print(f"Acceptable candidates: {acceptable_count}")
    print(f"Output: {out_dir}")
    if not pareto.empty:
        keep = [
            "range_min_for_relative_gate_m",
            "soft_sitting_min_prob",
            "relative_sitting_margin",
            "relative_sitting_frames",
            "standing_veto_prob",
            "standing_veto_margin",
            "max_default_ab_sitting_3m_4m_gain",
            "total_switch_increase_rate",
        ]
        print(pareto[keep].head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
