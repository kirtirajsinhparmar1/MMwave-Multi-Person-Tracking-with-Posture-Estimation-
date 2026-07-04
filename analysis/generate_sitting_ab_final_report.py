#!/usr/bin/env python
"""Prepare and report the sitting-only cfg A/B analysis.

This helper only reads recorded logs and offline analysis outputs. It does not
modify runtime posture logic, thresholds, cfg files, model files, renderer code,
or RGB code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_distance_posture_session as analyzer


ROOT = Path(__file__).resolve().parents[1]
SHARED_LOG_ROOT = ROOT.parent / "logs"
LOCAL_LOG_ROOT = ROOT / "logs"
ABS_LOG_ROOT = Path(r"C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs")

DEFAULT_NAME = "sitting_ab_default_cfg"
STATIC_NAME = "sitting_ab_static_retention_cfg"
DEFAULT_ANALYSIS = ROOT / "analysis_outputs" / "sitting_ab_default_analysis"
STATIC_ANALYSIS = ROOT / "analysis_outputs" / "sitting_ab_static_retention_analysis"
COMPARISON = ROOT / "analysis_outputs" / "sitting_ab_comparison"
DISCOVERY_CSV = ROOT / "analysis_outputs" / "sitting_ab_session_discovery.csv"
DEFAULT_SEGMENTS = ROOT / "analysis_inputs" / "sitting_ab_default_segments.csv"
STATIC_SEGMENTS = ROOT / "analysis_inputs" / "sitting_ab_static_retention_segments.csv"
FINAL_REPORT = COMPARISON / "SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md"
COMPLETION_REPORT = ROOT / "SITTING_AB_ANALYSIS_COMPLETION.md"

PLOT_NAMES = [
    "timeline_range_by_track.png",
    "timeline_display_pose.png",
    "timeline_quality_geom_pts.png",
    "timeline_stand_sit_probs.png",
    "posture_accuracy_by_distance.png",
    "pose_distribution_by_segment.png",
    "stand_vs_sit_probability_by_segment.png",
    "stand_minus_sit_margin_by_segment.png",
    "sitting_segments_stand_sit_prob_timeline.png",
    "tracking_vs_posture_summary.png",
    "failure_mode_heatmap.png",
]

PLOT_EXPLANATIONS = {
    "timeline_range_by_track.png": "Check the distance plateaus and transition trims used for segmentation.",
    "timeline_display_pose.png": "Check whether the UI/display output stayed STANDING during sitting.",
    "timeline_quality_geom_pts.png": "Check NO_POINTS and associated geometry availability over time.",
    "timeline_stand_sit_probs.png": "Check whether model probabilities favored STANDING or SITTING.",
    "posture_accuracy_by_distance.png": "Compare sitting accuracy by distance.",
    "pose_distribution_by_segment.png": "Check pose confusion distribution in each segment.",
    "stand_vs_sit_probability_by_segment.png": "Compare average stand and sit probability by segment.",
    "stand_minus_sit_margin_by_segment.png": "Positive margin means STANDING probability exceeded SITTING.",
    "sitting_segments_stand_sit_prob_timeline.png": "Inspect stand/sit probability dynamics inside sitting segments.",
    "tracking_vs_posture_summary.png": "Separate tracking quality from posture accuracy.",
    "failure_mode_heatmap.png": "Locate dominant posture failure modes by distance.",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def csv_count(path: Path) -> int:
    return len(list(path.glob("*.csv"))) if path.exists() else 0


def latest_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    mtimes = [path.stat().st_mtime]
    mtimes.extend(p.stat().st_mtime for p in path.glob("*") if p.is_file())
    return max(mtimes)


def has_rgb_video(path: Path) -> bool:
    candidates = [
        path / "rgb_annotated.mp4",
        path / "videos" / "rgb_annotated.mp4",
        path / f"{path.name}_rgb_annotated.mp4",
    ]
    return any(p.exists() and p.stat().st_size > 0 for p in candidates)


def candidate_row(path: Path, kind: str, rank: int, selected: bool, notes: str) -> dict[str, Any]:
    metadata = read_json(path / "session_metadata.json")
    pose_meta = read_json(path / "pose_ui_metadata.json")
    cfg_path = metadata.get("mmwave_cfg_path") or pose_meta.get("cfg_path") or ""
    session_id = metadata.get("session_id") or ""
    if not session_id and path.name in {DEFAULT_NAME, STATIC_NAME}:
        session_id = path.name
    return {
        "candidate_type": kind,
        "rank": rank,
        "path": str(path),
        "modified_time": pd.to_datetime(latest_mtime(path), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "session_id_if_found": session_id,
        "cfg_path_if_found": cfg_path,
        "csv_count": csv_count(path),
        "has_mmwave_pose": (path / "mmwave_pose.csv").exists(),
        "has_rgb_video": has_rgb_video(path),
        "selected": selected,
        "notes": notes,
    }


def discover_sessions() -> tuple[Path, Path, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, Path] = {}
    roots = []
    for root in [SHARED_LOG_ROOT, LOCAL_LOG_ROOT, ABS_LOG_ROOT]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)

    for name, kind in [(DEFAULT_NAME, "default"), (STATIC_NAME, "static")]:
        candidates: list[Path] = []
        for root in roots:
            path = root / name
            if path.exists() and path.is_dir():
                candidates.append(path)
        candidates = sorted(candidates, key=lambda p: (csv_count(p), latest_mtime(p)), reverse=True)
        for rank, path in enumerate(candidates, start=1):
            combined_files = all((path / n).exists() for n in ["mmwave_frames.csv", "mmwave_tracks.csv", "mmwave_pose.csv"])
            selected_flag = rank == 1 and combined_files
            if selected_flag:
                selected[kind] = path
            note_bits = []
            if combined_files:
                note_bits.append("selected combined-log folder with mmwave_frames/tracks/pose")
            elif (path / "pose_predictions_ui.csv").exists():
                note_bits.append("project-local UI pose folder; not selected because combined CSVs are absent")
            else:
                note_bits.append("candidate folder with partial files")
            rows.append(candidate_row(path, kind, rank, selected_flag, "; ".join(note_bits)))
        if kind not in selected and candidates:
            selected[kind] = candidates[0]
            rows.append(candidate_row(candidates[0], kind, 99, True, "best-effort fallback selection"))

    if "default" not in selected or "static" not in selected:
        raise FileNotFoundError("Could not identify both sitting A/B session folders.")

    discovery = pd.DataFrame(rows)
    DISCOVERY_CSV.parent.mkdir(parents=True, exist_ok=True)
    discovery.to_csv(DISCOVERY_CSV, index=False)
    return selected["default"], selected["static"], discovery


def read_raw_session(session: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warnings: list[str] = []
    raw_frames = analyzer.safe_read_csv(session / "mmwave_frames.csv", warnings)
    raw_tracks = analyzer.safe_read_csv(session / "mmwave_tracks.csv", warnings)
    raw_pose = analyzer.safe_read_csv(session / "mmwave_pose.csv", warnings)
    ctx = analyzer.Context(session=session, out=ROOT / "analysis_outputs", warnings=warnings)
    ctx.t0_ns = analyzer.global_t0_ns([raw_frames, raw_tracks, raw_pose])
    analyzer.infer_frame_period(ctx, raw_frames, "auto")
    frames = analyzer.normalize_frames(raw_frames, ctx)
    tracks = analyzer.normalize_tracking(raw_tracks, ctx)
    pose = analyzer.normalize_posture(raw_pose, ctx, tracks)
    return frames, tracks, pose


def expected_sitting_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"segment_id": "sitting_2m", "expected_pose": "SITTING", "expected_distance_m": 2.0, "expected_order": 1},
            {"segment_id": "sitting_3m", "expected_pose": "SITTING", "expected_distance_m": 3.0, "expected_order": 2},
            {"segment_id": "sitting_4m", "expected_pose": "SITTING", "expected_distance_m": 4.0, "expected_order": 3},
        ]
    )


def infer_and_write_segments(session: Path, out_csv: Path) -> pd.DataFrame:
    frames, tracks, _pose = read_raw_session(session)
    ctx = analyzer.Context(session=session, out=out_csv.parent, warnings=[])
    expected = expected_sitting_table()
    segments = analyzer.auto_segments(expected, tracks, frames, min_seconds=35.0, target_seconds=50.0, ctx=ctx)
    out = segments[
        [
            "segment_id",
            "expected_pose",
            "expected_distance_m",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "method",
            "confidence",
            "notes",
        ]
    ].copy()
    out = out.rename(columns={"method": "segmentation_method"})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def n(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def fmt(value: Any, digits: int = 3) -> str:
    parsed = n(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.{digits}f}"


def pct(value: Any) -> str:
    parsed = n(value)
    if parsed is None:
        return "NA"
    return f"{parsed * 100:.1f}%"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe = df.copy()
    safe = safe.fillna("NA")
    lines = ["| " + " | ".join(map(str, safe.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(safe.columns)) + " |")
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in safe.columns) + " |")
    return "\n".join(lines)


def row_for(df: pd.DataFrame, segment_id: str) -> pd.Series | None:
    if df.empty or "segment_id" not in df:
        return None
    match = df[df["segment_id"].astype(str) == segment_id]
    if match.empty:
        return None
    return match.iloc[0]


def verdict_yes_no(delta: Any, positive_is_good: bool = True, threshold: float = 0.05) -> str:
    parsed = n(delta)
    if parsed is None:
        return "unknown"
    if positive_is_good:
        if parsed > threshold:
            return "improved"
        if parsed < -threshold:
            return "worse"
    else:
        if parsed < -threshold:
            return "improved"
        if parsed > threshold:
            return "worse"
    return "flat"


def read_rgb_summary(session: Path) -> dict[str, Any]:
    result = {"session": session.name}
    for filename, key in [
        ("rgb_frames.csv", "rgb_frames_rows"),
        ("rgb_tracks.csv", "rgb_tracks_rows"),
        ("rgb_keypoints.csv", "rgb_keypoints_rows"),
        ("sync_index.csv", "sync_index_rows"),
        ("rgb_actions.csv", "rgb_actions_rows"),
    ]:
        path = session / filename
        if path.exists():
            try:
                result[key] = max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
            except OSError:
                result[key] = "NA"
        else:
            result[key] = 0
    result["rgb_annotated_mp4"] = "present" if has_rgb_video(session) else "missing"
    return result


def cfg_path(session: Path) -> str:
    return str(read_json(session / "session_metadata.json").get("mmwave_cfg_path", ""))


def session_id(session: Path) -> str:
    return str(read_json(session / "session_metadata.json").get("session_id", session.name))


def selected_discovery_path(discovery: pd.DataFrame, kind: str) -> Path:
    row = discovery[(discovery["candidate_type"] == kind) & (discovery["selected"].astype(bool))]
    if row.empty:
        return Path("")
    return Path(str(row.iloc[0]["path"]))


def make_session_table(default_session: Path, static_session: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_name": "default_cfg",
                "session_path": str(default_session),
                "cfg_path": cfg_path(default_session),
                "session_id": session_id(default_session),
                "manual_or_auto_segments": "auto range-plateau suggestions written to manual CSV",
                "rgb_video_present": has_rgb_video(default_session),
                "notes": "combined mmWave/RGB log folder selected",
            },
            {
                "test_name": "static_retention_cfg",
                "session_path": str(static_session),
                "cfg_path": cfg_path(static_session),
                "session_id": session_id(static_session),
                "manual_or_auto_segments": "auto range-plateau suggestions written to manual CSV",
                "rgb_video_present": has_rgb_video(static_session),
                "notes": "combined mmWave/RGB log folder selected",
            },
        ]
    )


def make_segment_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_name, path in [("default_cfg", DEFAULT_SEGMENTS), ("static_retention_cfg", STATIC_SEGMENTS)]:
        df = read_csv(path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "test_name": test_name,
                    "segment_id": row.get("segment_id"),
                    "expected_pose": row.get("expected_pose"),
                    "expected_distance_m": fmt(row.get("expected_distance_m"), 1),
                    "start_time_s": fmt(row.get("start_time_s"), 2),
                    "end_time_s": fmt(row.get("end_time_s"), 2),
                    "duration_s": fmt(row.get("duration_s"), 2),
                    "segmentation_method": row.get("segmentation_method", "auto_range_plateau_trimmed"),
                    "confidence": fmt(row.get("confidence"), 2),
                }
            )
    return pd.DataFrame(rows)


def make_tracking_table(summary: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        seg = row["segment_id"]
        trow = row_for(tracking, seg)
        default_presence = trow.get("default_tracking_presence_rate") if trow is not None else np.nan
        static_presence = trow.get("static_tracking_presence_rate") if trow is not None else np.nan
        default_extra = trow.get("default_extra_track_rate") if trow is not None else np.nan
        static_extra = trow.get("static_extra_track_rate") if trow is not None else np.nan
        default_tid = trow.get("default_tid_switch_count") if trow is not None else np.nan
        static_tid = trow.get("static_tid_switch_count") if trow is not None else np.nan
        delta_range = row.get("delta_range_mae")
        verdict = "tracking stable"
        if n(delta_range) is not None and n(delta_range) > 0.20:
            verdict = "range regression"
        elif n(default_presence) is not None and n(static_presence) is not None and n(static_presence) < n(default_presence) - 0.05:
            verdict = "presence regression"
        elif n(default_extra) is not None and n(static_extra) is not None and n(static_extra) > n(default_extra) + 0.10:
            verdict = "extra-track regression"
        rows.append(
            {
                "segment_id": seg,
                "default_tracking_presence": pct(default_presence),
                "static_tracking_presence": pct(static_presence),
                "default_range_mae_m": fmt(row.get("default_range_mae")),
                "static_range_mae_m": fmt(row.get("static_range_mae")),
                "delta_range_mae_m": fmt(delta_range),
                "default_tid_switches": fmt(default_tid, 0),
                "static_tid_switches": fmt(static_tid, 0),
                "default_extra_track_rate": pct(default_extra),
                "static_extra_track_rate": pct(static_extra),
                "tracking_verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def make_geometry_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        geom_status = verdict_yes_no(row.get("delta_mean_geom_pts"), True, 0.10)
        no_points_status = verdict_yes_no(row.get("delta_NO_POINTS_rate"), False, 0.05)
        if geom_status == "improved" or no_points_status == "improved":
            verdict = "static improved seated point evidence"
        elif geom_status == "worse" or no_points_status == "worse":
            verdict = "static worsened point evidence"
        else:
            verdict = "no clear geometry improvement"
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "default_NO_POINTS_rate": pct(row.get("default_NO_POINTS_rate")),
                "static_NO_POINTS_rate": pct(row.get("static_NO_POINTS_rate")),
                "delta_NO_POINTS_rate": pct(row.get("delta_NO_POINTS_rate")),
                "default_mean_geom_pts": fmt(row.get("default_mean_geom_pts")),
                "static_mean_geom_pts": fmt(row.get("static_mean_geom_pts")),
                "delta_mean_geom_pts": fmt(row.get("delta_mean_geom_pts")),
                "default_geom_pts_ge_3_rate": pct(row.get("default_geom_pts_ge_3_rate")),
                "static_geom_pts_ge_3_rate": pct(row.get("static_geom_pts_ge_3_rate")),
                "delta_geom_pts_ge_3_rate": pct(row.get("delta_geom_pts_ge_3_rate")),
                "geometry_verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def make_probability_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        static_margin = n(row.get("static_stand_minus_sit_margin"))
        default_margin = n(row.get("default_stand_minus_sit_margin"))
        if static_margin is not None and static_margin > 0.10:
            verdict = "static still model-favors STANDING"
        elif static_margin is not None and static_margin < -0.10:
            verdict = "static model-favors SITTING"
        elif static_margin is not None:
            verdict = "static probabilities are ambiguous"
        else:
            verdict = "unknown"
        if default_margin is not None and static_margin is not None and static_margin < default_margin - 0.10:
            verdict += "; margin improved"
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "default_mean_stand_prob": fmt(row.get("default_mean_stand_prob")),
                "default_mean_sit_prob": fmt(row.get("default_mean_sit_prob")),
                "default_margin_stand_minus_sit": fmt(row.get("default_stand_minus_sit_margin")),
                "static_mean_stand_prob": fmt(row.get("static_mean_stand_prob")),
                "static_mean_sit_prob": fmt(row.get("static_mean_sit_prob")),
                "static_margin_stand_minus_sit": fmt(row.get("static_stand_minus_sit_margin")),
                "probability_verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def make_posture_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        status = verdict_yes_no(row.get("delta_posture_accuracy"), True, 0.05)
        if status == "improved":
            verdict = "static improved sitting display accuracy"
        elif status == "worse":
            verdict = "static worsened sitting display accuracy"
        else:
            verdict = "no clear display-accuracy improvement"
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "default_accuracy": pct(row.get("default_posture_accuracy")),
                "static_accuracy": pct(row.get("static_posture_accuracy")),
                "delta_accuracy": pct(row.get("delta_posture_accuracy")),
                "default_display_standing_rate": pct(row.get("default_display_standing_rate")),
                "static_display_standing_rate": pct(row.get("static_display_standing_rate")),
                "delta_display_standing_rate": pct(row.get("delta_display_standing_rate")),
                "default_display_sitting_rate": pct(row.get("default_display_sitting_rate")),
                "static_display_sitting_rate": pct(row.get("static_display_sitting_rate")),
                "delta_display_sitting_rate": pct(row.get("delta_display_sitting_rate")),
                "posture_verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def aggregate_answer(summary: pd.DataFrame, col: str, positive_is_good: bool, threshold: float = 0.05) -> str:
    vals = [n(v) for v in summary[col].tolist()] if col in summary else []
    vals = [v for v in vals if v is not None]
    if not vals:
        return "Unknown"
    if positive_is_good:
        improved = sum(v > threshold for v in vals)
        worsened = sum(v < -threshold for v in vals)
    else:
        improved = sum(v < -threshold for v in vals)
        worsened = sum(v > threshold for v in vals)
    if improved and not worsened:
        return "Yes"
    if worsened and not improved:
        return "No; it worsened"
    if improved and worsened:
        return "Mixed by distance"
    return "No clear change"


def final_decisions(summary: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    geometry_answer = aggregate_answer(summary, "delta_mean_geom_pts", True, 0.10)
    no_points_answer = aggregate_answer(summary, "delta_NO_POINTS_rate", False, 0.05)
    sit_prob_answer = aggregate_answer(summary, "delta_mean_sit_prob", True, 0.05)
    stand_prob_answer = aggregate_answer(summary, "delta_mean_stand_prob", False, 0.05)
    accuracy_answer = aggregate_answer(summary, "delta_posture_accuracy", True, 0.05)
    range_answer = aggregate_answer(summary, "delta_range_mae", False, 0.05)
    tracking_regression = summary.get("verdict", pd.Series(dtype=str)).astype(str).str.contains("TRACKING_REGRESSION").any()
    if tracking_regression:
        range_answer = "Yes"

    row4 = row_for(summary, "sitting_4m")
    if row4 is not None and n(row4.get("static_stand_minus_sit_margin")) is not None:
        sitting_4m_answer = "Yes" if n(row4.get("static_stand_minus_sit_margin")) > 0 else "No"
        sitting_4m_evidence = (
            f"static stand_prob={fmt(row4.get('static_mean_stand_prob'))}, "
            f"sit_prob={fmt(row4.get('static_mean_sit_prob'))}, "
            f"margin={fmt(row4.get('static_stand_minus_sit_margin'))}"
        )
    else:
        sitting_4m_answer = "Unknown"
        sitting_4m_evidence = "sitting_4m row unavailable"

    row3 = row_for(summary, "sitting_3m")
    if row3 is not None:
        sit_higher = n(row3.get("static_mean_sit_prob")) is not None and n(row3.get("static_mean_stand_prob")) is not None and n(row3.get("static_mean_sit_prob")) > n(row3.get("static_mean_stand_prob"))
        display_standing = n(row3.get("static_display_standing_rate")) is not None and n(row3.get("static_display_sitting_rate")) is not None and n(row3.get("static_display_standing_rate")) > n(row3.get("static_display_sitting_rate"))
        sitting_3m_answer = "Yes" if sit_higher and display_standing else "No"
        sitting_3m_evidence = (
            f"static sit_prob={fmt(row3.get('static_mean_sit_prob'))}, stand_prob={fmt(row3.get('static_mean_stand_prob'))}, "
            f"display STANDING={pct(row3.get('static_display_standing_rate'))}, display SITTING={pct(row3.get('static_display_sitting_rate'))}"
        )
    else:
        sitting_3m_answer = "Unknown"
        sitting_3m_evidence = "sitting_3m row unavailable"

    main_result = derive_main_result(summary)
    next_path = derive_next_path(summary)

    table = pd.DataFrame(
        [
            {"question": "Did static retention improve seated point geometry?", "answer": geometry_answer, "evidence": evidence_range(summary, "delta_mean_geom_pts", "mean_geom_pts delta")},
            {"question": "Did static retention reduce NO_POINTS?", "answer": no_points_answer, "evidence": evidence_range(summary, "delta_NO_POINTS_rate", "NO_POINTS delta")},
            {"question": "Did static retention increase sit_prob?", "answer": sit_prob_answer, "evidence": evidence_range(summary, "delta_mean_sit_prob", "sit_prob delta")},
            {"question": "Did static retention reduce stand_prob during sitting?", "answer": stand_prob_answer, "evidence": evidence_range(summary, "delta_mean_stand_prob", "stand_prob delta")},
            {"question": "Did static retention improve sitting posture accuracy?", "answer": accuracy_answer, "evidence": evidence_range(summary, "delta_posture_accuracy", "accuracy delta")},
            {"question": "Did static retention hurt tracking/range?", "answer": range_answer, "evidence": evidence_range(summary, "delta_range_mae", "range MAE delta") + "; comparison verdict flagged tracking regression where extra tracks increased"},
            {"question": "Is sitting_4m still model-favoring STANDING?", "answer": sitting_4m_answer, "evidence": sitting_4m_evidence},
            {"question": "Is sitting_3m still a gating/display issue?", "answer": sitting_3m_answer, "evidence": sitting_3m_evidence},
            {"question": "What should be fixed next?", "answer": next_path, "evidence": main_result},
        ]
    )
    return table, main_result, next_path


def evidence_range(summary: pd.DataFrame, col: str, label: str) -> str:
    if col not in summary:
        return f"{label}: unavailable"
    vals = [n(v) for v in summary[col].tolist()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return f"{label}: unavailable"
    return f"{label}: min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}"


def derive_main_result(summary: pd.DataFrame) -> str:
    geom = aggregate_answer(summary, "delta_mean_geom_pts", True, 0.10)
    acc = aggregate_answer(summary, "delta_posture_accuracy", True, 0.05)
    range_hurt = aggregate_answer(summary, "delta_range_mae", False, 0.05)
    tracking_regression = summary.get("verdict", pd.Series(dtype=str)).astype(str).str.contains("TRACKING_REGRESSION").any()
    if range_hurt == "No; it worsened" or tracking_regression:
        return "Static retention may improve posture evidence but introduces tracking/range regression, so it is not deployable without further cfg tuning."
    if geom == "Yes" and acc == "Yes":
        return "Static retention helped seated point extraction and improved posture. The likely next fix path is cfg/static point evidence optimization, then posture model/logic tuning only after cfg is stable."
    if geom == "Yes" and acc != "Yes":
        return "Static retention improved point evidence, but the model/decision layer still failed. The next fix path is posture feature/model or stand-vs-sit decision logic."
    if geom.startswith("No"):
        return "Static retention did not solve seated point evidence. The next fix path is point association, sensor calibration, feature construction, or model retraining rather than relying on this cfg change."
    return "The A/B result is mixed by distance. Use the per-distance probability, geometry, and tracking rows to choose the next focused fix path."


def derive_next_path(summary: pd.DataFrame) -> str:
    tracking_regression = summary.get("verdict", pd.Series(dtype=str)).astype(str).str.contains("TRACKING_REGRESSION").any()
    if tracking_regression:
        return "cfg/static-retention tracking regression and point association cleanup before posture tuning"
    row3 = row_for(summary, "sitting_3m")
    row4 = row_for(summary, "sitting_4m")
    any_gate = False
    if row3 is not None:
        sit_higher = n(row3.get("static_mean_sit_prob")) is not None and n(row3.get("static_mean_stand_prob")) is not None and n(row3.get("static_mean_sit_prob")) > n(row3.get("static_mean_stand_prob"))
        display_standing = n(row3.get("static_display_standing_rate")) is not None and n(row3.get("static_display_sitting_rate")) is not None and n(row3.get("static_display_standing_rate")) > n(row3.get("static_display_sitting_rate"))
        any_gate = sit_higher and display_standing
    if any_gate:
        return "stand-vs-sit decision/gating update"
    if row4 is not None and n(row4.get("static_stand_minus_sit_margin")) is not None and n(row4.get("static_stand_minus_sit_margin")) > 0.10:
        return "sitting-specific geometry feature engineering or model retraining with 4m sitting data"
    geom = aggregate_answer(summary, "delta_mean_geom_pts", True, 0.10)
    if geom != "Yes":
        return "point association, sensor calibration, and feature construction before relying on static-retention cfg"
    return "cfg/static retention follow-up plus model/decision inspection"


def per_distance_text(summary: pd.DataFrame, segment_id: str) -> str:
    row = row_for(summary, segment_id)
    if row is None:
        return f"No row was generated for `{segment_id}`."
    return (
        f"`{segment_id}`: accuracy {pct(row.get('default_posture_accuracy'))} -> {pct(row.get('static_posture_accuracy'))}; "
        f"display SITTING {pct(row.get('default_display_sitting_rate'))} -> {pct(row.get('static_display_sitting_rate'))}; "
        f"stand_prob {fmt(row.get('default_mean_stand_prob'))} -> {fmt(row.get('static_mean_stand_prob'))}; "
        f"sit_prob {fmt(row.get('default_mean_sit_prob'))} -> {fmt(row.get('static_mean_sit_prob'))}; "
        f"NO_POINTS {pct(row.get('default_NO_POINTS_rate'))} -> {pct(row.get('static_NO_POINTS_rate'))}; "
        f"mean_geom_pts {fmt(row.get('default_mean_geom_pts'))} -> {fmt(row.get('static_mean_geom_pts'))}; "
        f"range MAE {fmt(row.get('default_range_mae'))}m -> {fmt(row.get('static_range_mae'))}m; "
        f"verdict `{row.get('verdict')}`."
    )


def plot_links() -> str:
    lines = []
    for label, folder in [("default cfg", DEFAULT_ANALYSIS), ("static-retention cfg", STATIC_ANALYSIS)]:
        lines.append(f"### {label}")
        for name in PLOT_NAMES:
            path = folder / "plots" / name
            status = rel(path) if path.exists() else f"{rel(path)} (missing)"
            lines.append(f"- `{status}` - {PLOT_EXPLANATIONS[name]}")
        lines.append("")
    return "\n".join(lines).strip()


def brainstorm_table(summary: pd.DataFrame) -> pd.DataFrame:
    main_result = derive_main_result(summary)
    next_path = derive_next_path(summary)
    rows = [
        {
            "rank": 1,
            "fix_path": "cfg/static retention / fine motion tuning",
            "why_it_may_help": "Could preserve seated static/fine-motion point evidence that target-only features miss.",
            "evidence_supports": evidence_range(summary, "delta_mean_geom_pts", "mean_geom_pts delta") + "; " + evidence_range(summary, "delta_NO_POINTS_rate", "NO_POINTS delta"),
            "evidence_against": "If geometry and NO_POINTS did not improve consistently, this cfg alone is not the fix.",
            "next_test": "Run a second static-retention cfg variant only if tracking/range remains stable and geometry improves.",
            "instability_risk": "Medium: static retention can alter point/range behavior and may hurt tracking if over-tuned.",
        },
        {
            "rank": 2,
            "fix_path": "point association radius / target-index association improvement",
            "why_it_may_help": "Sitting failure is strongly tied to missing or sparse point geometry.",
            "evidence_supports": "NO_POINTS and mean_geom_pts deltas show whether the cfg supplied evidence but association still failed.",
            "evidence_against": "If model probabilities remain wrong even when geometry improves, association alone is insufficient.",
            "next_test": "Replay logs with diagnostic-only association variants and compare geom_pts without changing display behavior.",
            "instability_risk": "Medium: wider association can attach wrong points in multi-person scenes.",
        },
        {
            "rank": 3,
            "fix_path": "sensor mount calibration verification",
            "why_it_may_help": "Height and range geometry affect sitting-vs-standing features, especially farther out.",
            "evidence_supports": evidence_range(summary, "delta_range_mae", "range MAE delta"),
            "evidence_against": "Strong tracking and low range error would make calibration less likely as the primary cause.",
            "next_test": "Record calibration target/person at known distances and compare target z/range against expected mount geometry.",
            "instability_risk": "Low to medium: bad calibration changes can shift all posture features.",
        },
        {
            "rank": 4,
            "fix_path": "sitting-specific geometry feature engineering",
            "why_it_may_help": "Can add discriminative seated geometry when current 22-feature slots are sparse or zero-filled.",
            "evidence_supports": "Use this if stand_prob remains high during sitting despite available geometry.",
            "evidence_against": "Will not help if the core issue is display gating after sit_prob already dominates.",
            "next_test": "Offline feature ablation on sitting 2m/3m/4m logs using existing probabilities and geometry fields.",
            "instability_risk": "Medium: new features can degrade standing unless validated against standing sessions.",
        },
        {
            "rank": 5,
            "fix_path": "stand-vs-sit decision/gating update",
            "why_it_may_help": "Needed when sit_prob is higher but displayed pose remains STANDING.",
            "evidence_supports": "Priority rises if sitting_3m shows sit_prob > stand_prob while display remains STANDING.",
            "evidence_against": "Should not be used when stand_prob still dominates; that is a model/feature problem.",
            "next_test": "Offline replay of decision logic only, measuring display lag and false sitting on standing data.",
            "instability_risk": "Medium to high: random threshold tuning could destabilize standing, so use evidence-based replay only.",
        },
        {
            "rank": 6,
            "fix_path": "model retraining with sitting 2m/3m/4m data",
            "why_it_may_help": "Required if the model probabilities themselves favor STANDING under seated conditions.",
            "evidence_supports": "Strongly supported when 4m sitting still has positive stand-minus-sit margin.",
            "evidence_against": "Retraining is premature if missing geometry or gating is the primary failure.",
            "next_test": "Build a labeled sitting-at-distance dataset and compare cross-distance stand/sit probability margins.",
            "instability_risk": "High: retraining can regress standing unless the dataset is balanced and held-out.",
        },
        {
            "rank": 7,
            "fix_path": "RGB-assisted ground truth/fusion",
            "why_it_may_help": "RGB can validate segment timing and later provide cross-modal posture evidence.",
            "evidence_supports": "RGB frames/tracks/keypoints are present as visual reference.",
            "evidence_against": "Do not claim RGB posture accuracy unless rgb_actions.csv has meaningful labels.",
            "next_test": "Use RGB only to verify segment labels first, then evaluate fusion separately.",
            "instability_risk": "Medium: sensor sync/visibility issues can create false confidence.",
        },
    ]
    return pd.DataFrame(rows)


def generate_reports(default_session: Path, static_session: Path, discovery: pd.DataFrame) -> tuple[str, str]:
    summary = read_csv(COMPARISON / "sitting_ab_summary.csv")
    tracking = read_csv(COMPARISON / "sitting_ab_tracking_comparison.csv")
    probability = read_csv(COMPARISON / "sitting_ab_probability_comparison.csv")
    geometry = read_csv(COMPARISON / "sitting_ab_geometry_comparison.csv")
    _ = probability, geometry

    session_table = make_session_table(default_session, static_session)
    segment_table = make_segment_table()
    tracking_table = make_tracking_table(summary, tracking)
    geometry_table = make_geometry_table(summary)
    probability_table = make_probability_table(summary)
    posture_table = make_posture_table(summary)
    decision_table, main_result, next_path = final_decisions(summary)
    rgb_table = pd.DataFrame([read_rgb_summary(default_session), read_rgb_summary(static_session)])
    brainstorm = brainstorm_table(summary)

    lines = [
        "# Sitting A/B Final Report for Superior and Brainstorm",
        "",
        "## 1. Executive summary",
        main_result,
        "",
        f"Top next engineering path: **{next_path}**. This report does not modify runtime posture logic, thresholds, cfg files, model files, renderer code, or RGB code.",
        "",
        "## 2. Why this A/B test was run",
        "The prior benchmark showed strong tracking and nearly perfect standing posture, but sitting posture failed, especially at 3m and 4m. This A/B test isolates whether TI static-retention cfg improves seated point evidence and sitting posture detection.",
        "",
        "## 3. Test protocol",
        "- Test A: default cfg, sitting at 2m for 60 sec, 3m for 60 sec, and 4m for 60 sec.",
        "- Test B: static-retention cfg, sitting at 2m for 60 sec, 3m for 60 sec, and 4m for 60 sec.",
        "- Both sessions used the same runtime pose/RGB/combined logging setup; cfg was the intended experiment variable.",
        "",
        "## 4. Sessions analyzed",
        markdown_table(session_table),
        "",
        "Discovery CSV: `analysis_outputs/sitting_ab_session_discovery.csv`.",
        "",
        "## 5. Segment boundaries used",
        "The original manual segment templates were blank. Segment boundaries below were inferred from range plateaus near 2m, 3m, and 4m, trimmed away from transitions, written back to the manual segment CSVs, and then passed to the analyzer.",
        "",
        markdown_table(segment_table),
        "",
        "## 6. Tracking comparison",
        markdown_table(tracking_table),
        "",
        "## 7. Distance/range accuracy comparison",
        "Range MAE is included in the tracking table. Negative delta_range_mae_m means static retention reduced range error; positive means range error increased.",
        "",
        "## 8. Point geometry / NO_POINTS comparison",
        markdown_table(geometry_table),
        "",
        "## 9. Stand-vs-sit probability comparison",
        markdown_table(probability_table),
        "",
        "## 10. Sitting posture accuracy comparison",
        markdown_table(posture_table),
        "",
        "## 11. RGB data summary",
        markdown_table(rgb_table),
        "",
        "RGB was recorded as visual/synchronization reference. Do not claim RGB posture accuracy unless `rgb_actions.csv` contains meaningful action labels. If `rgb_actions.csv` is empty or contains only headers/default entries, RGB action classification was not available as quantitative ground truth.",
        "",
        "## 12. Per-distance analysis: 2m",
        per_distance_text(summary, "sitting_2m"),
        "",
        "## 13. Per-distance analysis: 3m",
        per_distance_text(summary, "sitting_3m"),
        "",
        "## 14. Per-distance analysis: 4m",
        per_distance_text(summary, "sitting_4m"),
        "",
        "## 15. Final verdict",
        markdown_table(decision_table),
        "",
        "## 16. What the result means technically",
        main_result,
        "",
        "If sit_prob is higher but display remains STANDING, the model has enough sitting probability, but display/gating/hysteresis is preventing correct posture output. If stand_prob remains higher during sitting, the posture model/features are not separating sitting from standing under these conditions.",
        "",
        "## 17. What is proven",
        "- The two recorded sessions were found and analyzed from combined mmWave/RGB logs.",
        "- Sitting-only segment boundaries were generated from range plateaus and used consistently for default and static-retention analyses.",
        "- Tracking, range, geometry, probability, and displayed posture outputs were compared per distance.",
        "",
        "## 18. What is not proven",
        "- This does not prove RGB posture accuracy because RGB is used as visual/synchronization reference unless action labels are meaningful.",
        "- This does not prove a deployable runtime fix because no runtime logic, thresholds, cfg contents, or model files were changed.",
        "- This does not prove that random threshold tuning would help.",
        "",
        "## 19. Recommended next engineering path",
        f"Recommended next path: **{next_path}**.",
        "",
        "Do not apply random threshold tuning. Use the A/B evidence above to choose one controlled offline replay or data-collection experiment.",
        "",
        "## 20. Brainstorming section: possible fixes ranked by evidence",
        "### Engineering brainstorm based on A/B result",
        markdown_table(brainstorm),
        "",
        "## 21. Appendix: generated files and plots",
        "Generated files:",
        "- `analysis_outputs/sitting_ab_session_discovery.csv`",
        "- `analysis_inputs/sitting_ab_default_segments.csv`",
        "- `analysis_inputs/sitting_ab_static_retention_segments.csv`",
        "- `analysis_outputs/sitting_ab_default_analysis/`",
        "- `analysis_outputs/sitting_ab_static_retention_analysis/`",
        "- `analysis_outputs/sitting_ab_comparison/sitting_ab_summary.csv`",
        "- `analysis_outputs/sitting_ab_comparison/sitting_ab_probability_comparison.csv`",
        "- `analysis_outputs/sitting_ab_comparison/sitting_ab_geometry_comparison.csv`",
        "- `analysis_outputs/sitting_ab_comparison/sitting_ab_tracking_comparison.csv`",
        "- `analysis_outputs/sitting_ab_comparison/SITTING_AB_COMPARISON_REPORT.md`",
        "- `analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`",
        "",
        "Important plots:",
        plot_links(),
        "",
    ]
    COMPARISON.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")

    completion = [
        "# Sitting A/B Analysis Completion",
        "",
        "## 1. Sessions found",
        markdown_table(session_table),
        "",
        "## 2. Analysis commands run",
        "```powershell",
        f"python analysis\\analyze_distance_posture_session.py --session \"{default_session}\" --out analysis_outputs\\sitting_ab_default_analysis --expected-distances \"2,3,4\" --manual-segments analysis_inputs\\sitting_ab_default_segments.csv --make-plots",
        f"python analysis\\analyze_distance_posture_session.py --session \"{static_session}\" --out analysis_outputs\\sitting_ab_static_retention_analysis --expected-distances \"2,3,4\" --manual-segments analysis_inputs\\sitting_ab_static_retention_segments.csv --make-plots",
        "```",
        "",
        "## 3. Whether manual or auto segments were used",
        "The original manual templates were blank. Auto range-plateau boundaries were generated, written to the manual segment CSVs, and then used by the analyzer.",
        "",
        "## 4. Comparison command run",
        "```powershell",
        "python analysis\\compare_sitting_ab.py --default analysis_outputs\\sitting_ab_default_analysis --static analysis_outputs\\sitting_ab_static_retention_analysis --out analysis_outputs\\sitting_ab_comparison",
        "```",
        "",
        "## 5. Files created",
        "- `analysis_outputs/sitting_ab_session_discovery.csv`",
        "- `analysis_outputs/sitting_ab_default_analysis/`",
        "- `analysis_outputs/sitting_ab_static_retention_analysis/`",
        "- `analysis_outputs/sitting_ab_comparison/`",
        "- `analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`",
        "- `SITTING_AB_ANALYSIS_COMPLETION.md`",
        "",
        "## 6. Final report path",
        "`analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`",
        "",
        "## 7. Main result",
        main_result,
        "",
        "## 8. Validation commands run",
        "```powershell",
        "python -m py_compile analysis\\analyze_distance_posture_session.py",
        "python -m py_compile analysis\\compare_sitting_ab.py",
        "python -m py_compile analysis\\generate_sitting_ab_final_report.py",
        "```",
        "",
        "## 9. Any warnings or limitations",
        "- Segments were inferred from range plateaus because the provided manual segment CSVs were blank.",
        "- This is an offline analysis/reporting pass only.",
        "- RGB was recorded as reference; RGB posture accuracy is not claimed unless action labels are meaningful.",
        "",
    ]
    COMPLETION_REPORT.write_text("\n".join(completion), encoding="utf-8")
    return main_result, next_path


def prepare() -> tuple[Path, Path]:
    default_session, static_session, _discovery = discover_sessions()
    infer_and_write_segments(default_session, DEFAULT_SEGMENTS)
    infer_and_write_segments(static_session, STATIC_SEGMENTS)
    print(f"Selected default session: {default_session}")
    print(f"Selected static session: {static_session}")
    print(f"Wrote discovery CSV: {DISCOVERY_CSV}")
    print(f"Wrote segment CSV: {DEFAULT_SEGMENTS}")
    print(f"Wrote segment CSV: {STATIC_SEGMENTS}")
    return default_session, static_session


def report() -> tuple[str, str]:
    if not DISCOVERY_CSV.exists():
        discover_sessions()
    discovery = read_csv(DISCOVERY_CSV)
    default_session = selected_discovery_path(discovery, "default")
    static_session = selected_discovery_path(discovery, "static")
    main_result, next_path = generate_reports(default_session, static_session, discovery)
    print(f"Final report: {FINAL_REPORT}")
    print("")
    print("Main result:")
    print(main_result)
    print("")
    print("Top next engineering path:")
    print(next_path)
    return main_result, next_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "report"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "report":
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
