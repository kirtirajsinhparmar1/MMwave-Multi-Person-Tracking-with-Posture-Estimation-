#!/usr/bin/env python
"""Clean and label posture sessions for the bounded RadarPostureNet-v2 pass."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posturenet_v2_common import (
    REPO_ROOT,
    SEGMENT_FIELDS,
    confidence_rank,
    ensure_dir,
    first_existing,
    normalized_label,
    read_csv_rows,
    safe_div,
    to_float,
    to_int,
    write_csv,
)


LEGACY_SEGMENT_FILES = {
    "session_20260703_205540": REPO_ROOT / "analysis_inputs" / "session_20260703_205540_segments.csv",
    "sitting_ab_default_cfg": REPO_ROOT / "analysis_inputs" / "sitting_ab_default_segments_corrected_1to4.csv",
    "sitting_ab_static_retention_cfg": REPO_ROOT / "analysis_inputs" / "sitting_ab_static_retention_segments_corrected_1to4.csv",
    "sitting_relative_gate_refined_live_test": REPO_ROOT / "analysis_inputs" / "sitting_relative_gate_live_segments.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--segments-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def add_time_s(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time_s" in df.columns:
        return df
    df = df.copy()
    if "host_monotonic_ns" in df.columns:
        ns = pd.to_numeric(df["host_monotonic_ns"], errors="coerce")
        first = ns.dropna().min()
        df["time_s"] = (ns - first) / 1_000_000_000.0 if pd.notna(first) else np.nan
    elif "time" in df.columns:
        numeric = pd.to_numeric(df["time"], errors="coerce")
        if numeric.notna().mean() > 0.9:
            first = numeric.dropna().min()
            df["time_s"] = numeric - first
        else:
            parsed = pd.to_datetime(df["time"], errors="coerce", utc=True)
            first = parsed.dropna().min()
            df["time_s"] = (parsed - first).dt.total_seconds() if pd.notna(first) else np.nan
    elif "host_wall_time_iso" in df.columns:
        parsed = pd.to_datetime(df["host_wall_time_iso"], errors="coerce", utc=True)
        first = parsed.dropna().min()
        df["time_s"] = (parsed - first).dt.total_seconds() if pd.notna(first) else np.nan
    elif "frame" in df.columns:
        df["time_s"] = pd.to_numeric(df["frame"], errors="coerce") / 20.0
    elif "mmwave_frame_num" in df.columns:
        df["time_s"] = pd.to_numeric(df["mmwave_frame_num"], errors="coerce") / 20.0
    else:
        df["time_s"] = np.nan
    return df


def standardize_tid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "tid" not in df.columns:
        for candidate in ["track_id", "target_id", "track_index"]:
            if candidate in df.columns:
                df["tid"] = df[candidate]
                break
    if "tid" in df.columns:
        df["tid"] = pd.to_numeric(df["tid"], errors="coerce")
    if "mmwave_frame_num" not in df.columns and "frame" in df.columns:
        df["mmwave_frame_num"] = df["frame"]
    return df


def add_track_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    x_col = "x_m" if "x_m" in df.columns else "x" if "x" in df.columns else None
    y_col = "y_m" if "y_m" in df.columns else "y" if "y" in df.columns else None
    z_col = "z_m" if "z_m" in df.columns else "z" if "z" in df.columns else None
    if x_col:
        df["lateral_x_m"] = pd.to_numeric(df[x_col], errors="coerce")
    else:
        df["lateral_x_m"] = np.nan
    if z_col:
        df["target_z_m"] = pd.to_numeric(df[z_col], errors="coerce")
    else:
        df["target_z_m"] = np.nan
    if x_col and y_col:
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        df["range_m"] = np.sqrt(x * x + y * y)
    elif "range_m" in df.columns:
        df["range_m"] = pd.to_numeric(df["range_m"], errors="coerce")
    else:
        df["range_m"] = np.nan
    v_cols = []
    for name in ["vx_mps", "vy_mps", "vz_mps", "vx", "vy", "vz"]:
        if name in df.columns:
            v_cols.append(name)
    if {"vx_mps", "vy_mps", "vz_mps"}.issubset(df.columns):
        df["speed_mps_calc"] = np.sqrt(
            pd.to_numeric(df["vx_mps"], errors="coerce") ** 2
            + pd.to_numeric(df["vy_mps"], errors="coerce") ** 2
            + pd.to_numeric(df["vz_mps"], errors="coerce") ** 2
        )
    elif {"vx", "vy", "vz"}.issubset(df.columns):
        df["speed_mps_calc"] = np.sqrt(
            pd.to_numeric(df["vx"], errors="coerce") ** 2
            + pd.to_numeric(df["vy"], errors="coerce") ** 2
            + pd.to_numeric(df["vz"], errors="coerce") ** 2
        )
    else:
        df["speed_mps_calc"] = np.nan
    return df


def load_session_tables(session_path: Path) -> dict[str, pd.DataFrame]:
    pose_path = first_existing(session_path, ["mmwave_pose.csv", "pose_predictions_ui.csv"])
    tracks_path = first_existing(session_path, ["mmwave_tracks.csv", "targets.csv"])
    frames_path = first_existing(session_path, ["mmwave_frames.csv", "frames_summary.csv"])
    pose = standardize_tid(add_time_s(read_table(pose_path)))
    tracks = add_track_geometry(standardize_tid(add_time_s(read_table(tracks_path))))
    frames = add_time_s(read_table(frames_path))
    return {"pose": pose, "tracks": tracks, "frames": frames}


def load_legacy_times(session_id: str) -> dict[str, dict[str, Any]]:
    path = LEGACY_SEGMENT_FILES.get(session_id)
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        sid = str(row.get("segment_id") or "")
        if sid:
            out[sid] = row
    return out


def confidence_from_legacy(value: Any, duration: float) -> str:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    if duration < 30:
        return "LOW"
    if number >= 0.8 and duration >= 40:
        return "HIGH"
    if number >= 0.5:
        return "MEDIUM"
    return "LOW"


def grouped_segments(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for row in rows:
        key = (
            row["segment_id"],
            row["expected_pose"],
            row["expected_subpose"],
            row["expected_distance_m"],
        )
        if key in seen:
            continue
        seen.add(key)
        groups.append({"key": key, "rows": [r for r in rows if (r["segment_id"], r["expected_pose"], r["expected_subpose"], r["expected_distance_m"]) == key]})
    return groups


def fill_segment_times(session_id: str, segment_rows: list[dict[str, str]], tables: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rows = [dict(r) for r in segment_rows]
    legacy = load_legacy_times(session_id)
    groups = grouped_segments(rows)
    time_source = tables["pose"] if not tables["pose"].empty else tables["tracks"]
    valid_times = pd.to_numeric(time_source.get("time_s", pd.Series(dtype=float)), errors="coerce").dropna()
    if valid_times.empty:
        session_start, session_end = 0.0, float(len(groups) * 45)
    else:
        session_start, session_end = float(valid_times.min()), float(valid_times.max())
    usable_start = session_start + 5.0
    usable_end = max(usable_start + len(groups) * 10.0, session_end - 5.0)
    fallback_span = max(1.0, usable_end - usable_start)
    fallback_width = fallback_span / max(1, len(groups))
    group_times: dict[tuple, tuple[float, float, str, str]] = {}

    for idx, group in enumerate(groups):
        segment_id = group["key"][0]
        if segment_id in legacy:
            legacy_row = legacy[segment_id]
            start = to_float(legacy_row.get("start_time_s"), math.nan)
            end = to_float(legacy_row.get("end_time_s"), math.nan)
            if math.isnan(start) or math.isnan(end) or end <= start:
                start = usable_start + idx * fallback_width + 3.0
                end = usable_start + (idx + 1) * fallback_width - 3.0
                source = "time_order_fallback_after_invalid_legacy"
            else:
                source = "legacy_segment_time"
            duration = max(0.0, end - start)
            label_conf = confidence_from_legacy(legacy_row.get("label_confidence"), duration)
            notes = str(legacy_row.get("notes") or "")
        else:
            start = usable_start + idx * fallback_width
            end = usable_start + (idx + 1) * fallback_width
            if fallback_width > 12:
                start += 3.0
                end -= 3.0
            duration = max(0.0, end - start)
            label_conf = "HIGH" if duration >= 40 else "MEDIUM" if duration >= 30 else "LOW"
            if label_conf == "HIGH":
                label_conf = "MEDIUM"
            source = "protocol_order_time_fallback"
            notes = "start/end inferred from protocol order; no manual segment timing file"
        group_times[group["key"]] = (round(start, 3), round(end, 3), label_conf, f"{source}; {notes}".strip("; "))

    for row in rows:
        key = (row["segment_id"], row["expected_pose"], row["expected_subpose"], row["expected_distance_m"])
        start, end, label_conf, timing_notes = group_times[key]
        row["start_time_s"] = f"{start:.3f}"
        row["end_time_s"] = f"{end:.3f}"
        if confidence_rank(label_conf) < confidence_rank(str(row.get("label_confidence") or "")) or not row.get("label_confidence"):
            row["label_confidence"] = label_conf
        row["notes"] = "; ".join(part for part in [str(row.get("notes") or ""), timing_notes] if part)
    return rows


def rows_in_segment(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    times = pd.to_numeric(df["time_s"], errors="coerce")
    return df[(times >= start) & (times <= end)].copy()


def dominant_tid(df: pd.DataFrame) -> tuple[int | None, str, str]:
    if df.empty or "tid" not in df.columns:
        return None, "LOW", "no track/pose rows with tid"
    tids = pd.to_numeric(df["tid"], errors="coerce").dropna().astype(int)
    if tids.empty:
        return None, "LOW", "no valid tid"
    counts = Counter(tids.tolist())
    tid, count = counts.most_common(1)[0]
    ratio = count / max(1, sum(counts.values()))
    conf = "HIGH" if ratio >= 0.75 else "MEDIUM" if ratio >= 0.5 else "LOW"
    return tid, conf, f"dominant_tid_ratio={ratio:.3f}; tid_counts={dict(counts)}"


def assign_two_person(group_rows: list[dict[str, str]], tracks: pd.DataFrame) -> dict[str, tuple[int | None, str, str]]:
    tids = []
    for tid_value, tid_df in tracks.groupby("tid"):
        if pd.isna(tid_value):
            continue
        tids.append(
            {
                "tid": int(tid_value),
                "count": len(tid_df),
                "mean_x": float(pd.to_numeric(tid_df.get("lateral_x_m"), errors="coerce").mean()),
            }
        )
    tids.sort(key=lambda item: item["count"], reverse=True)
    if len(tids) < 2:
        tid, conf, note = dominant_tid(tracks)
        return {
            "LEFT": (tid, "LOW", f"two-person assignment has fewer than two tids; {note}"),
            "RIGHT": (None, "LOW", "two-person assignment has fewer than two tids"),
        }
    selected = sorted(tids[:2], key=lambda item: item["mean_x"])
    separation = abs(selected[1]["mean_x"] - selected[0]["mean_x"])
    conf = "HIGH" if separation >= 0.75 else "MEDIUM" if separation >= 0.35 else "LOW"
    note = (
        "left/right inferred from verified lateral ordering within this segment; "
        f"mean_x_left={selected[0]['mean_x']:.3f}; mean_x_right={selected[1]['mean_x']:.3f}; "
        f"separation={separation:.3f}"
    )
    return {
        "LEFT": (selected[0]["tid"], conf, note),
        "RIGHT": (selected[1]["tid"], conf, note),
    }


def assign_tids(rows: list[dict[str, str]], tables: dict[str, pd.DataFrame]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    out = [dict(r) for r in rows]
    debug: list[dict[str, Any]] = []
    tracks_all = tables["tracks"] if not tables["tracks"].empty else tables["pose"]
    for group in grouped_segments(out):
        start = to_float(group["rows"][0]["start_time_s"], 0.0)
        end = to_float(group["rows"][0]["end_time_s"], 0.0)
        tracks = rows_in_segment(tracks_all, start, end)
        if len(group["rows"]) > 1:
            assignments = assign_two_person(group["rows"], tracks)
            for row in out:
                key = (row["segment_id"], row["expected_pose"], row["expected_subpose"], row["expected_distance_m"])
                if key != group["key"]:
                    continue
                position = row["expected_position"].upper()
                tid, conf, note = assignments.get(position, (None, "LOW", "unknown two-person slot"))
                row["assigned_tid"] = "" if tid is None else str(tid)
                row["assignment_confidence"] = conf
                row["notes"] = "; ".join([row.get("notes", ""), note]).strip("; ")
                debug.append({**row, "assignment_note": note, "segment_track_rows": len(tracks)})
        else:
            tid, conf, note = dominant_tid(tracks)
            row = group["rows"][0]
            for target in out:
                if target is row or (
                    target["segment_id"],
                    target["expected_pose"],
                    target["expected_subpose"],
                    target["expected_distance_m"],
                ) == group["key"]:
                    target["assigned_tid"] = "" if tid is None else str(tid)
                    target["assignment_confidence"] = conf
                    target["notes"] = "; ".join([target.get("notes", ""), note]).strip("; ")
                    debug.append({**target, "assignment_note": note, "segment_track_rows": len(tracks)})
    return out, debug


def quality_for_segment(row: dict[str, str], tables: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = to_float(row.get("start_time_s"), 0.0)
    end = to_float(row.get("end_time_s"), 0.0)
    duration = max(0.0, end - start)
    tid = to_int(row.get("assigned_tid"))
    frames = rows_in_segment(tables["frames"], start, end)
    expected_frames = len(frames) if not frames.empty else max(1, int(duration * 18.0))
    track_seg = rows_in_segment(tables["tracks"], start, end)
    pose_seg = rows_in_segment(tables["pose"], start, end)
    if tid is not None:
        track_tid = track_seg[pd.to_numeric(track_seg.get("tid"), errors="coerce") == tid] if "tid" in track_seg.columns else pd.DataFrame()
        pose_tid = pose_seg[pd.to_numeric(pose_seg.get("tid"), errors="coerce") == tid] if "tid" in pose_seg.columns else pd.DataFrame()
    else:
        track_tid = pd.DataFrame()
        pose_tid = pd.DataFrame()
    track_frames = track_tid.get("mmwave_frame_num", pd.Series(dtype=float)).nunique() if not track_tid.empty else 0
    pose_frames = pose_tid.get("mmwave_frame_num", pd.Series(dtype=float)).nunique() if not pose_tid.empty else len(pose_tid)
    tracking_presence = safe_div(float(track_frames), float(expected_frames))
    pose_presence = safe_div(float(pose_frames), float(expected_frames))
    quality_col = None
    for candidate in ["quality_flag", "quality", "geom_quality"]:
        if candidate in pose_tid.columns:
            quality_col = candidate
            break
    quality_values = pose_tid[quality_col].astype(str).str.upper() if quality_col else pd.Series(dtype=str)
    no_points_rate = safe_div(float(quality_values.str.contains("NO_POINTS").sum()), max(1.0, float(len(quality_values))))
    low_points_rate = safe_div(float(quality_values.str.contains("LOW_POINTS|LOW QUALITY|LOW_QUALITY").sum()), max(1.0, float(len(quality_values))))
    ok_rate = safe_div(float(quality_values.str.contains("OK|POINT_GEOMETRY|GOOD").sum()), max(1.0, float(len(quality_values))))
    if "num_points" in pose_tid.columns:
        points = pd.to_numeric(pose_tid["num_points"], errors="coerce")
    elif "geom_pts" in pose_tid.columns:
        points = pd.to_numeric(pose_tid["geom_pts"], errors="coerce")
    else:
        points = pd.Series(dtype=float)
    low_points_numeric = safe_div(float((points < 5).sum()), max(1.0, float(points.notna().sum()))) if not points.empty else 0.0
    low_points_rate = max(low_points_rate, low_points_numeric)
    pose_times = pd.to_numeric(pose_tid.get("time_s", pd.Series(dtype=float)), errors="coerce").dropna().sort_values()
    gap_count = int((pose_times.diff() > 1.0).sum()) if len(pose_times) > 1 else 0
    tid_counts = Counter(pd.to_numeric(track_seg.get("tid", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()) if not track_seg.empty and "tid" in track_seg.columns else Counter()
    tid_switch_count = max(0, len([count for count in tid_counts.values() if count >= max(5, 0.1 * sum(tid_counts.values()))]) - 1)
    if "range_m" in track_tid.columns:
        ranges = pd.to_numeric(track_tid["range_m"], errors="coerce").dropna().sort_index()
        range_jump_count = int((ranges.diff().abs() > 1.0).sum()) if len(ranges) > 1 else 0
    else:
        range_jump_count = 0
    display_cols = {"display_status", "final_display_pose", "displayed_label"} & set(pose_tid.columns)
    ui_visible_rate = 0.0
    render_confirmed = False
    if display_cols:
        display_col = sorted(display_cols)[0]
        labels = pose_tid[display_col].map(normalized_label)
        ui_visible_rate = safe_div(float((labels != "UNKNOWN").sum()), max(1.0, float(len(labels))))
        render_confirmed = True

    label_conf = str(row.get("label_confidence") or "")
    assignment_conf = str(row.get("assignment_confidence") or "")
    combined_conf = min(confidence_rank(label_conf), confidence_rank(assignment_conf))
    quality_level = "HIGH" if combined_conf >= 3 and duration >= 40 and tracking_presence >= 0.7 else "MEDIUM"
    if duration < 30 or combined_conf <= 1 or tracking_presence < 0.3:
        quality_level = "LOW"

    summary = {
        **row,
        "duration_s": round(duration, 3),
        "expected_frames": expected_frames,
        "track_frames": track_frames,
        "pose_frames": pose_frames,
        "tracking_presence_rate": round(tracking_presence, 4),
        "pose_presence_rate": round(pose_presence, 4),
        "NO_POINTS_rate": round(no_points_rate, 4),
        "LOW_POINTS_rate": round(low_points_rate, 4),
        "OK_rate": round(ok_rate, 4),
        "gap_count_gt_1s": gap_count,
        "tid_switch_count": tid_switch_count,
        "range_jump_count": range_jump_count,
        "ui_visible_rate": round(ui_visible_rate, 4),
        "render_confirmed": render_confirmed,
        "segment_quality": quality_level,
    }
    events: list[dict[str, Any]] = []

    def add_event(event_type: str, severity: str, detail: str) -> None:
        events.append(
            {
                "session_id": row["session_id"],
                "segment_id": row["segment_id"],
                "person_slot": row["person_slot"],
                "assigned_tid": row.get("assigned_tid", ""),
                "event_type": event_type,
                "severity": severity,
                "start_time_s": row["start_time_s"],
                "end_time_s": row["end_time_s"],
                "detail": detail,
            }
        )

    if tid is None:
        add_event("ASSIGNMENT_MISSING", "HIGH", "no assigned TID")
    if tracking_presence < 0.7:
        add_event("TRACK_MISSING", "MEDIUM" if tracking_presence >= 0.3 else "HIGH", f"tracking_presence_rate={tracking_presence:.3f}")
    if pose_presence < 0.7:
        add_event("POSE_ROW_MISSING", "MEDIUM" if pose_presence >= 0.3 else "HIGH", f"pose_presence_rate={pose_presence:.3f}")
    if gap_count:
        add_event("TRACK_MISSING_GAP", "MEDIUM", f"pose gaps >1s: {gap_count}")
    if tid_switch_count:
        add_event("TID_SWITCH_OR_COMPETING_TID", "MEDIUM", f"competing tids={dict(tid_counts)}")
    if range_jump_count:
        add_event("RANGE_JUMP", "MEDIUM", f"range jumps >1m: {range_jump_count}")
    if no_points_rate > 0:
        add_event("NO_POINTS", "MEDIUM", f"NO_POINTS_rate={no_points_rate:.3f}")
    if low_points_rate > 0:
        add_event("LOW_POINTS", "LOW", f"LOW_POINTS_rate={low_points_rate:.3f}")
    if not render_confirmed:
        add_event("RENDER_NOT_CONFIRMED", "LOW", "no per-row render visibility field available in mmWave pose logs")
    return summary, events


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    segments_root = Path(args.segments_root)
    out = ensure_dir(Path(args.out))
    filled_root = ensure_dir(out / "filled_segments")
    registry = pd.read_csv(registry_path)
    all_quality: list[dict[str, Any]] = []
    all_debug: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for _, session in registry.iterrows():
        session_id = str(session["session_id"])
        session_path = Path(str(session.get("session_path") or ""))
        segment_file = Path(str(session.get("segment_file") or ""))
        if not segment_file.is_absolute():
            segment_file = REPO_ROOT / segment_file
        if not segment_file.exists():
            segment_file = segments_root / f"{session_id}_segments.csv"
        rows = read_csv_rows(segment_file)
        tables = load_session_tables(session_path) if session_path.exists() else {"pose": pd.DataFrame(), "tracks": pd.DataFrame(), "frames": pd.DataFrame()}
        rows = fill_segment_times(session_id, rows, tables)
        rows, debug_rows = assign_tids(rows, tables)
        for row in rows:
            summary, events = quality_for_segment(row, tables)
            all_quality.append(summary)
            all_events.extend(events)
        all_debug.extend(debug_rows)
        write_csv(filled_root / f"{session_id}_segments_filled.csv", rows, SEGMENT_FIELDS)
        q_rows = [q for q in all_quality if q["session_id"] == session_id]
        summary_rows.append(
            {
                "session_id": session_id,
                "session_path": str(session_path),
                "segments": len({q["segment_id"] for q in q_rows}),
                "person_instances": len(q_rows),
                "assigned_person_instances": sum(1 for q in q_rows if str(q.get("assigned_tid") or "") != ""),
                "low_quality_person_instances": sum(1 for q in q_rows if q.get("segment_quality") == "LOW"),
                "mean_tracking_presence_rate": round(float(np.nanmean([q["tracking_presence_rate"] for q in q_rows])) if q_rows else 0.0, 4),
                "mean_pose_presence_rate": round(float(np.nanmean([q["pose_presence_rate"] for q in q_rows])) if q_rows else 0.0, 4),
            }
        )

    write_csv(out / "cleaning_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else ["session_id"])
    quality_fields = list(all_quality[0].keys()) if all_quality else SEGMENT_FIELDS
    write_csv(out / "segment_quality.csv", all_quality, quality_fields)
    debug_fields = list(all_debug[0].keys()) if all_debug else SEGMENT_FIELDS + ["assignment_note", "segment_track_rows"]
    write_csv(out / "tid_assignment_debug.csv", all_debug, debug_fields)
    event_fields = list(all_events[0].keys()) if all_events else ["session_id", "segment_id", "person_slot", "assigned_tid", "event_type", "severity", "start_time_s", "end_time_s", "detail"]
    write_csv(out / "disappearance_events.csv", all_events, event_fields)

    with (out / "DATA_CLEANING_REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# Data Cleaning Report\n\n")
        handle.write("Labels came from the user-provided segment protocols. Displayed posture was never used as ground truth.\n\n")
        handle.write("Segment times were imported from prior segment files where available; otherwise they were inferred from protocol order and marked with fallback notes.\n\n")
        handle.write(f"Sessions processed: {len(summary_rows)}\n\n")
        handle.write(f"Person-instances labeled: {len(all_quality)}\n\n")
        handle.write(f"Disappearance/reliability events: {len(all_events)}\n\n")
        handle.write("| session_id | segments | person_instances | assigned | low_quality | mean_tracking_presence | mean_pose_presence |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['session_id']} | {row['segments']} | {row['person_instances']} | "
                f"{row['assigned_person_instances']} | {row['low_quality_person_instances']} | "
                f"{row['mean_tracking_presence_rate']} | {row['mean_pose_presence_rate']} |\n"
            )
        handle.write("\nLow-confidence segments were retained and marked; disappearance periods were retained as reliability evidence.\n")

    print(f"Sessions processed: {len(summary_rows)}")
    print(f"Segments labeled: {sum(row['segments'] for row in summary_rows)}")
    print(f"Person-instances labeled: {len(all_quality)}")
    print(f"Disappearance/reliability events: {len(all_events)}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
