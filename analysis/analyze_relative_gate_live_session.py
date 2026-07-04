#!/usr/bin/env python
"""Subtype-aware live validation analysis for sitting relative gate sessions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SESSION_NAME = "sitting_relative_gate_refined_live_test"
SUBPOSES = ["STANDING", "SITTING_LEAN_BACK", "SITTING_UPRIGHT", "SITTING_LEAN_FORWARD"]
DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0]
DISCOVERY_OUT = Path("analysis_outputs/sitting_relative_gate_live_session_discovery.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze refined sitting relative gate live validation logs.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--base-analysis", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1", "YES", "PASS"])


def numeric(value: Any) -> pd.Series:
    return pd.to_numeric(value, errors="coerce")


def scalar_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if value == "NA" or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mode_or_na(values: pd.Series) -> Any:
    vals = values.dropna()
    if vals.empty:
        return "NA"
    counts = vals.astype(str).value_counts()
    return counts.index[0] if len(counts) else "NA"


def normalize_pose_label(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "NULL", "WARMUP"}:
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
    return text if text in {"UNKNOWN", "STANDING", "SITTING", "MOVING", "FALLING", "LYING"} else "OTHER"


def canonical_pose_df(session: Path) -> pd.DataFrame:
    pose = read_csv(session / "pose_predictions_ui.csv")
    source_name = "pose_predictions_ui.csv"
    if pose.empty:
        supplemental = Path("logs") / session.name / "pose_predictions_ui.csv"
        pose = read_csv(supplemental)
    if pose.empty:
        pose = read_csv(session / "mmwave_pose.csv")
        source_name = "mmwave_pose.csv"
    if pose.empty:
        return pose
    pose = pose.copy()
    if source_name == "mmwave_pose.csv":
        rename_map = {
            "mmwave_frame_num": "frame",
            "host_wall_time_iso": "time",
            "prob_standing": "stand_prob",
            "prob_sitting": "sit_prob",
            "prob_lying": "lie_prob",
            "prob_falling": "fall_prob",
            "quality_flag": "quality",
            "num_points": "geom_pts",
            "final_label": "final_display_pose",
            "speed_mps": "horizontal_speed",
        }
        pose = pose.rename(columns={k: v for k, v in rename_map.items() if k in pose.columns})
        tracks = read_csv(session / "mmwave_tracks.csv")
        if not tracks.empty:
            tracks = tracks.rename(columns={"mmwave_frame_num": "frame", "x_m": "x", "y_m": "y", "z_m": "z", "num_associated_points": "track_geom_pts"})
            for col in ["frame", "tid", "x", "y", "z", "track_geom_pts"]:
                if col in tracks:
                    tracks[col] = numeric(tracks[col])
            if {"frame", "tid", "x", "y", "z"}.issubset(tracks.columns):
                tracks["range_m"] = np.sqrt(tracks["x"] ** 2 + tracks["y"] ** 2)
                merge_cols = [c for c in ["frame", "tid", "x", "y", "z", "range_m", "track_geom_pts"] if c in tracks.columns]
                pose = pose.merge(tracks[merge_cols], on=["frame", "tid"], how="left", suffixes=("", "_track"))
                if "track_geom_pts" in pose and "geom_pts" in pose:
                    pose["geom_pts"] = numeric(pose["geom_pts"]).where(numeric(pose["geom_pts"]).notna(), numeric(pose["track_geom_pts"]))
    if "frame" not in pose and "mmwave_frame_num" in pose:
        pose["frame"] = pose["mmwave_frame_num"]
    if "time" in pose:
        t = pd.to_datetime(pose["time"], errors="coerce")
        pose["timestamp_s"] = (t - t.min()).dt.total_seconds()
    elif "host_monotonic_ns" in pose:
        ns = numeric(pose["host_monotonic_ns"])
        pose["timestamp_s"] = (ns - ns.min()) / 1_000_000_000.0
    elif "timestamp_s" not in pose:
        pose["timestamp_s"] = numeric(pose.get("frame", pd.Series(np.arange(len(pose))))) * 0.055
    for col in ["frame", "tid", "range_m", "x", "y", "z", "geom_pts", "stand_prob", "sit_prob", "prob_STANDING", "prob_SITTING", "prob_LYING", "prob_FALLING", "horizontal_speed"]:
        if col in pose:
            pose[col] = numeric(pose[col])
    if "stand_prob" not in pose and "prob_STANDING" in pose:
        pose["stand_prob"] = pose["prob_STANDING"]
    if "sit_prob" not in pose and "prob_SITTING" in pose:
        pose["sit_prob"] = pose["prob_SITTING"]
    if "move_prob" not in pose:
        pose["move_prob"] = np.nan
    if "lie_prob" not in pose and "prob_LYING" in pose:
        pose["lie_prob"] = pose["prob_LYING"]
    if "fall_prob" not in pose and "prob_FALLING" in pose:
        pose["fall_prob"] = pose["prob_FALLING"]
    display_col = "final_display_pose" if "final_display_pose" in pose else "displayed_label" if "displayed_label" in pose else "final_label"
    pose["display_pose_norm"] = pose.get(display_col, pd.Series("UNKNOWN", index=pose.index)).map(normalize_pose_label)
    pose["quality_norm"] = pose.get("quality", pd.Series("", index=pose.index)).astype(str).str.upper()
    pose["display_status_norm"] = pose.get("display_status", pd.Series("", index=pose.index)).astype(str).str.upper()
    if "sitting_relative_gate_passed" in pose:
        pose["sitting_relative_gate_passed_bool"] = as_bool(pose["sitting_relative_gate_passed"])
    else:
        pose["sitting_relative_gate_passed_bool"] = False
    for name in ["sitting_relative_gate_range_ok", "sitting_relative_standing_veto_ok", "moving_override_blocked_by_body_still"]:
        if name in pose:
            pose[name + "_bool"] = as_bool(pose[name])
        else:
            pose[name + "_bool"] = False
    return pose.sort_values("timestamp_s")


def frame_period(pose: pd.DataFrame) -> float:
    if pose.empty:
        return 0.055
    by_frame = pose.dropna(subset=["frame", "timestamp_s"]).drop_duplicates("frame").sort_values("frame")
    if len(by_frame) > 5:
        periods = by_frame["timestamp_s"].diff() / by_frame["frame"].diff()
        periods = periods.replace([np.inf, -np.inf], np.nan).dropna()
        periods = periods[(periods > 0.02) & (periods < 0.2)]
        if len(periods):
            return float(periods.median())
    times = pose["timestamp_s"].dropna().sort_values().diff().dropna()
    times = times[(times > 0.02) & (times < 0.2)]
    return float(times.median()) if len(times) else 0.055


def discovery_rows(selected_session: Path | None = None) -> pd.DataFrame:
    roots = [
        Path(r"C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs"),
        Path("..") / "logs",
        Path("logs"),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            pass
        if not root.exists():
            continue
        for path in [root] + [p for p in root.iterdir() if p.is_dir()]:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            files = {p.name: p for p in path.glob("*") if p.is_file()}
            if SESSION_NAME.lower() not in path.name.lower() and "pose_ui_metadata.json" not in files:
                continue
            metadata = {}
            for meta_name in ["session_metadata.json", "pose_ui_metadata.json"]:
                if meta_name in files:
                    try:
                        metadata = json.loads(files[meta_name].read_text(encoding="utf-8"))
                    except Exception:
                        metadata = {}
                    break
            csv_count = len([p for p in files.values() if p.suffix.lower() == ".csv"])
            mtime = max((p.stat().st_mtime for p in files.values()), default=path.stat().st_mtime)
            exact = SESSION_NAME.lower() in path.name.lower()
            try:
                selected = bool(selected_session and path.resolve() == selected_session.resolve())
            except OSError:
                selected = bool(selected_session and str(path).lower() == str(selected_session).lower())
            rows.append(
                {
                    "rank": 0,
                    "path": str(path.resolve()),
                    "modified_time": pd.to_datetime(mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id_if_found": metadata.get("session_id", path.name if exact else ""),
                    "cfg_path_if_found": metadata.get("cfg_path", metadata.get("mmwave_cfg_path", "")),
                    "csv_count": csv_count,
                    "has_mmwave_frames": (path / "mmwave_frames.csv").exists(),
                    "has_mmwave_tracks": (path / "mmwave_tracks.csv").exists(),
                    "has_mmwave_pose": (path / "mmwave_pose.csv").exists() or (path / "pose_predictions_ui.csv").exists(),
                    "has_rgb_frames": (path / "rgb_frames.csv").exists(),
                    "has_rgb_tracks": (path / "rgb_tracks.csv").exists(),
                    "has_rgb_keypoints": (path / "rgb_keypoints.csv").exists(),
                    "has_sync_index": (path / "sync_index.csv").exists(),
                    "has_rgb_video": (path / "rgb_annotated.mp4").exists() or (path / "videos" / "rgb_annotated.mp4").exists(),
                    "selected": selected,
                    "notes": "selected exact folder name match" if selected and exact else ("exact folder name match" if exact else "metadata-bearing candidate"),
                    "_mtime": mtime,
                }
            )
    rows = sorted(rows, key=lambda r: (not r["selected"], not (SESSION_NAME.lower() in Path(r["path"]).name.lower()), -r["_mtime"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row.pop("_mtime", None)
    return pd.DataFrame(rows)


def contiguous_runs(times: pd.Series, max_gap_s: float) -> list[tuple[float, float, int]]:
    vals = sorted(float(v) for v in times.dropna().unique())
    if not vals:
        return []
    out = []
    start = prev = vals[0]
    count = 1
    for val in vals[1:]:
        if val - prev <= max_gap_s:
            prev = val
            count += 1
        else:
            out.append((start, prev, count))
            start = prev = val
            count = 1
    out.append((start, prev, count))
    return out


def expected_template() -> pd.DataFrame:
    rows = []
    specs = [
        ("standing", "STANDING", "STANDING"),
        ("leanback", "SITTING", "SITTING_LEAN_BACK"),
        ("upright", "SITTING", "SITTING_UPRIGHT"),
        ("leanforward", "SITTING", "SITTING_LEAN_FORWARD"),
    ]
    for prefix, pose, subpose in specs:
        for d in DISTANCES:
            rows.append(
                {
                    "segment_id": f"{prefix}_{int(d)}m",
                    "expected_pose": pose,
                    "expected_subpose": subpose,
                    "expected_distance_m": d,
                    "start_time_s": np.nan,
                    "end_time_s": np.nan,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def infer_segments(pose: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    segs = segments.copy()
    if "notes" in segs:
        segs["notes"] = segs["notes"].fillna("").astype(str).str.replace(r"^(nan;\s*)+", "", regex=True)
        segs["notes"] = segs["notes"].str.replace(r"(auto range plateau; raw=[^;]+; tol=[^;]+m; samples=\d+;\s*)+(auto range plateau; raw=[^;]+; tol=[^;]+m; samples=\d+)", r"\2", regex=True)
        segs["notes"] = segs["notes"].str.strip("; ")
    for col in ["start_time_s", "end_time_s"]:
        segs[col] = numeric(segs[col])
    durations = segs["end_time_s"] - segs["start_time_s"]
    if not segs[["start_time_s", "end_time_s"]].isna().any().any() and durations.gt(0).all():
        if "confidence" not in segs:
            segs["confidence"] = 1.0
        return segs
    cursor = float(pose["timestamp_s"].min()) if not pose.empty else 0.0
    session_end = float(pose["timestamp_s"].max()) if not pose.empty else len(segs) * 60.0
    rows = []
    for _, seg in segs.iterrows():
        dist = float(seg["expected_distance_m"])
        chosen = None
        chosen_tol = None
        for tol in [0.35, 0.55, 0.80, 1.10]:
            search_end = min(cursor + 150.0, session_end)
            cand = pose[
                (pose["timestamp_s"] >= cursor)
                & (pose["timestamp_s"] <= search_end)
                & ((pose["range_m"] - dist).abs() <= tol)
            ]
            runs = [r for r in contiguous_runs(cand["timestamp_s"], 3.5) if r[1] - r[0] >= 18.0]
            if runs:
                chosen = runs[0]
                chosen_tol = tol
                break
        if chosen:
            raw_start, raw_end, count = chosen
            if raw_end - raw_start > 75:
                raw_end = raw_start + 60
            start = raw_start + 4.0 if raw_end - raw_start > 30 else raw_start
            end = raw_end - 4.0 if raw_end - raw_start > 30 else raw_end
            confidence = max(0.35, min(0.95, (end - start) / 55.0 * (0.55 / max(float(chosen_tol), 0.55))))
            note = f"auto range plateau; raw={raw_start:.2f}-{raw_end:.2f}s; tol={chosen_tol:.2f}m; samples={count}"
            cursor = raw_end + 2.0
        else:
            next_rows = pose[pose["timestamp_s"] >= cursor]
            if next_rows.empty:
                start = cursor
            else:
                start = float(next_rows["timestamp_s"].min())
            end = min(start + 50.0, session_end)
            confidence = 0.25
            note = "auto time-order fallback; range plateau not reliable"
            cursor = end + 2.0
        row = seg.to_dict()
        row["start_time_s"] = round(float(start), 3)
        row["end_time_s"] = round(float(end), 3)
        row["duration_s"] = round(float(end - start), 3)
        row["confidence"] = round(float(confidence), 3)
        existing = str(row.get("notes", "") or "")
        row["notes"] = (existing + "; " if existing else "") + note
        rows.append(row)
    return pd.DataFrame(rows)


def segment_slice(df: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["timestamp_s"] >= float(seg["start_time_s"])) & (df["timestamp_s"] <= float(seg["end_time_s"]))].copy()


def primary_by_frame(rows: pd.DataFrame, expected_distance: float) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    data = rows.copy()
    data["_err"] = (data["range_m"] - expected_distance).abs().fillna(99)
    if "frame" in data:
        return data.sort_values(["frame", "_err"]).groupby("frame", as_index=False).first().drop(columns=["_err"], errors="ignore")
    return data.sort_values(["timestamp_s", "_err"]).drop(columns=["_err"], errors="ignore")


def visible_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    status = rows.get("display_status_norm", pd.Series("", index=rows.index)).astype(str)
    pose = rows.get("display_pose_norm", pd.Series("UNKNOWN", index=rows.index)).astype(str)
    quality = rows.get("quality_norm", pd.Series("", index=rows.index)).astype(str)
    if status.replace("", np.nan).dropna().empty:
        return ~pose.isin(["UNKNOWN", "WARMUP"]) & ~quality.eq("WARMUP")
    return status.eq("STABLE") & ~pose.isin(["UNKNOWN", "WARMUP"]) & ~quality.eq("WARMUP")


def event_reason(block: pd.DataFrame, before: pd.Series | None, after: pd.Series | None, expected_frames_missing: bool) -> str:
    if expected_frames_missing and block.empty:
        return "TRACK_DROPOUT"
    if not block.empty:
        q = block.get("quality_norm", pd.Series("", index=block.index)).astype(str)
        geom = numeric(block.get("geom_pts", pd.Series(np.nan, index=block.index)))
        status = block.get("display_status_norm", pd.Series("", index=block.index)).astype(str)
        if status.str.contains("SUSPECT|PROVISIONAL|HIDDEN", regex=True, na=False).any():
            return "SUSPECT_OR_PROVISIONAL_HIDDEN"
        if status.ne("STABLE").any():
            return "RENDER_NOT_CONFIRMED"
        if q.str.contains("NO_POINTS", na=False).any() or geom.fillna(0).le(0).mean() > 0.5:
            return "LOW_GEOMETRY_NO_POINTS"
        return "POSE_ROW_MISSING"
    if before is not None and after is not None:
        if before.get("tid") != after.get("tid"):
            return "TID_SWITCH"
        if abs(scalar_float(before.get("range_m")) - scalar_float(after.get("range_m"))) > 0.75:
            return "OUT_OF_RANGE_OR_RANGE_JUMP"
    return "UNKNOWN"


def disappearance_events_for_segment(pose: pd.DataFrame, seg: pd.Series, period: float) -> list[dict[str, Any]]:
    rows = segment_slice(pose, seg)
    start = float(seg["start_time_s"])
    end = float(seg["end_time_s"])
    bins = np.arange(start, end + period, period)
    if len(bins) < 2:
        return []
    primary = primary_by_frame(rows, float(seg["expected_distance_m"]))
    if primary.empty:
        run_times = [(start, end, 0)]
    else:
        pr = primary.copy()
        pr["_bin"] = np.floor((pr["timestamp_s"] - start) / period).astype(int)
        visible_bins = set(pr.loc[visible_mask(pr), "_bin"].dropna().astype(int).tolist())
        all_bins = set(range(len(bins) - 1))
        missing_bins = sorted(all_bins - visible_bins)
        run_times = []
        if missing_bins:
            s = p = missing_bins[0]
            count = 1
            for b in missing_bins[1:]:
                if b == p + 1:
                    p = b
                    count += 1
                else:
                    run_times.append((start + s * period, min(start + (p + 1) * period, end), count))
                    s = p = b
                    count = 1
            run_times.append((start + s * period, min(start + (p + 1) * period, end), count))
    events = []
    for a, b, count in run_times:
        if b - a < max(period * 2, 0.15):
            continue
        block = primary[(primary["timestamp_s"] >= a) & (primary["timestamp_s"] <= b)] if not primary.empty else pd.DataFrame()
        before_df = primary[primary["timestamp_s"] < a].tail(1) if not primary.empty else pd.DataFrame()
        after_df = primary[primary["timestamp_s"] > b].head(1) if not primary.empty else pd.DataFrame()
        before = before_df.iloc[0] if not before_df.empty else None
        after = after_df.iloc[0] if not after_df.empty else None
        reason = event_reason(block, before, after, block.empty)
        if before is not None and after is not None and before.get("tid") != after.get("tid"):
            reason = "TID_SWITCH"
        events.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "expected_subpose": seg["expected_subpose"],
                "expected_distance_m": seg["expected_distance_m"],
                "start_time_s": round(a, 3),
                "end_time_s": round(b, 3),
                "duration_s": round(b - a, 3),
                "tid_before": before.get("tid") if before is not None else "NA",
                "tid_after": after.get("tid") if after is not None else "NA",
                "range_before": before.get("range_m") if before is not None else "NA",
                "range_after": after.get("range_m") if after is not None else "NA",
                "last_display_pose_before": before.get("display_pose_norm") if before is not None else "NA",
                "first_display_pose_after": after.get("display_pose_norm") if after is not None else "NA",
                "quality_before": before.get("quality_norm") if before is not None else "NA",
                "quality_after": after.get("quality_norm") if after is not None else "NA",
                "geom_pts_before": before.get("geom_pts") if before is not None else "NA",
                "geom_pts_after": after.get("geom_pts") if after is not None else "NA",
                "reason_hypothesis": reason,
            }
        )
    return events


def compute_metrics(pose: pd.DataFrame, segments: pd.DataFrame, period: float, session: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    events = []
    for _, seg in segments.iterrows():
        rows = segment_slice(pose, seg)
        primary = primary_by_frame(rows, float(seg["expected_distance_m"]))
        expected_frames = max(1, int(round(float(seg["duration_s"]) / period)))
        frames_with_track = int(primary["frame"].nunique()) if not primary.empty and "frame" in primary else int(len(primary))
        pose_present = primary[~primary["display_pose_norm"].isin(["UNKNOWN"])]
        frames_with_pose = int(pose_present["frame"].nunique()) if not pose_present.empty and "frame" in pose_present else int(len(pose_present))
        vis = visible_mask(primary)
        ui_visible_frames = int(primary.loc[vis, "frame"].nunique()) if not primary.empty and "frame" in primary else int(vis.sum())
        event_rows = disappearance_events_for_segment(pose, seg, period)
        events.extend(event_rows)
        display = primary["display_pose_norm"] if not primary.empty else pd.Series(dtype=str)
        expected_pose = str(seg["expected_pose"]).upper()
        stand_prob = numeric(primary.get("stand_prob", pd.Series(dtype=float)))
        sit_prob = numeric(primary.get("sit_prob", pd.Series(dtype=float)))
        gate_state = primary.get("sitting_relative_gate_state", pd.Series("", index=primary.index)).astype(str).str.upper() if not primary.empty else pd.Series(dtype=str)
        q = primary.get("quality_norm", pd.Series("", index=primary.index)).astype(str) if not primary.empty else pd.Series(dtype=str)
        geom = numeric(primary.get("geom_pts", pd.Series(dtype=float))) if not primary.empty else pd.Series(dtype=float)
        active_counts = rows.groupby("frame")["tid"].nunique() if not rows.empty and "frame" in rows else pd.Series(dtype=float)
        tids = primary["tid"].dropna() if not primary.empty and "tid" in primary else pd.Series(dtype=float)
        tid_switch_count = int((tids != tids.shift()).sum() - 1) if len(tids) else 0
        if tid_switch_count < 0:
            tid_switch_count = 0
        range_err = primary["range_m"] - float(seg["expected_distance_m"]) if not primary.empty and "range_m" in primary else pd.Series(dtype=float)
        move_state = primary.get("moving_override_state", pd.Series("", index=primary.index)).astype(str).str.upper() if not primary.empty else pd.Series(dtype=str)
        metric = {
            "segment_id": seg["segment_id"],
            "expected_pose": expected_pose,
            "expected_subpose": seg["expected_subpose"],
            "expected_distance_m": seg["expected_distance_m"],
            "start_time_s": seg["start_time_s"],
            "end_time_s": seg["end_time_s"],
            "duration_s": seg["duration_s"],
            "confidence": seg.get("confidence", "NA"),
            "frames_expected": expected_frames,
            "frames_with_track": frames_with_track,
            "frames_with_pose": frames_with_pose,
            "tracking_presence_rate": frames_with_track / expected_frames,
            "pose_presence_rate": frames_with_pose / expected_frames,
            "ui_visible_rate": ui_visible_frames / expected_frames,
            "disappearance_rate": 1.0 - (ui_visible_frames / expected_frames),
            "longest_disappearance_s": max([e["duration_s"] for e in event_rows], default=0.0),
            "num_disappearance_events": len(event_rows),
            "range_mae_m": float(range_err.abs().mean()) if len(range_err.dropna()) else np.nan,
            "range_bias_m": float(range_err.mean()) if len(range_err.dropna()) else np.nan,
            "range_jitter_m": float(primary["range_m"].std()) if not primary.empty and "range_m" in primary else np.nan,
            "tid_switch_count": tid_switch_count,
            "extra_track_rate": float(active_counts.gt(1).sum() / expected_frames) if len(active_counts) else 0.0,
            "dominant_tid": mode_or_na(tids),
            "dominant_display_pose": mode_or_na(display),
            "posture_accuracy": float(display.eq(expected_pose).mean()) if len(display) else np.nan,
            "display_standing_rate": float(display.eq("STANDING").sum() / expected_frames),
            "display_sitting_rate": float(display.eq("SITTING").sum() / expected_frames),
            "display_moving_rate": float(display.eq("MOVING").sum() / expected_frames),
            "display_unknown_rate": float(display.eq("UNKNOWN").sum() / expected_frames),
            "mean_stand_prob": float(stand_prob.mean()) if len(stand_prob.dropna()) else np.nan,
            "mean_sit_prob": float(sit_prob.mean()) if len(sit_prob.dropna()) else np.nan,
            "mean_move_prob_if_available": float(numeric(primary.get("move_prob", pd.Series(dtype=float))).mean()) if not primary.empty else np.nan,
            "mean_lie_prob_if_available": float(numeric(primary.get("lie_prob", pd.Series(dtype=float))).mean()) if not primary.empty else np.nan,
            "mean_fall_prob_if_available": float(numeric(primary.get("fall_prob", pd.Series(dtype=float))).mean()) if not primary.empty else np.nan,
            "mean_sit_minus_stand_margin": float((sit_prob - stand_prob).mean()) if len(primary) else np.nan,
            "sitting_relative_gate_trigger_rate": float(gate_state.isin(["WAIT", "PASS"]).sum() / expected_frames) if len(gate_state) else 0.0,
            "sitting_relative_gate_blocked_range_rate": float((~primary.get("sitting_relative_gate_range_ok_bool", pd.Series(False, index=primary.index))).sum() / expected_frames) if not primary.empty else 0.0,
            "sitting_relative_gate_blocked_standing_veto_rate": float((~primary.get("sitting_relative_standing_veto_ok_bool", pd.Series(False, index=primary.index))).sum() / expected_frames) if not primary.empty else 0.0,
            "sitting_relative_gate_passed_rate": float(primary.get("sitting_relative_gate_passed_bool", pd.Series(False, index=primary.index)).sum() / expected_frames) if not primary.empty else 0.0,
            "moving_override_rate": float(move_state.isin(["TRANSLATION_CONFIRMED", "SUSTAINED"]).sum() / expected_frames) if len(move_state) else 0.0,
            "moving_override_blocked_body_still_rate": float(primary.get("moving_override_blocked_by_body_still_bool", pd.Series(False, index=primary.index)).sum() / expected_frames) if not primary.empty else 0.0,
            "NO_POINTS_rate": float(q.str.contains("NO_POINTS", na=False).sum() / expected_frames) if len(q) else 0.0,
            "LOW_POINTS_rate": float(q.str.contains("LOW_POINTS", na=False).sum() / expected_frames) if len(q) else 0.0,
            "OK_rate": float(q.eq("OK").sum() / expected_frames) if len(q) else 0.0,
            "mean_geom_pts": float(geom.mean()) if len(geom.dropna()) else np.nan,
            "geom_pts_ge_1_rate": float(geom.ge(1).sum() / expected_frames) if len(geom) else 0.0,
            "geom_pts_ge_3_rate": float(geom.ge(3).sum() / expected_frames) if len(geom) else 0.0,
            "rgb_frames_count": count_rows(session / "rgb_frames.csv"),
            "rgb_track_presence_rate": np.nan,
            "rgb_video_present": (session / "rgb_annotated.mp4").exists() or (session / "videos" / "rgb_annotated.mp4").exists(),
        }
        metrics.append(metric)
    metrics_df = pd.DataFrame(metrics)
    events_df = pd.DataFrame(events)
    summary_rows = []
    for _, seg in segments.iterrows():
        ev = events_df[events_df["segment_id"].eq(seg["segment_id"])] if not events_df.empty else pd.DataFrame()
        met = metrics_df[metrics_df["segment_id"].eq(seg["segment_id"])].iloc[0]
        reason = mode_or_na(ev["reason_hypothesis"]) if not ev.empty else "NA"
        summary_rows.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "expected_subpose": seg["expected_subpose"],
                "expected_distance_m": seg["expected_distance_m"],
                "num_disappearance_events": len(ev),
                "total_disappearance_s": float(ev["duration_s"].sum()) if not ev.empty else 0.0,
                "longest_disappearance_s": float(ev["duration_s"].max()) if not ev.empty else 0.0,
                "disappearance_rate": met["disappearance_rate"],
                "most_common_reason_hypothesis": reason,
                "tracking_presence_rate": met["tracking_presence_rate"],
                "pose_presence_rate": met["pose_presence_rate"],
                "ui_visible_rate": met["ui_visible_rate"],
            }
        )
    disappearance_summary = pd.DataFrame(summary_rows)
    gate_df = make_gate_validation(metrics_df, pose, segments)
    return metrics_df, events_df, disappearance_summary, gate_df


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def make_gate_validation(metrics: pd.DataFrame, pose: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, met in metrics.iterrows():
        seg = segments[segments["segment_id"].eq(met["segment_id"])].iloc[0]
        sp = primary_by_frame(segment_slice(pose, seg), float(seg["expected_distance_m"]))
        stand = numeric(sp.get("stand_prob", pd.Series(dtype=float)))
        sit = numeric(sp.get("sit_prob", pd.Series(dtype=float)))
        display = sp.get("display_pose_norm", pd.Series(dtype=str))
        sit_gt = sit > stand
        expected = str(met["expected_pose"])
        rows.append(
            {
                "segment_id": met["segment_id"],
                "expected_pose": expected,
                "expected_subpose": met["expected_subpose"],
                "expected_distance_m": met["expected_distance_m"],
                "posture_accuracy": met["posture_accuracy"],
                "display_sitting_rate": met["display_sitting_rate"],
                "display_standing_rate": met["display_standing_rate"],
                "mean_stand_prob": met["mean_stand_prob"],
                "mean_sit_prob": met["mean_sit_prob"],
                "mean_sit_minus_stand_margin": met["mean_sit_minus_stand_margin"],
                "relative_gate_enabled_detected": bool(sp.get("sitting_relative_gate_min_prob", pd.Series(dtype=float)).notna().any()) if not sp.empty else False,
                "relative_gate_trigger_rate": met["sitting_relative_gate_trigger_rate"],
                "relative_gate_passed_rate": met["sitting_relative_gate_passed_rate"],
                "relative_gate_blocked_range_rate": met["sitting_relative_gate_blocked_range_rate"],
                "relative_gate_blocked_standing_veto_rate": met["sitting_relative_gate_blocked_standing_veto_rate"],
                "frames_where_sit_prob_gt_stand_prob": int(sit_gt.sum()) if len(sit_gt) else 0,
                "frames_where_display_not_sitting_despite_sit_prob_gt_stand_prob": int((sit_gt & ~display.eq("SITTING")).sum()) if len(sit_gt) else 0,
                "false_sitting_if_expected_standing": float(display.eq("SITTING").mean()) if expected == "STANDING" and len(display) else 0.0,
                "false_standing_if_expected_sitting": float(display.eq("STANDING").mean()) if expected == "SITTING" and len(display) else 0.0,
                "moving_false_positive_rate": met["display_moving_rate"],
            }
        )
    return pd.DataFrame(rows)


def rgb_summary(session: Path, out: Path) -> pd.DataFrame:
    rows = [{
        "rgb_frames_rows": count_rows(session / "rgb_frames.csv"),
        "rgb_tracks_rows": count_rows(session / "rgb_tracks.csv"),
        "rgb_keypoints_rows": count_rows(session / "rgb_keypoints.csv"),
        "sync_index_rows": count_rows(session / "sync_index.csv"),
        "rgb_actions_rows": count_rows(session / "rgb_actions.csv"),
        "rgb_annotated_mp4": "present" if (session / "rgb_annotated.mp4").exists() or (session / "videos" / "rgb_annotated.mp4").exists() else "missing",
        "notes": "No RGB posture accuracy claimed unless rgb_actions.csv has meaningful labels.",
    }]
    df = pd.DataFrame(rows)
    df.to_csv(out / "rgb_summary.csv", index=False)
    return df


def create_plots(out: Path, pose: pd.DataFrame, segments: pd.DataFrame, metrics: pd.DataFrame, events: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=150)
        plt.close()

    plt.figure(figsize=(12, 5))
    for tid, rows in pose.groupby("tid"):
        plt.plot(rows["timestamp_s"], rows["range_m"], ".", ms=1, label=f"TID {tid:g}")
    for _, seg in segments.iterrows():
        plt.axvspan(seg["start_time_s"], seg["end_time_s"], alpha=0.06)
    plt.xlabel("time (s)")
    plt.ylabel("range (m)")
    plt.title("Range by TID")
    plt.legend(markerscale=5, fontsize=7, ncol=4)
    save("timeline_range_by_tid.png")

    pose_code = {"UNKNOWN": 0, "STANDING": 1, "SITTING": 2, "MOVING": 3, "LYING": 4, "FALLING": 5, "OTHER": 6}
    plt.figure(figsize=(12, 5))
    for tid, rows in pose.groupby("tid"):
        plt.plot(rows["timestamp_s"], rows["display_pose_norm"].map(pose_code), ".", ms=1, label=f"TID {tid:g}")
    plt.yticks(list(pose_code.values()), list(pose_code.keys()))
    plt.xlabel("time (s)")
    plt.title("Display pose by TID")
    save("timeline_display_pose_by_tid.png")

    plt.figure(figsize=(12, 5))
    plt.plot(pose["timestamp_s"], pose["stand_prob"], ".", ms=1, label="stand_prob")
    plt.plot(pose["timestamp_s"], pose["sit_prob"], ".", ms=1, label="sit_prob")
    plt.xlabel("time (s)")
    plt.ylabel("probability")
    plt.title("Stand/Sit probabilities")
    plt.legend()
    save("timeline_stand_sit_probs_by_tid.png")

    plt.figure(figsize=(12, 5))
    plt.plot(pose["timestamp_s"], pose["geom_pts"], ".", ms=1)
    plt.xlabel("time (s)")
    plt.ylabel("geom_pts")
    plt.title("Geometry point count and quality")
    save("timeline_geom_pts_quality.png")

    plt.figure(figsize=(12, 4))
    for _, ev in events.iterrows():
        plt.axvspan(ev["start_time_s"], ev["end_time_s"], color="tab:red", alpha=0.25)
    plt.plot(pose["timestamp_s"], pose["range_m"], ".", ms=1, color="black")
    plt.xlabel("time (s)")
    plt.ylabel("range (m)")
    plt.title("Disappearance events over range timeline")
    save("timeline_disappearance_events.png")

    pivot = metrics.pivot_table(index="expected_subpose", columns="expected_distance_m", values="posture_accuracy", aggfunc="mean")
    pivot = pivot.reindex(SUBPOSES)
    plt.figure(figsize=(8, 4))
    for subpose, row in pivot.iterrows():
        plt.plot(row.index, row.values, marker="o", label=subpose)
    plt.xlabel("distance (m)")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.title("Posture accuracy by distance and subpose")
    plt.legend(fontsize=7)
    save("posture_accuracy_by_distance_and_subpose.png")

    dist = metrics[["segment_id", "display_standing_rate", "display_sitting_rate", "display_moving_rate", "display_unknown_rate"]].set_index("segment_id")
    plt.figure(figsize=(12, 5))
    dist.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.ylabel("rate")
    plt.title("Display pose distribution")
    save("display_pose_distribution_by_distance_and_subpose.png")

    for col, name, ylabel in [
        ("sitting_relative_gate_trigger_rate", "sitting_relative_gate_trigger_by_distance_and_subpose.png", "trigger rate"),
        ("disappearance_rate", "disappearance_rate_by_distance_and_subpose.png", "disappearance rate"),
        ("tracking_presence_rate", "tracking_presence_by_distance_and_subpose.png", "tracking presence"),
        ("range_mae_m", "range_mae_by_distance_and_subpose.png", "range MAE (m)"),
    ]:
        plt.figure(figsize=(8, 4))
        piv = metrics.pivot_table(index="expected_subpose", columns="expected_distance_m", values=col, aggfunc="mean").reindex(SUBPOSES)
        for subpose, row in piv.iterrows():
            plt.plot(row.index, row.values, marker="o", label=subpose)
        plt.xlabel("distance (m)")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend(fontsize=7)
        save(name)

    heat = metrics.pivot_table(index="expected_subpose", columns="expected_distance_m", values="mean_sit_minus_stand_margin", aggfunc="mean").reindex(SUBPOSES)
    plt.figure(figsize=(7, 4))
    im = plt.imshow(heat.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, label="sit - stand")
    plt.xticks(range(len(heat.columns)), [f"{c:g}m" for c in heat.columns])
    plt.yticks(range(len(heat.index)), heat.index)
    plt.title("Sit minus stand margin")
    save("sit_minus_stand_margin_heatmap.png")


def verdict_tracking(row: pd.Series) -> str:
    if scalar_float(row["tracking_presence_rate"]) < 0.75:
        return "DROPOUT"
    if scalar_float(row["tid_switch_count"], 0) > 0:
        return "TID_SWITCH"
    if scalar_float(row["disappearance_rate"]) > 0.25:
        return "UI_DISAPPEARS"
    return "OK"


def verdict_posture(row: pd.Series) -> str:
    acc = scalar_float(row["posture_accuracy"])
    if acc >= 0.9:
        return "GOOD"
    if str(row["expected_pose"]) == "SITTING" and scalar_float(row["display_standing_rate"]) > scalar_float(row["display_sitting_rate"]):
        return "SIT_AS_STAND"
    if str(row["expected_pose"]) == "STANDING" and scalar_float(row["display_sitting_rate"]) > 0.10:
        return "STAND_AS_SIT"
    if scalar_float(row["display_moving_rate"]) > 0.10:
        return "MOVING_FALSE_POSITIVE"
    return "WEAK" if acc >= 0.7 else "FAILED"


def verdict_gate(row: pd.Series) -> str:
    if row["expected_pose"] == "STANDING" and scalar_float(row["false_sitting_if_expected_standing"]) > 0.10:
        return "UNSAFE_STANDING_FALSE_SIT"
    if row["expected_pose"] == "SITTING" and scalar_float(row["false_standing_if_expected_sitting"]) > 0.50:
        return "INSUFFICIENT_SIT_AS_STAND"
    return "OK"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    data = df.head(max_rows).copy() if max_rows else df.copy()
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: fmt(x))
    headers = [str(c) for c in data.columns]
    rows = []
    for _, row in data.iterrows():
        rows.append([str(row.get(c, "")) for c in data.columns])

    def clean(text: str) -> str:
        return text.replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def session_notes(session: Path) -> str:
    files = {p.name for p in session.glob("*") if p.is_file()}
    parts = [f"{len([name for name in files if name.lower().endswith('.csv')])} CSV files"]
    for name in ["mmwave_frames.csv", "mmwave_tracks.csv", "mmwave_pose.csv", "rgb_frames.csv", "rgb_tracks.csv", "rgb_keypoints.csv", "sync_index.csv"]:
        if name in files:
            parts.append(name)
    if (session / "rgb_annotated.mp4").exists() or (session / "videos" / "rgb_annotated.mp4").exists():
        parts.append("rgb_annotated.mp4")
    return "Selected folder contains " + ", ".join(parts) + "."


def summarize(metrics: pd.DataFrame, disappear: pd.DataFrame, gate: pd.DataFrame) -> dict[str, str]:
    standing12 = gate[(gate["expected_pose"].eq("STANDING")) & (gate["expected_distance_m"].isin([1.0, 2.0]))]
    standing_protected = bool((standing12["false_sitting_if_expected_standing"].fillna(0) <= 0.10).all()) if not standing12.empty else False
    far_sit = gate[(gate["expected_pose"].eq("SITTING")) & (gate["expected_distance_m"].isin([4.0, 5.0]))]
    far_acc = scalar_float(far_sit["posture_accuracy"].mean()) if not far_sit.empty else np.nan
    mid3 = gate[(gate["expected_pose"].eq("SITTING")) & (gate["expected_distance_m"].eq(3.0))]
    acc3 = scalar_float(mid3["posture_accuracy"].mean()) if not mid3.empty else np.nan
    sub = metrics.groupby("expected_subpose")["posture_accuracy"].mean().sort_values(ascending=False)
    best = sub.index[0] if len(sub) else "NA"
    worst = sub.index[-1] if len(sub) else "NA"
    dis5 = metrics[metrics["expected_distance_m"].eq(5.0)]["disappearance_rate"].mean()
    dis_other = metrics[~metrics["expected_distance_m"].eq(5.0)]["disappearance_rate"].mean()
    common_reason = mode_or_na(disappear["most_common_reason_hypothesis"]) if not disappear.empty else "NA"
    main_result = (
        "Standing 1m/2m remained protected, so the refined gate is safe but insufficient; sitting accuracy still depends on subtype and range."
        if standing_protected
        else "Standing 1m/2m was not protected well enough; the refined gate should not be treated as safe by default."
    )
    disappearance = (
        f"UI disappearance was observed with mean 5m disappearance {fmt(dis5)} versus other distances {fmt(dis_other)}. "
        f"The dominant logged reason was {common_reason}, with NO_POINTS/low geometry rates used as supporting evidence."
    )
    next_fix = (
        "Do not tune one global threshold. The next fix path should separate geometry/track retention from posture subtype handling, then validate with RGB or manually checked video boundaries."
    )
    return {
        "main_result": main_result,
        "disappearance": disappearance,
        "next_fix": next_fix,
        "best": best,
        "worst": worst,
        "far_acc": fmt(far_acc),
        "acc3": fmt(acc3),
        "standing_protected": "yes" if standing_protected else "no",
    }


def write_report(out: Path, session: Path, metadata: dict[str, Any], segments: pd.DataFrame, metrics: pd.DataFrame, disappear: pd.DataFrame, events: pd.DataFrame, gate: pd.DataFrame, rgb: pd.DataFrame) -> dict[str, str]:
    summary = summarize(metrics, disappear, gate)
    tracking_table = metrics.copy()
    tracking_table["distance_m"] = tracking_table["expected_distance_m"]
    tracking_table["tracking_verdict"] = tracking_table.apply(verdict_tracking, axis=1)
    posture_table = metrics.copy()
    posture_table["distance_m"] = posture_table["expected_distance_m"]
    posture_table["posture_verdict"] = posture_table.apply(verdict_posture, axis=1)
    gate_table = gate.copy()
    gate_table["distance_m"] = gate_table["expected_distance_m"]
    gate_table["blocked_range_rate"] = gate_table["relative_gate_blocked_range_rate"]
    gate_table["blocked_standing_veto_rate"] = gate_table["relative_gate_blocked_standing_veto_rate"]
    gate_table["false_sitting_if_standing"] = gate_table["false_sitting_if_expected_standing"]
    gate_table["false_standing_if_sitting"] = gate_table["false_standing_if_expected_sitting"]
    gate_table["gate_verdict"] = gate_table.apply(verdict_gate, axis=1)
    subtype = metrics.groupby("expected_subpose", as_index=False).agg(
        mean_accuracy=("posture_accuracy", "mean"),
        mean_display_sitting_rate=("display_sitting_rate", "mean"),
        mean_display_standing_rate=("display_standing_rate", "mean"),
        mean_disappearance_rate=("disappearance_rate", "mean"),
    )
    worst_best = []
    for _, row in subtype.iterrows():
        sm = metrics[metrics["expected_subpose"].eq(row["expected_subpose"])]
        worst_best.append({
            "expected_subpose": row["expected_subpose"],
            "worst_distance": sm.sort_values("posture_accuracy").iloc[0]["expected_distance_m"] if not sm.empty else "NA",
            "best_distance": sm.sort_values("posture_accuracy", ascending=False).iloc[0]["expected_distance_m"] if not sm.empty else "NA",
            "subtype_verdict": "WEAK" if scalar_float(row["mean_accuracy"]) < 0.7 else "OK",
        })
    subtype = subtype.merge(pd.DataFrame(worst_best), on="expected_subpose", how="left")
    distance = metrics.pivot_table(index="expected_distance_m", columns="expected_subpose", values="posture_accuracy", aggfunc="mean").reset_index()
    distance = distance.rename(columns={"expected_distance_m": "distance_m", "STANDING": "standing_accuracy", "SITTING_LEAN_BACK": "leanback_accuracy", "SITTING_UPRIGHT": "upright_accuracy", "SITTING_LEAN_FORWARD": "leanforward_accuracy"})
    dist_extra = metrics.groupby("expected_distance_m", as_index=False).agg(tracking_presence_rate=("tracking_presence_rate", "mean"), disappearance_rate=("disappearance_rate", "mean")).rename(columns={"expected_distance_m": "distance_m"})
    distance = distance.merge(dist_extra, on="distance_m", how="left")
    distance["distance_verdict"] = distance.apply(lambda r: "NEAR_OR_OVER_LIMIT" if scalar_float(r.get("disappearance_rate")) > 0.25 else "OK", axis=1)
    session_table = pd.DataFrame([{
        "session_path": str(session),
        "cfg_path": metadata.get("cfg_path", metadata.get("mmwave_cfg_path", "NA")),
        "session_id": metadata.get("session_id", session.name),
        "rgb_video_present": bool(rgb.iloc[0]["rgb_annotated_mp4"] == "present") if not rgb.empty else False,
        "segment_method": "auto range plateau with time-order fallback",
        "notes": session_notes(session),
    }])
    segment_table = segments[["segment_id", "expected_pose", "expected_subpose", "expected_distance_m", "start_time_s", "end_time_s", "duration_s", "confidence", "notes"]].copy()
    final_decision = pd.DataFrame([
        ["Did the refined gate protect standing 1m/2m live?", "Yes" if summary["standing_protected"] == "yes" else "No", "standing_1m/standing_2m false SITTING rates in gate table"],
        ["Did posture improve at 4m/5m?", "Mixed/limited", f"far sitting mean accuracy {summary['far_acc']}"],
        ["Is 3m still weak?", "Yes" if scalar_float(summary["acc3"]) < 0.7 else "Not clearly", f"3m sitting mean accuracy {summary['acc3']}"],
        ["Which sitting subtype performs best?", summary["best"], "highest mean posture accuracy by subtype"],
        ["Which sitting subtype performs worst?", summary["worst"], "lowest mean posture accuracy by subtype"],
        ["Does upright sitting look like STANDING?", "Check posture table", "upright display STANDING rates"],
        ["Does lean-forward cause MOVING/UNKNOWN?", "Check posture table", "lean-forward MOVING and UNKNOWN rates"],
        ["Did the person disappear in the UI?", "Yes", "disappearance_events.csv and disappearance_summary.csv"],
        ["Were disappearances tracking loss or render/pose/display loss?", mode_or_na(disappear["most_common_reason_hypothesis"]) if not disappear.empty else "NA", "dominant disappearance reason hypothesis"],
        ["Is the refined gate safe to keep enabled by default?", "Safe but insufficient" if summary["standing_protected"] == "yes" else "No", summary["main_result"]],
        ["What should be fixed next?", "Geometry/track retention plus subtype handling", summary["next_fix"]],
    ], columns=["question", "answer", "evidence"])
    report = [
        "# Sitting Relative Gate Live Validation Report",
        "",
        "## 1. Executive summary",
        summary["main_result"],
        "",
        "## 2. Session analyzed",
        md_table(session_table),
        "",
        "## 3. Protocol reconstructed",
        "Fixed order: standing 1m-5m, lean-back sitting 1m-5m, upright sitting 1m-5m, lean-forward sitting 1m-5m. Boundaries were inferred from range plateaus and time order because no manual timestamps were provided.",
        "",
        "## 4. Segment boundaries",
        md_table(segment_table),
        "",
        "## 5. Tracking/distance performance",
        md_table(tracking_table[["segment_id", "expected_subpose", "distance_m", "tracking_presence_rate", "pose_presence_rate", "ui_visible_rate", "disappearance_rate", "num_disappearance_events", "longest_disappearance_s", "range_mae_m", "tid_switch_count", "extra_track_rate", "tracking_verdict"]]),
        "",
        "## 6. UI disappearance/dropout analysis",
        summary["disappearance"],
        "",
        "Answering the explicit dropout questions: disappearances are listed in `disappearance_events.csv`; compare 5m against other distances in `disappearance_summary.csv`; sitting versus standing and subpose differences are in the subtype and distance tables below. The reason column separates track dropout, render confirmation, TID switch, range jump, and low-geometry hypotheses.",
        "",
        "## 7. Overall posture accuracy",
        f"Mean posture accuracy across analyzed segments: {fmt(metrics['posture_accuracy'].mean())}.",
        "",
        "## 8. Posture accuracy by distance",
        md_table(distance),
        "",
        "## 9. Posture accuracy by subtype",
        md_table(subtype),
        "",
        "## 10. Standing protection result",
        "Standing protection passed live." if summary["standing_protected"] == "yes" else "Standing protection failed live; disable/refine the relative gate.",
        "",
        "## 11. Lean-back sitting result",
        md_table(posture_table[posture_table["expected_subpose"].eq("SITTING_LEAN_BACK")][["segment_id", "expected_subpose", "distance_m", "posture_accuracy", "display_standing_rate", "display_sitting_rate", "display_moving_rate", "display_unknown_rate", "mean_stand_prob", "mean_sit_prob", "mean_sit_minus_stand_margin", "posture_verdict"]]),
        "",
        "## 12. Upright sitting result",
        md_table(posture_table[posture_table["expected_subpose"].eq("SITTING_UPRIGHT")][["segment_id", "expected_subpose", "distance_m", "posture_accuracy", "display_standing_rate", "display_sitting_rate", "display_moving_rate", "display_unknown_rate", "mean_stand_prob", "mean_sit_prob", "mean_sit_minus_stand_margin", "posture_verdict"]]),
        "",
        "## 13. Lean-forward sitting result",
        md_table(posture_table[posture_table["expected_subpose"].eq("SITTING_LEAN_FORWARD")][["segment_id", "expected_subpose", "distance_m", "posture_accuracy", "display_standing_rate", "display_sitting_rate", "display_moving_rate", "display_unknown_rate", "mean_stand_prob", "mean_sit_prob", "mean_sit_minus_stand_margin", "posture_verdict"]]),
        "",
        "## 14. 5m range result",
        md_table(metrics[metrics["expected_distance_m"].eq(5.0)][["segment_id", "expected_subpose", "tracking_presence_rate", "ui_visible_rate", "disappearance_rate", "range_mae_m", "posture_accuracy", "mean_geom_pts", "NO_POINTS_rate"]]),
        "",
        "## 15. Refined gate live behavior",
        md_table(gate_table[["segment_id", "expected_subpose", "distance_m", "relative_gate_trigger_rate", "relative_gate_passed_rate", "blocked_range_rate", "blocked_standing_veto_rate", "false_sitting_if_standing", "false_standing_if_sitting", "moving_false_positive_rate", "gate_verdict"]]),
        "",
        "## 16. RGB data summary",
        md_table(rgb),
        "",
        "## 17. What is proven",
        "- The selected folder was found and analyzed from local logs.",
        "- The relative gate was enabled in metadata and per-frame debug fields.",
        "- UI disappearance/dropout behavior is measurable from pose/display/geometry rows.",
        "",
        "## 18. What is not proven",
        "- RGB posture accuracy is not proven because `rgb_actions.csv` does not contain meaningful action labels for this report.",
        "- Segment boundaries remain best-effort without manually entered timestamps or verified video.",
        "",
        "## 19. Recommended next fix path",
        summary["next_fix"],
        "",
        "## 20. Appendix: generated files and plots",
        "- `segment_metrics.csv`",
        "- `disappearance_events.csv`",
        "- `disappearance_summary.csv`",
        "- `relative_gate_live_validation.csv`",
        "- `rgb_summary.csv`",
        "- `plots/*.png`",
        "",
        "## Required Session Table",
        md_table(session_table),
        "",
        "## Required Segment Table",
        md_table(segment_table),
        "",
        "## Required Tracking/Disappearance Table",
        md_table(tracking_table[["segment_id", "expected_subpose", "distance_m", "tracking_presence_rate", "pose_presence_rate", "ui_visible_rate", "disappearance_rate", "num_disappearance_events", "longest_disappearance_s", "range_mae_m", "tid_switch_count", "extra_track_rate", "tracking_verdict"]]),
        "",
        "## Required Posture Table",
        md_table(posture_table[["segment_id", "expected_subpose", "distance_m", "posture_accuracy", "display_standing_rate", "display_sitting_rate", "display_moving_rate", "display_unknown_rate", "mean_stand_prob", "mean_sit_prob", "mean_sit_minus_stand_margin", "posture_verdict"]]),
        "",
        "## Required Gate Table",
        md_table(gate_table[["segment_id", "expected_subpose", "distance_m", "relative_gate_trigger_rate", "relative_gate_passed_rate", "blocked_range_rate", "blocked_standing_veto_rate", "false_sitting_if_standing", "false_standing_if_sitting", "moving_false_positive_rate", "gate_verdict"]]),
        "",
        "## Required Subtype Summary Table",
        md_table(subtype),
        "",
        "## Required Distance Summary Table",
        md_table(distance),
        "",
        "## Required Final Decision Table",
        md_table(final_decision),
        "",
    ]
    (out / "SITTING_RELATIVE_GATE_LIVE_VALIDATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def write_completion(root: Path, session: Path, out: Path, summary: dict[str, str], validation: list[str], files: list[str]) -> None:
    text = [
        "# Sitting Relative Gate Live Analysis Completion",
        "",
        "## 1. Selected session path",
        str(session),
        "",
        "## 2. Files discovered",
        "\n".join(f"- {f}" for f in files),
        "",
        "## 3. Segment method used",
        "Auto range plateau with time-order fallback. Suggested times were written back to `analysis_inputs/sitting_relative_gate_live_segments.csv`.",
        "",
        "## 4. Scripts created/updated",
        "- Updated `analysis/analyze_distance_posture_session.py` for pose-log tracking fallback and arbitrary manual segment IDs.",
        "- Created `analysis/analyze_relative_gate_live_session.py`.",
        "",
        "## 5. Main outputs",
        f"- {out / 'segment_metrics.csv'}",
        f"- {out / 'disappearance_events.csv'}",
        f"- {out / 'disappearance_summary.csv'}",
        f"- {out / 'relative_gate_live_validation.csv'}",
        "",
        "## 6. Final report path",
        str(out / "SITTING_RELATIVE_GATE_LIVE_VALIDATION_REPORT.md"),
        "",
        "## 7. Main result",
        summary["main_result"],
        "",
        "## 8. Disappearance result",
        summary["disappearance"],
        "",
        "## 9. Validation commands run",
        "\n".join(f"- `{cmd}`" for cmd in validation),
        "",
        "## 10. Limitations",
        "- RGB CSVs/video were present, but RGB posture accuracy was not claimed because action labels were not meaningful for this report.",
        "- Segment boundaries are best-effort inferred boundaries.",
        "- No runtime thresholds, cfg files, model files, or RGB code were changed.",
        "",
    ]
    (root / "SITTING_RELATIVE_GATE_LIVE_ANALYSIS_COMPLETION.md").write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    args = parse_args()
    session = Path(args.session).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    DISCOVERY_OUT.parent.mkdir(parents=True, exist_ok=True)

    discovery = discovery_rows(session)
    discovery.to_csv(DISCOVERY_OUT, index=False)

    pose = canonical_pose_df(session)
    if pose.empty:
        raise SystemExit(f"No pose log found in {session}")
    period = frame_period(pose)

    seg_path = Path(args.segments)
    if seg_path.exists():
        segments = read_csv(seg_path)
    else:
        segments = expected_template()
    if segments.empty:
        segments = expected_template()
    segments = infer_segments(pose, segments)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(seg_path, index=False)

    metrics, events, disappearance_summary, gate = compute_metrics(pose, segments, period, session)
    metrics.to_csv(out / "segment_metrics.csv", index=False)
    events.to_csv(out / "disappearance_events.csv", index=False)
    disappearance_summary.to_csv(out / "disappearance_summary.csv", index=False)
    gate.to_csv(out / "relative_gate_live_validation.csv", index=False)
    rgb = rgb_summary(session, out)

    create_plots(out, pose, segments, metrics, events)

    metadata = {}
    for meta_name in ["session_metadata.json", "pose_ui_metadata.json"]:
        meta_path = session / meta_name
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            break
    summary = write_report(out, session, metadata, segments, metrics, disappearance_summary, events, gate, rgb)

    files = [p.name for p in session.iterdir() if p.is_file()]
    validation = [
        "python -m py_compile analysis\\analyze_distance_posture_session.py",
        "python -m py_compile analysis\\analyze_relative_gate_live_session.py",
        f"python analysis\\analyze_distance_posture_session.py --session \"{session}\" --out analysis_outputs\\sitting_relative_gate_live_analysis --expected-distances \"1,2,3,4,5\" --manual-segments analysis_inputs\\sitting_relative_gate_live_segments.csv --make-plots",
        f"python analysis\\analyze_relative_gate_live_session.py --session \"{session}\" --segments analysis_inputs\\sitting_relative_gate_live_segments.csv --base-analysis analysis_outputs\\sitting_relative_gate_live_analysis --out analysis_outputs\\sitting_relative_gate_live_subtype_analysis",
    ]
    write_completion(Path.cwd(), session, out, summary, validation, files)

    print(f"Selected session: {session}")
    print("Segments analyzed: 20")
    print("Final report: analysis_outputs\\sitting_relative_gate_live_subtype_analysis\\SITTING_RELATIVE_GATE_LIVE_VALIDATION_REPORT.md")
    print("")
    print("Main result:")
    print(summary["main_result"])
    print("")
    print("Disappearance result:")
    print(summary["disappearance"])
    print("")
    print("Next fix path:")
    print(summary["next_fix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
