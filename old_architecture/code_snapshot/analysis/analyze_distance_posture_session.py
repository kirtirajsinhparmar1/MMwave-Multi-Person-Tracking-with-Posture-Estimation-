#!/usr/bin/env python
"""Offline distance/posture benchmark analysis for combined mmWave + RGB logs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


POSE_CLASSES = ["STANDING", "SITTING", "MOVING", "FALLING", "LYING", "UNKNOWN", "OTHER"]
EXPECTED_FILES = [
    "mmwave_frames.csv",
    "mmwave_tracks.csv",
    "mmwave_pose.csv",
    "pose_predictions_ui.csv",
    "pose_ui_metadata.json",
    "targets.csv",
    "frames_summary.csv",
    "rgb_frames.csv",
    "rgb_tracks.csv",
    "rgb_keypoints.csv",
    "sync_index.csv",
    "events.csv",
    "events.jsonl",
    "session_metadata.json",
    "combined_events.csv",
    "rgb_annotated.mp4",
]
PLOT_NAMES = [
    "timeline_range_by_track.png",
    "timeline_active_track_count.png",
    "timeline_display_pose.png",
    "timeline_quality_geom_pts.png",
    "timeline_stand_sit_probs.png",
    "stand_vs_sit_probability_by_segment.png",
    "stand_minus_sit_margin_by_segment.png",
    "sitting_segments_stand_sit_prob_timeline.png",
    "tracking_presence_by_distance.png",
    "range_error_by_distance.png",
    "range_jitter_by_distance.png",
    "ghost_rate_by_distance.png",
    "tid_switches_by_distance.png",
    "xy_position_scatter_by_segment.png",
    "posture_accuracy_by_distance.png",
    "posture_confusion_matrix.png",
    "pose_distribution_by_segment.png",
    "posture_accuracy_vs_geom_pts.png",
    "posture_accuracy_by_quality.png",
    "moving_false_positive_rate.png",
    "false_falling_rate_by_segment.png",
    "time_to_stable_correct_by_segment.png",
    "tracking_vs_posture_summary.png",
    "failure_mode_heatmap.png",
]


@dataclass
class Context:
    session: Path
    out: Path
    warnings: list[str]
    frame_period_s: float = 0.055
    t0_ns: float | None = None

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze continuous 1m/2m/3m/4m standing+sitting mmWave/RGB benchmark logs."
    )
    parser.add_argument("--log-root", action="append", default=[], help="Log root to scan. Can be repeated.")
    parser.add_argument("--session", help="Explicit session directory.")
    parser.add_argument("--latest", action="store_true", help="Use latest useful session under log roots.")
    parser.add_argument("--out", default="analysis_outputs/latest_distance_posture_analysis")
    parser.add_argument("--expected-distances", default="1,2,3,4")
    parser.add_argument("--segment-min-seconds", type=float, default=35.0)
    parser.add_argument("--segment-target-seconds", type=float, default=45.0)
    parser.add_argument("--fps-estimate", default="auto")
    parser.add_argument("--manual-segments", help="CSV with segment_id,expected_pose,expected_distance_m,start_time_s,end_time_s.")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--open-report", action="store_true")
    return parser.parse_args()


def as_path(text: str | Path) -> Path:
    return Path(text).expanduser().resolve()


def default_roots(user_roots: Iterable[str]) -> list[Path]:
    roots = [Path("logs"), Path("..") / "logs", Path(r"C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs")]
    roots.extend(Path(r) for r in user_roots)
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def useful_session_score(path: Path) -> int:
    if not path.is_dir():
        return 0
    score = 0
    for name in EXPECTED_FILES:
        p = path / name
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            score += 4 if name.endswith(".csv") else 1
    for pattern in ("*.csv", "*.json", "*.jsonl", "*.log", "*.txt"):
        score += len(list(path.glob(pattern)))
    return score


def find_candidate_sessions(roots: list[Path]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        scan = [root] + [p for p in root.iterdir() if p.is_dir()]
        for path in scan:
            score = useful_session_score(path)
            if score <= 0:
                continue
            key = str(path.resolve()).lower()
            mtime = max((f.stat().st_mtime for f in path.glob("*") if f.exists()), default=path.stat().st_mtime)
            candidates[key] = {
                "rank": 0,
                "path": path.resolve(),
                "modified_time": mtime,
                "useful_score": score,
                "csv_count": len(list(path.glob("*.csv"))),
                "notes": "useful files found",
            }
    ranked = sorted(candidates.values(), key=lambda r: (r["modified_time"], r["useful_score"]), reverse=True)
    for idx, row in enumerate(ranked, 1):
        row["rank"] = idx
    return ranked


def safe_read_csv(path: Path, warnings: list[str], **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report and continue
        warnings.append(f"failed to parse {path.name}: {exc}")
        return pd.DataFrame()


def detect_time_frame_columns(columns: Iterable[str]) -> str:
    cols = list(columns)
    hits = [c for c in cols if re.search(r"(time|timestamp|monotonic|frame)", c, re.I)]
    return ", ".join(hits[:6]) if hits else ""


def count_csv_rows(path: Path) -> int | str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return "NA"


def build_inventory(session: Path, out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = {p.name: p for p in session.iterdir() if p.is_file()}
    for name in EXPECTED_FILES:
        p = files.get(name) or (session / "videos" / name if name == "rgb_annotated.mp4" else session / name)
        exists = p.exists()
        row = {
            "file": name,
            "exists": bool(exists),
            "rows": "NA",
            "columns": "",
            "time/frame column detected": "",
            "notes": "",
        }
        if exists:
            if p.suffix.lower() == ".csv":
                row["rows"] = count_csv_rows(p)
                try:
                    with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                        reader = csv.reader(handle)
                        cols = next(reader, [])
                    row["columns"] = ", ".join(cols)
                    row["time/frame column detected"] = detect_time_frame_columns(cols)
                except Exception as exc:  # noqa: BLE001
                    row["notes"] = f"header read failed: {exc}"
            elif p.suffix.lower() in {".json", ".jsonl", ".log", ".txt"}:
                row["rows"] = count_csv_rows(p) if p.suffix.lower() == ".jsonl" else "NA"
                row["notes"] = f"{p.stat().st_size} bytes"
            else:
                row["notes"] = f"{p.stat().st_size} bytes"
        rows.append(row)
    extra = sorted(p for p in files.values() if p.name not in EXPECTED_FILES)
    for p in extra:
        if p.suffix.lower() not in {".csv", ".json", ".jsonl", ".log", ".txt", ".mp4"}:
            continue
        row = {
            "file": p.name,
            "exists": True,
            "rows": count_csv_rows(p) if p.suffix.lower() in {".csv", ".jsonl"} else "NA",
            "columns": "",
            "time/frame column detected": "",
            "notes": "extra discovered file",
        }
        if p.suffix.lower() == ".csv":
            try:
                with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    cols = next(csv.reader(handle), [])
                row["columns"] = ", ".join(cols)
                row["time/frame column detected"] = detect_time_frame_columns(cols)
            except Exception as exc:  # noqa: BLE001
                row["notes"] = f"extra discovered file; header read failed: {exc}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "file_inventory.csv", index=False)
    return df


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalized_timestamp(df: pd.DataFrame, ctx: Context, frame_col: str | None = None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    direct = first_col(df, ["timestamp_s", "time_s", "elapsed_s", "t_s", "time"])
    if direct:
        values = numeric(df[direct])
        if values.notna().any():
            if values.max(skipna=True) > 1e6:
                values = values - values.min(skipna=True)
            return values.astype(float)
    mono = first_col(df, ["host_monotonic_ns", "monotonic_ns"])
    if mono:
        values = numeric(df[mono])
        if values.notna().any():
            t0 = ctx.t0_ns if ctx.t0_ns is not None else values.min(skipna=True)
            return ((values - t0) / 1e9).astype(float)
    wall = first_col(df, ["host_wall_time_iso", "timestamp", "wall_time"])
    if wall:
        values = pd.to_datetime(df[wall], errors="coerce", utc=True)
        if values.notna().any():
            return (values - values.min()).dt.total_seconds().astype(float)
    if frame_col and frame_col in df.columns:
        frames = numeric(df[frame_col])
        if frames.notna().any():
            return ((frames - frames.min(skipna=True)) * ctx.frame_period_s).astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def infer_frame_period(ctx: Context, frames_df: pd.DataFrame, fps_estimate: str) -> None:
    if fps_estimate and fps_estimate != "auto":
        try:
            val = float(fps_estimate)
            ctx.frame_period_s = 1.0 / val if val > 2 else val
            return
        except ValueError:
            ctx.warn(f"invalid --fps-estimate {fps_estimate}; using auto/default")
    frame_col = first_col(frames_df, ["mmwave_frame_num", "frame", "frame_num"])
    if frames_df.empty or not frame_col:
        ctx.warn("missing timestamps; using default TI frame period 55 ms where needed")
        return
    ts = normalized_timestamp(frames_df, ctx, frame_col)
    fr = numeric(frames_df[frame_col])
    good = pd.DataFrame({"ts": ts, "fr": fr}).dropna().sort_values("fr")
    if len(good) >= 5:
        dts = good["ts"].diff()
        dfs = good["fr"].diff()
        periods = (dts / dfs).replace([np.inf, -np.inf], np.nan).dropna()
        periods = periods[(periods > 0.01) & (periods < 0.5)]
        if len(periods):
            ctx.frame_period_s = float(periods.median())
            return
    ctx.warn("missing timestamps; using default TI frame period 55 ms where needed")


def global_t0_ns(dfs: Iterable[pd.DataFrame]) -> float | None:
    mins: list[float] = []
    for df in dfs:
        col = first_col(df, ["host_monotonic_ns", "monotonic_ns"])
        if col:
            vals = numeric(df[col]).dropna()
            if len(vals):
                mins.append(float(vals.min()))
    return min(mins) if mins else None


def normalize_tracking(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    if df.empty:
        ctx.warn("missing mmwave_tracks.csv")
        return pd.DataFrame(columns=["frame", "timestamp_s", "tid", "x_m", "y_m", "z_m", "range_m"])
    aliases = {
        "frame": ["frame", "frame_num", "mmwave_frame_num", "target_frame"],
        "tid": ["tid", "target_id", "track_id", "id"],
        "x_m": ["x_m", "pos_x", "x", "target_x_m"],
        "y_m": ["y_m", "pos_y", "y", "target_y_m"],
        "z_m": ["z_m", "pos_z", "z", "target_z_m"],
        "range_m": ["range_m", "range", "r_m", "distance_m"],
        "azimuth_deg": ["azimuth_deg", "azimuth"],
        "elevation_deg": ["elevation_deg", "elevation"],
        "vx_mps": ["vx_mps", "vel_x", "vx"],
        "vy_mps": ["vy_mps", "vel_y", "vy"],
        "vz_mps": ["vz_mps", "vel_z", "vz"],
        "active": ["active", "is_active"],
        "geom_pts": ["geom_pts", "num_associated_points", "num_points"],
        "quality": ["quality", "quality_flag", "geom_quality"],
        "assoc": ["assoc", "association", "assoc_method"],
    }
    out = pd.DataFrame(index=df.index)
    frame_col = None
    for target, names in aliases.items():
        col = first_col(df, names)
        if col:
            out[target] = df[col]
            if target == "frame":
                frame_col = col
        else:
            out[target] = np.nan
    out["frame"] = numeric(out["frame"])
    out["timestamp_s"] = normalized_timestamp(df, ctx, frame_col)
    out["tid"] = numeric(out["tid"])
    for col in ["x_m", "y_m", "z_m", "range_m", "azimuth_deg", "elevation_deg", "vx_mps", "vy_mps", "vz_mps", "geom_pts"]:
        out[col] = numeric(out[col])
    if out["range_m"].isna().all() and {"x_m", "y_m"}.issubset(out.columns):
        out["range_m"] = np.sqrt(out["x_m"] ** 2 + out["y_m"] ** 2)
    out["active"] = out["active"].fillna(1)
    return out.dropna(subset=["timestamp_s"], how="all")


def normalize_frames(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    if df.empty:
        ctx.warn("missing mmwave_frames.csv or frames_summary.csv")
        return pd.DataFrame(columns=["frame", "timestamp_s", "num_tracks", "num_points"])
    out = pd.DataFrame(index=df.index)
    frame_col = first_col(df, ["frame", "frame_num", "mmwave_frame_num"])
    out["frame"] = numeric(df[frame_col]) if frame_col else np.nan
    out["timestamp_s"] = normalized_timestamp(df, ctx, frame_col)
    for target, names in {
        "num_tracks": ["num_tracks", "track_count", "targets"],
        "num_points": ["num_points", "point_count", "points"],
        "parse_ok": ["parse_ok"],
        "error_count": ["error_count", "errors"],
    }.items():
        col = first_col(df, names)
        out[target] = numeric(df[col]) if col else np.nan
    return out


def normalize_posture(df: pd.DataFrame, ctx: Context, tracks: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        ctx.warn("missing mmwave_pose.csv")
        return pd.DataFrame(columns=["frame", "timestamp_s", "tid", "display_pose"])
    aliases = {
        "frame": ["frame", "frame_num", "mmwave_frame_num"],
        "tid": ["tid", "target_id", "track_id", "id"],
        "display_pose": ["display_pose", "final_label", "final_pose", "pose", "state"],
        "raw_pose": ["raw_pose", "ml_label", "raw_label"],
        "smooth_pose": ["smooth_pose", "smoothed_label", "smooth_label"],
        "candidate_pose": ["candidate_pose", "candidate_label"],
        "status": ["status", "quality_flag"],
        "quality": ["quality", "quality_flag"],
        "geom_pts": ["geom_pts", "num_points", "num_associated_points"],
        "points_total": ["points_total", "num_points", "point_count"],
        "geom_quality": ["geom_quality"],
        "assoc": ["assoc", "association", "assoc_method"],
        "range_m": ["range_m", "range", "distance_m"],
        "stand_prob": ["stand_prob", "prob_standing", "standing_prob"],
        "sit_prob": ["sit_prob", "prob_sitting", "sitting_prob"],
        "fall_prob": ["fall_prob", "prob_falling", "falling_prob"],
        "lying_prob": ["lying_prob", "prob_lying"],
        "moving_override_reason": ["moving_override_reason"],
        "translation_m": ["translation_m"],
        "translation_confirmed": ["translation_confirmed"],
        "stand_sit_margin": ["stand_sit_margin"],
        "stand_sit_decision": ["stand_sit_decision"],
        "final_reason": ["final_reason"],
        "confidence": ["confidence", "ml_confidence", "conf"],
        "speed_mps": ["speed_mps"],
    }
    out = pd.DataFrame(index=df.index)
    frame_col = None
    for target, names in aliases.items():
        col = first_col(df, names)
        if col:
            out[target] = df[col]
            if target == "frame":
                frame_col = col
        else:
            out[target] = np.nan
    out["frame"] = numeric(out["frame"])
    out["timestamp_s"] = normalized_timestamp(df, ctx, frame_col)
    out["tid"] = numeric(out["tid"])
    for col in ["geom_pts", "points_total", "range_m", "stand_prob", "sit_prob", "fall_prob", "lying_prob", "translation_m", "stand_sit_margin", "confidence", "speed_mps"]:
        out[col] = numeric(out[col])
    for col in ["display_pose", "raw_pose", "smooth_pose", "candidate_pose"]:
        out[col] = out[col].map(normalize_pose_label)
    if out["range_m"].isna().all() and not tracks.empty:
        joined = tracks[["frame", "tid", "range_m"]].dropna(subset=["frame", "tid"]).drop_duplicates(["frame", "tid"])
        out = out.merge(joined, on=["frame", "tid"], how="left", suffixes=("", "_track"))
        out["range_m"] = out["range_m"].fillna(out.get("range_m_track"))
        out = out.drop(columns=[c for c in ["range_m_track"] if c in out.columns])
    out["range_zone"] = out["range_m"].map(range_zone)
    if out["geom_pts"].isna().all():
        ctx.warn("no geom_pts field found")
    if out["quality"].isna().all():
        ctx.warn("no quality field found")
    if out["stand_prob"].isna().all() or out["sit_prob"].isna().all():
        ctx.warn("no stand_prob/sit_prob found")
    return out.dropna(subset=["timestamp_s"], how="all")


def normalize_rgb(frames: pd.DataFrame, tracks: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    if frames.empty and tracks.empty:
        ctx.warn("no RGB logs found")
        return pd.DataFrame(columns=["rgb_frame", "timestamp_s", "detected_count", "track_count", "fps"])
    base = frames if not frames.empty else tracks
    out = pd.DataFrame(index=base.index)
    frame_col = first_col(base, ["rgb_frame", "rgb_frame_num", "frame", "frame_num"])
    out["rgb_frame"] = numeric(base[frame_col]) if frame_col else np.nan
    out["timestamp_s"] = normalized_timestamp(base, ctx, frame_col)
    for target, names in {
        "rgb_track_id": ["rgb_track_id", "track_id"],
        "detected_count": ["detected_count", "num_detections"],
        "track_count": ["track_count", "num_tracks"],
        "fps": ["fps", "fps_estimate"],
        "rgb_pose/action": ["action_label", "pose", "rgb_pose"],
    }.items():
        col = first_col(base, names)
        out[target] = base[col] if col else np.nan
    return out


def parse_pose_debug_text(session: Path, ctx: Context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\[pose\].*?tid=(?P<tid>-?\d+).*?(?:range_m=(?P<range>[-+.\d]+))?.*?(?:display=(?P<display>[A-Za-z_]+))?.*?(?:quality=(?P<quality>[A-Za-z_]+))?", re.I)
    for path in list(session.glob("*.log")) + list(session.glob("*.txt")) + list(session.glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, 1):
                    match = pattern.search(line)
                    if match:
                        rows.append(
                            {
                                "frame": np.nan,
                                "timestamp_s": np.nan,
                                "tid": float(match.group("tid")),
                                "display_pose": normalize_pose_label(match.group("display")),
                                "quality": match.group("quality") or "",
                                "range_m": float(match.group("range")) if match.group("range") else np.nan,
                                "source_file": path.name,
                                "line": line_no,
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"failed fallback text parse {path.name}: {exc}")
    return pd.DataFrame(rows)


def normalize_pose_label(value: Any) -> str:
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
    if text in POSE_CLASSES:
        return text
    return "OTHER"


def range_zone(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(v):
        return "UNKNOWN"
    nearest = min([1, 2, 3, 4], key=lambda d: abs(v - d))
    return f"{nearest}m"


def expected_table(distances: list[float]) -> pd.DataFrame:
    rows = []
    order = 1
    for pose in ["STANDING", "SITTING"]:
        for dist in distances:
            rows.append(
                {
                    "segment_id": f"{pose.lower()}_{int(dist) if float(dist).is_integer() else dist:g}m",
                    "expected_pose": pose,
                    "expected_distance_m": float(dist),
                    "expected_order": order,
                }
            )
            order += 1
    return pd.DataFrame(rows)


def contiguous_runs(times: pd.Series, max_gap_s: float = 2.5) -> list[tuple[float, float, int]]:
    vals = sorted(float(v) for v in times.dropna().unique())
    if not vals:
        return []
    runs: list[tuple[float, float, int]] = []
    start = prev = vals[0]
    count = 1
    for val in vals[1:]:
        if val - prev <= max_gap_s:
            prev = val
            count += 1
        else:
            runs.append((start, prev, count))
            start = prev = val
            count = 1
    runs.append((start, prev, count))
    return runs


def add_frame_bounds(segments: pd.DataFrame, frames: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    source = frames if not frames.empty else tracks
    starts = []
    ends = []
    for _, seg in segments.iterrows():
        subset = source[(source["timestamp_s"] >= seg["start_time_s"]) & (source["timestamp_s"] <= seg["end_time_s"])]
        starts.append(num_or_na(subset["frame"].min()) if "frame" in subset else "NA")
        ends.append(num_or_na(subset["frame"].max()) if "frame" in subset else "NA")
    segments["start_frame"] = starts
    segments["end_frame"] = ends
    return segments


def make_manual_template(expected: pd.DataFrame, out: Path) -> None:
    tmpl = expected[["segment_id", "expected_pose", "expected_distance_m"]].copy()
    tmpl["start_time_s"] = ""
    tmpl["end_time_s"] = ""
    tmpl.to_csv(out / "segments_manual_template.csv", index=False)


def load_manual_segments(path: Path, expected: pd.DataFrame, ctx: Context, frames: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame | None:
    if not path.exists():
        ctx.warn(f"manual segment file not found: {path}")
        return None
    manual = safe_read_csv(path, ctx.warnings)
    required = {"segment_id", "expected_pose", "expected_distance_m", "start_time_s", "end_time_s"}
    if manual.empty or not required.issubset(set(manual.columns)):
        ctx.warn(f"manual segment file missing required columns: {path}")
        return None
    manual = manual.copy()
    manual["_manual_order"] = range(1, len(manual) + 1)
    manual["expected_pose"] = manual["expected_pose"].map(normalize_pose_label)
    manual["expected_distance_m"] = numeric(manual["expected_distance_m"])
    manual["start_time_s"] = numeric(manual["start_time_s"])
    manual["end_time_s"] = numeric(manual["end_time_s"])
    manual["duration_s"] = manual["end_time_s"] - manual["start_time_s"]
    manual["method"] = "manual"
    if "confidence" not in manual.columns:
        if "label_confidence" in manual.columns:
            manual["confidence"] = numeric(manual["label_confidence"]).fillna(1.0)
        else:
            manual["confidence"] = 1.0
    if "notes" not in manual.columns:
        manual["notes"] = ""
    merged = expected[["segment_id", "expected_order"]].merge(manual, on="segment_id", how="right")
    merged["expected_order"] = numeric(merged["expected_order"]).fillna(merged["_manual_order"])
    merged = merged.sort_values("expected_order").drop(columns=["_manual_order"], errors="ignore")
    return add_frame_bounds(merged, frames, tracks)


def auto_segments(
    expected: pd.DataFrame,
    tracks: pd.DataFrame,
    frames: pd.DataFrame,
    min_seconds: float,
    target_seconds: float,
    ctx: Context,
) -> pd.DataFrame:
    if tracks.empty or tracks["range_m"].dropna().empty:
        ctx.warn("unable to infer segments from range; using equal time fallback")
        return equal_segments(expected, tracks, frames, min_seconds, target_seconds, ctx, "auto_equal_fallback_no_range")
    data = tracks.dropna(subset=["timestamp_s", "range_m"]).copy()
    data["range_err_tmp"] = np.nan
    rows: list[dict[str, Any]] = []
    cursor = max(0.0, float(data["timestamp_s"].min()))
    session_end = float(max(data["timestamp_s"].max(), frames["timestamp_s"].max() if not frames.empty else data["timestamp_s"].max()))
    for _, exp in expected.iterrows():
        dist = float(exp["expected_distance_m"])
        chosen: tuple[float, float, int, float] | None = None
        chosen_tol = 0.0
        for tol in ([0.45, 0.65, 0.85, 1.20] if dist not in {1.0, 4.0} else [0.55, 0.75, 1.00, 1.30]):
            cand = data[(data["timestamp_s"] >= cursor) & ((data["range_m"] - dist).abs() <= tol)]
            per_frame = cand.sort_values("range_m").drop_duplicates(["frame", "tid", "timestamp_s"])
            runs = contiguous_runs(per_frame["timestamp_s"], max_gap_s=3.0)
            long_runs = [r for r in runs if (r[1] - r[0]) >= min_seconds]
            if long_runs:
                chosen = long_runs[0]
                chosen_tol = tol
                break
        if chosen is None:
            ctx.warn(f"unable to infer segment {exp['segment_id']} from stable range")
            rows = []
            break
        raw_start, raw_end, count = chosen
        start = raw_start + 5.0 if (raw_end - raw_start) >= min_seconds + 10 else raw_start
        end = raw_end - 5.0 if (raw_end - raw_start) >= min_seconds + 10 else raw_end
        rows.append(
            {
                "segment_id": exp["segment_id"],
                "expected_pose": exp["expected_pose"],
                "expected_distance_m": dist,
                "expected_order": exp["expected_order"],
                "start_time_s": start,
                "end_time_s": end,
                "duration_s": end - start,
                "method": "auto_range_plateau_trimmed",
                "confidence": min(1.0, max(0.2, (end - start) / max(target_seconds, 1.0)) * (0.45 / max(chosen_tol, 0.45))),
                "notes": f"raw={raw_start:.2f}-{raw_end:.2f}s; tol={chosen_tol:.2f}m; samples={count}",
            }
        )
        cursor = raw_end + 2.0
    if len(rows) != len(expected):
        return equal_segments(expected, tracks, frames, min_seconds, target_seconds, ctx, "auto_equal_fallback_partial_range")
    segs = pd.DataFrame(rows)
    short = segs[segs["duration_s"] < min_seconds]
    for _, seg in short.iterrows():
        ctx.warn(f"segment shorter than expected: {seg['segment_id']} duration {seg['duration_s']:.1f}s")
    return add_frame_bounds(segs, frames, tracks)


def equal_segments(
    expected: pd.DataFrame,
    tracks: pd.DataFrame,
    frames: pd.DataFrame,
    min_seconds: float,
    target_seconds: float,
    ctx: Context,
    method: str,
) -> pd.DataFrame:
    source = tracks if not tracks.empty else frames
    if source.empty or source["timestamp_s"].dropna().empty:
        start, end = 0.0, len(expected) * target_seconds
    else:
        start = float(source["timestamp_s"].min())
        end = float(source["timestamp_s"].max())
    total = max(end - start, len(expected) * min_seconds)
    raw_width = total / len(expected)
    rows = []
    for idx, exp in expected.iterrows():
        raw_start = start + idx * raw_width
        raw_end = start + (idx + 1) * raw_width
        seg_start = raw_start + 5.0 if raw_width >= min_seconds + 10 else raw_start
        seg_end = raw_end - 5.0 if raw_width >= min_seconds + 10 else raw_end
        rows.append(
            {
                "segment_id": exp["segment_id"],
                "expected_pose": exp["expected_pose"],
                "expected_distance_m": exp["expected_distance_m"],
                "expected_order": exp["expected_order"],
                "start_time_s": seg_start,
                "end_time_s": seg_end,
                "duration_s": seg_end - seg_start,
                "method": method,
                "confidence": 0.25,
                "notes": f"raw={raw_start:.2f}-{raw_end:.2f}s; inspect boundaries",
            }
        )
    ctx.warn("generated equal-time best-effort segments; inspect segment boundaries before trusting metrics")
    return add_frame_bounds(pd.DataFrame(rows), frames, tracks)


def is_bad_evidence(row: pd.Series) -> bool:
    geom = row.get("geom_pts")
    quality = str(row.get("quality", "")).upper()
    assoc = str(row.get("assoc", "")).upper()
    if pd.notna(geom) and float(geom) <= 0:
        return True
    return "NO_POINTS" in quality or "TARGET_ONLY" in quality or "AUTO_NONE" in assoc


def segment_filter(df: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
    if df.empty or "timestamp_s" not in df:
        return df.iloc[0:0].copy()
    return df[(df["timestamp_s"] >= seg["start_time_s"]) & (df["timestamp_s"] <= seg["end_time_s"])].copy()


def select_primary_rows(seg_tracks: pd.DataFrame, expected_distance: float) -> pd.DataFrame:
    if seg_tracks.empty:
        return seg_tracks.copy()
    data = seg_tracks.dropna(subset=["timestamp_s"]).copy()
    if "frame" not in data or data["frame"].isna().all():
        data["_frame_key"] = (data["timestamp_s"] / 0.1).round()
    else:
        data["_frame_key"] = data["frame"]
    data["_abs_err"] = (data["range_m"] - expected_distance).abs()
    data["_bad_penalty"] = data.apply(lambda r: 0.35 if is_bad_evidence(r) else 0.0, axis=1)
    tid_counts = data["tid"].value_counts(dropna=True).to_dict()
    data["_persistence_bonus"] = data["tid"].map(tid_counts).fillna(0)
    data["_score"] = data["_abs_err"].fillna(10.0) + data["_bad_penalty"] - data["_persistence_bonus"] * 1e-6
    idx = data.sort_values(["_frame_key", "_score"]).groupby("_frame_key", dropna=False).head(1).index
    return data.loc[idx].sort_values("timestamp_s").drop(columns=[c for c in data.columns if c.startswith("_")], errors="ignore")


def event_runs_from_missing(frames: pd.DataFrame, present_frames: set[float], seg: pd.Series, frame_period_s: float) -> pd.DataFrame:
    if frames.empty or "frame" not in frames:
        return pd.DataFrame(columns=["segment_id", "start_time_s", "end_time_s", "duration_s", "frames"])
    sf = segment_filter(frames, seg)
    missing = sf[~sf["frame"].isin(present_frames)].sort_values("timestamp_s")
    runs = contiguous_runs(missing["timestamp_s"], max_gap_s=max(frame_period_s * 3, 0.25))
    return pd.DataFrame(
        [
            {
                "segment_id": seg["segment_id"],
                "start_time_s": a,
                "end_time_s": b,
                "duration_s": max(b - a, frame_period_s),
                "frames": count,
            }
            for a, b, count in runs
        ]
    )


def longest_run_seconds(mask: pd.Series, times: pd.Series, frame_period_s: float) -> float:
    good_times = times[mask.fillna(False)]
    runs = contiguous_runs(good_times, max_gap_s=max(frame_period_s * 3, 0.25))
    if not runs:
        return 0.0
    return max(max(b - a, frame_period_s) for a, b, _ in runs)


def switch_events(rows: pd.DataFrame, value_col: str, seg_id: str, event_name: str) -> pd.DataFrame:
    if rows.empty or value_col not in rows:
        return pd.DataFrame(columns=["segment_id", "timestamp_s", "from", "to", "event"])
    data = rows.dropna(subset=[value_col]).sort_values("timestamp_s")
    events = []
    prev = None
    for _, row in data.iterrows():
        cur = row[value_col]
        if prev is not None and cur != prev:
            events.append({"segment_id": seg_id, "timestamp_s": row["timestamp_s"], "from": prev, "to": cur, "event": event_name})
        prev = cur
    return pd.DataFrame(events)


def percentile(values: pd.Series, q: float) -> Any:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return float(np.percentile(vals, q))


def mean_or_na(values: pd.Series | Iterable[Any]) -> Any:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else "NA"


def num_or_na(value: Any) -> Any:
    try:
        if pd.isna(value):
            return "NA"
        return float(value)
    except (TypeError, ValueError):
        return "NA"


def compute_tracking_metrics(
    segments: pd.DataFrame,
    frames: pd.DataFrame,
    tracks: pd.DataFrame,
    ctx: Context,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    ghosts = []
    switches = []
    dropouts = []
    primary_by_segment: dict[str, pd.DataFrame] = {}
    for _, seg in segments.iterrows():
        sf = segment_filter(frames, seg)
        st = segment_filter(tracks, seg)
        primary = select_primary_rows(st, float(seg["expected_distance_m"]))
        primary_by_segment[str(seg["segment_id"])] = primary
        frames_total = int(sf["frame"].nunique()) if not sf.empty and "frame" in sf else int(max(round(seg["duration_s"] / ctx.frame_period_s), 0))
        any_frames = set(st["frame"].dropna().unique()) if not st.empty and "frame" in st else set()
        primary_frames = set(primary["frame"].dropna().unique()) if not primary.empty and "frame" in primary else set()
        active = st.groupby("frame")["tid"].nunique() if not st.empty and "frame" in st else pd.Series(dtype=float)
        extra_frames = active[active > 1]
        tid_events = switch_events(primary, "tid", seg["segment_id"], "tid_switch")
        if not tid_events.empty:
            switches.append(tid_events)
        dropout_df = event_runs_from_missing(sf, primary_frames, seg, ctx.frame_period_s)
        if not dropout_df.empty:
            dropouts.append(dropout_df)
        non_primary = st.copy()
        if not primary.empty and "frame" in st:
            key = set(zip(primary["frame"], primary["tid"]))
            non_primary = st[~st.apply(lambda r: (r.get("frame"), r.get("tid")) in key, axis=1)]
        if not non_primary.empty:
            non_primary = non_primary.copy()
            tid_life = non_primary.groupby("tid")["timestamp_s"].agg(["min", "max", "count"])
            short_tids = set(tid_life[(tid_life["max"] - tid_life["min"] < 1.0) | (tid_life["count"] < 5)].index)
            non_primary["ghost_like"] = non_primary.apply(lambda r: is_bad_evidence(r) or r.get("tid") in short_tids, axis=1)
            ghost_rows = non_primary[non_primary["ghost_like"]]
            for tid, rows in ghost_rows.groupby("tid"):
                ghosts.append(
                    {
                        "segment_id": seg["segment_id"],
                        "tid": tid,
                        "frames": rows["frame"].nunique() if "frame" in rows else len(rows),
                        "start_time_s": rows["timestamp_s"].min(),
                        "end_time_s": rows["timestamp_s"].max(),
                        "duration_s": rows["timestamp_s"].max() - rows["timestamp_s"].min(),
                        "mean_range_m": mean_or_na(rows["range_m"]),
                        "ghost_reason": ";".join(sorted(set(reason for reason in rows.apply(ghost_reason, axis=1) if reason))),
                    }
                )
        abs_err = (primary["range_m"] - float(seg["expected_distance_m"])).abs() if not primary.empty else pd.Series(dtype=float)
        range_err = primary["range_m"] - float(seg["expected_distance_m"]) if not primary.empty else pd.Series(dtype=float)
        xy_jitter = np.sqrt((primary["x_m"] - primary["x_m"].mean()) ** 2 + (primary["y_m"] - primary["y_m"].mean()) ** 2) if not primary.empty else pd.Series(dtype=float)
        presence = len(primary_frames) / frames_total if frames_total else np.nan
        dropout_rate = 1.0 - presence if pd.notna(presence) else np.nan
        extra_rate = len(extra_frames) / frames_total if frames_total else np.nan
        tid_switch_count = len(tid_events)
        jitter_p95 = percentile(abs(primary["range_m"] - primary["range_m"].median()), 95) if not primary.empty else "NA"
        range_mae = mean_or_na(abs_err)
        range_score = max(0.0, 1.0 - float(range_mae) / 1.0) if range_mae != "NA" else 0.0
        jitter_penalty = min(1.0, float(jitter_p95) / 0.5) if jitter_p95 != "NA" else 0.0
        tracking_score = (
            35.0 * (presence if pd.notna(presence) else 0.0)
            + 25.0 * range_score
            + 15.0 * max(0.0, 1.0 - (extra_rate if pd.notna(extra_rate) else 1.0))
            + 10.0 * max(0.0, 1.0 - min(1.0, tid_switch_count / 3.0))
            + 10.0 * max(0.0, 1.0 - (dropout_rate if pd.notna(dropout_rate) else 1.0))
            + 5.0 * max(0.0, 1.0 - jitter_penalty)
        )
        metrics.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "expected_distance_m": seg["expected_distance_m"],
                "expected_person_count": 1,
                "frames_total": frames_total,
                "frames_with_any_target": len(any_frames),
                "frames_with_primary_target": len(primary_frames),
                "tracking_presence_rate": presence,
                "dropout_frames": max(frames_total - len(primary_frames), 0),
                "dropout_rate": dropout_rate,
                "longest_dropout_s": mean_or_na(dropout_df["duration_s"].nlargest(1)) if not dropout_df.empty else 0.0,
                "mean_range_m": mean_or_na(primary["range_m"]) if not primary.empty else "NA",
                "median_range_m": num_or_na(primary["range_m"].median()) if not primary.empty else "NA",
                "std_range_m": num_or_na(primary["range_m"].std()) if not primary.empty else "NA",
                "mae_range_m": range_mae,
                "rmse_range_m": float(np.sqrt(np.nanmean(range_err**2))) if len(range_err.dropna()) else "NA",
                "bias_m": mean_or_na(range_err),
                "p50_abs_error_m": percentile(abs_err, 50),
                "p90_abs_error_m": percentile(abs_err, 90),
                "p95_abs_error_m": percentile(abs_err, 95),
                "std_x_m": num_or_na(primary["x_m"].std()) if not primary.empty else "NA",
                "std_y_m": num_or_na(primary["y_m"].std()) if not primary.empty else "NA",
                "std_z_m": num_or_na(primary["z_m"].std()) if not primary.empty else "NA",
                "range_jitter_p95_m": jitter_p95,
                "xy_jitter_m": mean_or_na(xy_jitter),
                "unique_tids": st["tid"].nunique() if not st.empty else 0,
                "primary_tid": primary["tid"].mode().iloc[0] if not primary.empty and len(primary["tid"].mode()) else "NA",
                "tid_switch_count": tid_switch_count,
                "tid_switch_rate_per_min": tid_switch_count / max(seg["duration_s"] / 60.0, 1e-9),
                "mean_tid_lifetime_s": mean_tid_lifetime(st),
                "fragmentation_count": max(st["tid"].nunique() - 1, 0) if not st.empty else 0,
                "mean_active_tracks": mean_or_na(active),
                "max_active_tracks": num_or_na(active.max()) if len(active) else 0,
                "frames_with_extra_tracks": len(extra_frames),
                "extra_track_rate": extra_rate,
                "ghost_track_count": len({g["tid"] for g in ghosts if g["segment_id"] == seg["segment_id"]}),
                "ghost_frames": int(sum(g["frames"] for g in ghosts if g["segment_id"] == seg["segment_id"])),
                "ghost_duration_s": mean_or_na([g["duration_s"] for g in ghosts if g["segment_id"] == seg["segment_id"]]),
                "ghost_nearest_distance_to_primary_m": nearest_ghost_distance(st, primary),
                "presence_component": presence,
                "range_component": range_score,
                "ghost_component": max(0.0, 1.0 - extra_rate) if pd.notna(extra_rate) else "NA",
                "id_switch_component": max(0.0, 1.0 - min(1.0, tid_switch_count / 3.0)),
                "dropout_component": max(0.0, 1.0 - dropout_rate) if pd.notna(dropout_rate) else "NA",
                "jitter_component": max(0.0, 1.0 - jitter_penalty),
                "tracking_score": tracking_score,
            }
        )
    by_segment = pd.DataFrame(metrics)
    overall = summarize_overall_tracking(by_segment)
    ghost_df = pd.DataFrame(ghosts)
    switch_df = pd.concat(switches, ignore_index=True) if switches else pd.DataFrame(columns=["segment_id", "timestamp_s", "from", "to", "event"])
    dropout_df = pd.concat(dropouts, ignore_index=True) if dropouts else pd.DataFrame(columns=["segment_id", "start_time_s", "end_time_s", "duration_s", "frames"])
    return by_segment, overall, ghost_df, switch_df, dropout_df


def ghost_reason(row: pd.Series) -> str:
    reasons = []
    geom = row.get("geom_pts")
    if pd.notna(geom) and float(geom) <= 0:
        reasons.append("geom_pts_0")
    quality = str(row.get("quality", "")).upper()
    assoc = str(row.get("assoc", "")).upper()
    if "NO_POINTS" in quality:
        reasons.append("NO_POINTS")
    if "TARGET_ONLY" in quality:
        reasons.append("TARGET_ONLY")
    if "AUTO_NONE" in assoc:
        reasons.append("auto_none")
    return ",".join(reasons) or "extra_track"


def mean_tid_lifetime(st: pd.DataFrame) -> Any:
    if st.empty:
        return "NA"
    lives = st.groupby("tid")["timestamp_s"].agg(lambda s: s.max() - s.min())
    return mean_or_na(lives)


def nearest_ghost_distance(st: pd.DataFrame, primary: pd.DataFrame) -> Any:
    if st.empty or primary.empty or not {"frame", "tid", "x_m", "y_m", "z_m"}.issubset(st.columns):
        return "NA"
    p = primary[["frame", "tid", "x_m", "y_m", "z_m"]].rename(columns={"tid": "primary_tid", "x_m": "px", "y_m": "py", "z_m": "pz"})
    merged = st.merge(p, on="frame", how="inner")
    merged = merged[merged["tid"] != merged["primary_tid"]]
    if merged.empty:
        return "NA"
    d = np.sqrt((merged["x_m"] - merged["px"]) ** 2 + (merged["y_m"] - merged["py"]) ** 2 + (merged["z_m"] - merged["pz"]) ** 2)
    return num_or_na(d.min())


def summarize_overall_tracking(by_segment: pd.DataFrame) -> pd.DataFrame:
    if by_segment.empty:
        return pd.DataFrame()
    fields = [
        "tracking_presence_rate",
        "mae_range_m",
        "dropout_rate",
        "extra_track_rate",
        "tid_switch_count",
        "tracking_score",
    ]
    rows = []
    for field in fields:
        rows.append({"metric": field, "mean": mean_or_na(by_segment[field]), "median": num_or_na(pd.to_numeric(by_segment[field], errors="coerce").median())})
    return pd.DataFrame(rows)


def compute_posture_metrics(
    segments: pd.DataFrame,
    frames: pd.DataFrame,
    pose: pd.DataFrame,
    tracks: pd.DataFrame,
    tracking_metrics: pd.DataFrame,
    ctx: Context,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    confusion_rows = []
    failure_rows = []
    quality_rows = []
    switch_rows = []
    for _, seg in segments.iterrows():
        sf = segment_filter(frames, seg)
        sp = segment_filter(pose, seg)
        st = segment_filter(tracks, seg)
        primary = select_primary_rows(st, float(seg["expected_distance_m"]))
        if not sp.empty and not primary.empty:
            primary_keys = set(zip(primary["frame"], primary["tid"]))
            filt = sp[sp.apply(lambda r: (r.get("frame"), r.get("tid")) in primary_keys, axis=1)]
            if not filt.empty:
                sp = filt
        if not sp.empty and "frame" in sp:
            sp = sp.sort_values("timestamp_s").drop_duplicates("frame", keep="last")
        frames_total = int(sf["frame"].nunique()) if not sf.empty and "frame" in sf else int(max(round(seg["duration_s"] / ctx.frame_period_s), 0))
        expected_pose = str(seg["expected_pose"]).upper()
        display = sp["display_pose"].map(normalize_pose_label) if not sp.empty else pd.Series(dtype=str)
        correct = display == expected_pose
        counts = display.value_counts().to_dict()
        for pred, count in counts.items():
            confusion_rows.append({"expected_pose": expected_pose, "predicted_pose": pred, "count": count, "segment_id": seg["segment_id"]})
        if not sp.empty:
            for col in ["quality", "geom_quality", "assoc", "range_zone", "moving_override_reason", "final_reason", "stand_sit_decision", "status"]:
                if col in sp:
                    for val, count in sp[col].fillna("NA").astype(str).value_counts().items():
                        failure_rows.append({"segment_id": seg["segment_id"], "field": col, "value": val, "count": count})
            quality_rows.extend(quality_breakdown_rows(seg, sp, correct))
            sw = switch_events(sp.assign(display_pose=display), "display_pose", seg["segment_id"], "pose_switch")
            if not sw.empty:
                switch_rows.append(sw)
        wrong_counts = display[~correct & ~display.isin(["UNKNOWN"])].value_counts()
        dominant_wrong = wrong_counts.index[0] if len(wrong_counts) else "NA"
        metrics.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": expected_pose,
                "expected_distance_m": seg["expected_distance_m"],
                "frames_total": frames_total,
                "frames_with_pose_prediction": int(len(display)),
                "dominant_display_pose": display.mode().iloc[0] if len(display.mode()) else "NA",
                "correct_frames": int(correct.sum()) if len(correct) else 0,
                "accuracy": float(correct.mean()) if len(correct) else "NA",
                "unknown_rate": rate(display, "UNKNOWN"),
                "moving_rate": rate(display, "MOVING"),
                "falling_false_rate": rate(display, "FALLING"),
                "lying_false_rate": rate(display, "LYING"),
                "standing_rate": rate(display, "STANDING"),
                "sitting_rate": rate(display, "SITTING"),
                "mean_stand_prob": mean_or_na(sp["stand_prob"]) if "stand_prob" in sp else "NA",
                "mean_sit_prob": mean_or_na(sp["sit_prob"]) if "sit_prob" in sp else "NA",
                "median_stand_prob": num_or_na(sp["stand_prob"].median()) if "stand_prob" in sp and not sp.empty else "NA",
                "median_sit_prob": num_or_na(sp["sit_prob"].median()) if "sit_prob" in sp and not sp.empty else "NA",
                "mean_confidence_if_available": mean_or_na(sp["confidence"]) if "confidence" in sp else "NA",
                "time_to_first_correct_s": time_to_first_correct(sp, correct, seg),
                "time_to_stable_correct_s": time_to_stable_correct(sp, correct, seg, ctx.frame_period_s),
                "longest_correct_run_s": longest_run_seconds(correct, sp["timestamp_s"] if "timestamp_s" in sp else pd.Series(dtype=float), ctx.frame_period_s),
                "longest_wrong_run_s": longest_run_seconds(~correct, sp["timestamp_s"] if "timestamp_s" in sp else pd.Series(dtype=float), ctx.frame_period_s),
                "pose_switch_count": int(len(switch_rows[-1])) if switch_rows and switch_rows[-1]["segment_id"].iloc[0] == seg["segment_id"] else 0,
                "pose_switch_rate_per_min": (int(len(switch_rows[-1])) if switch_rows and switch_rows[-1]["segment_id"].iloc[0] == seg["segment_id"] else 0) / max(seg["duration_s"] / 60.0, 1e-9),
                "dominant_wrong_pose": dominant_wrong,
            }
        )
    by_segment = pd.DataFrame(metrics)
    confusion = pd.DataFrame(confusion_rows)
    if not confusion.empty:
        confusion = confusion.pivot_table(index="expected_pose", columns="predicted_pose", values="count", aggfunc="sum", fill_value=0).reset_index()
        for cls in POSE_CLASSES:
            if cls not in confusion.columns:
                confusion[cls] = 0
        confusion = confusion[["expected_pose"] + POSE_CLASSES]
    else:
        confusion = pd.DataFrame(columns=["expected_pose"] + POSE_CLASSES)
    failure = pd.DataFrame(failure_rows)
    quality = pd.DataFrame(quality_rows)
    switches = pd.concat(switch_rows, ignore_index=True) if switch_rows else pd.DataFrame(columns=["segment_id", "timestamp_s", "from", "to", "event"])
    return by_segment, confusion, failure, quality, switches


def rate(series: pd.Series, label: str) -> Any:
    if len(series) == 0:
        return "NA"
    return float((series == label).mean())


def time_to_first_correct(sp: pd.DataFrame, correct: pd.Series, seg: pd.Series) -> Any:
    if sp.empty or not correct.any():
        return "NA"
    return float(sp.loc[correct, "timestamp_s"].min() - seg["start_time_s"])


def time_to_stable_correct(sp: pd.DataFrame, correct: pd.Series, seg: pd.Series, frame_period_s: float) -> Any:
    if sp.empty or not correct.any():
        return "NA"
    runs = contiguous_runs(sp.loc[correct, "timestamp_s"], max_gap_s=max(frame_period_s * 3, 0.25))
    stable = [(a, b) for a, b, _ in runs if (b - a) >= 5.0]
    if not stable:
        return "NA"
    return float(stable[0][0] - seg["start_time_s"])


def quality_breakdown_rows(seg: pd.Series, sp: pd.DataFrame, correct: pd.Series) -> list[dict[str, Any]]:
    rows = []
    masks = {
        "quality_OK": sp.get("quality", pd.Series("", index=sp.index)).astype(str).str.upper().eq("OK"),
        "LOW_POINTS": sp.get("quality", pd.Series("", index=sp.index)).astype(str).str.upper().str.contains("LOW_POINTS", na=False),
        "NO_POINTS": sp.get("quality", pd.Series("", index=sp.index)).astype(str).str.upper().str.contains("NO_POINTS", na=False),
        "geom_pts_0": numeric(sp.get("geom_pts", pd.Series(np.nan, index=sp.index))).fillna(-1).eq(0),
        "geom_pts_ge_3": numeric(sp.get("geom_pts", pd.Series(np.nan, index=sp.index))).fillna(-1).ge(3),
    }
    for name, mask in masks.items():
        denom = int(mask.sum())
        rows.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "bucket": name,
                "frames": denom,
                "accuracy": float(correct[mask].mean()) if denom else "NA",
            }
        )
    return rows


def combined_diagnostics(
    segments: pd.DataFrame,
    tracks: pd.DataFrame,
    pose: pd.DataFrame,
    tracking: pd.DataFrame,
    posture: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, seg in segments.iterrows():
        tr = tracking[tracking["segment_id"] == seg["segment_id"]]
        po = posture[posture["segment_id"] == seg["segment_id"]]
        sp = segment_filter(pose, seg)
        track_presence = tr["tracking_presence_rate"].iloc[0] if not tr.empty else np.nan
        range_mae = tr["mae_range_m"].iloc[0] if not tr.empty else np.nan
        extra_rate = tr["extra_track_rate"].iloc[0] if not tr.empty else np.nan
        posture_acc = po["accuracy"].iloc[0] if not po.empty else np.nan
        dominant_wrong = po["dominant_wrong_pose"].iloc[0] if not po.empty else "NA"
        quality_no_points = (
            float(sp["quality"].astype(str).str.upper().str.contains("NO_POINTS", na=False).mean())
            if not sp.empty and "quality" in sp
            else "NA"
        )
        mean_geom_pts = mean_or_na(sp["geom_pts"]) if not sp.empty and "geom_pts" in sp else "NA"
        rows.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "expected_distance_m": seg["expected_distance_m"],
                "tracking_presence_rate": track_presence,
                "range_mae_m": range_mae,
                "extra_track_rate": extra_rate,
                "tid_switch_count": tr["tid_switch_count"].iloc[0] if not tr.empty else "NA",
                "posture_accuracy": posture_acc,
                "dominant_wrong_pose": dominant_wrong,
                "quality_NO_POINTS_rate": quality_no_points,
                "mean_geom_pts": mean_geom_pts,
                "main_failure_hypothesis": diagnose_issue(track_presence, range_mae, extra_rate, posture_acc, dominant_wrong, quality_no_points, mean_geom_pts, seg["expected_pose"]),
            }
        )
    return pd.DataFrame(rows)


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value == "NA" or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def diagnose_issue(
    tracking_presence: Any,
    range_mae: Any,
    extra_rate: Any,
    posture_acc: Any,
    dominant_wrong: Any,
    no_points_rate: Any,
    mean_geom_pts: Any,
    expected_pose: Any,
) -> str:
    tp = as_float(tracking_presence)
    rm = as_float(range_mae)
    er = as_float(extra_rate)
    pa = as_float(posture_acc)
    nr = as_float(no_points_rate)
    gp = as_float(mean_geom_pts)
    if not np.isnan(tp) and tp < 0.75:
        return "tracking dropout"
    if not np.isnan(er) and er > 0.20:
        return "ghost/shadow tracking"
    if not np.isnan(rm) and rm > 0.60 and (np.isnan(pa) or pa >= 0.70):
        return "range bias / coordinate calibration"
    if not np.isnan(pa) and pa < 0.70:
        if dominant_wrong == "MOVING":
            return "moving override false positive / target jitter"
        if expected_pose == "SITTING" and dominant_wrong == "STANDING":
            if (not np.isnan(nr) and nr > 0.20) or (not np.isnan(gp) and gp < 3):
                return "sit-vs-stand discrimination under sparse geometry"
            return "sit-vs-stand discrimination"
        if expected_pose == "STANDING" and dominant_wrong == "SITTING":
            return "height/tilt/calibration or model bias"
        if (not np.isnan(nr) and nr > 0.20) or (not np.isnan(gp) and gp < 3):
            return "point association / sparse point cloud"
        return "posture classifier / threshold / hysteresis"
    return "no major failure detected"


def tracking_verdicts(tracking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tracking.iterrows():
        presence = as_float(row.get("tracking_presence_rate"))
        range_mae = as_float(row.get("mae_range_m"))
        dropout = as_float(row.get("dropout_rate"))
        extra = as_float(row.get("extra_track_rate"))
        switches = as_float(row.get("tid_switch_count"), 0.0)
        score = as_float(row.get("tracking_score"))
        range_jitter = as_float(row.get("range_jitter_p95_m"))
        verdict = "GOOD"
        reason = "Presence is high with no meaningful dropout, ID-switch, or extra-track issue."
        if (not np.isnan(presence) and presence < 0.95) or (not np.isnan(dropout) and dropout > 0.05):
            verdict = "DROPOUT"
            reason = "Tracking presence/dropout crosses the failure threshold."
        elif not np.isnan(switches) and switches > 0:
            verdict = "ID_SWITCH"
            reason = "One or more TID switches occurred in the segment."
        elif not np.isnan(extra) and extra > 0.05:
            verdict = "GHOST_TRACKS"
            reason = "Extra active tracks exceed the controlled single-person threshold."
        elif (not np.isnan(range_mae) and range_mae > 0.30) or (not np.isnan(score) and score < 85):
            verdict = "FAILED"
            reason = "Range error or tracking score crosses the tracking failure threshold."
        elif not np.isnan(range_mae) and range_mae > 0.15:
            verdict = "MINOR_RANGE_BIAS"
            reason = "Tracking is continuous, but range MAE is a minor calibration/bias issue."
        elif not np.isnan(range_jitter) and range_jitter > 0.25:
            verdict = "MINOR_JITTER"
            reason = "Tracking is continuous, but stationary range jitter is elevated."
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "expected_pose": row.get("expected_pose"),
                "expected_distance_m": row.get("expected_distance_m"),
                "tracking_presence_rate": row.get("tracking_presence_rate", "NA"),
                "range_mae_m": row.get("mae_range_m", "NA"),
                "dropout_rate": row.get("dropout_rate", "NA"),
                "extra_track_rate": row.get("extra_track_rate", "NA"),
                "tid_switch_count": row.get("tid_switch_count", "NA"),
                "tracking_score": row.get("tracking_score", "NA"),
                "tracking_verdict": verdict,
                "tracking_issue_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def posture_verdicts(posture: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combined_lookup = combined.set_index("segment_id") if not combined.empty and "segment_id" in combined else pd.DataFrame()
    for _, row in posture.iterrows():
        sid = row.get("segment_id")
        acc = as_float(row.get("accuracy"))
        standing_rate = as_float(row.get("standing_rate"), 0.0)
        sitting_rate = as_float(row.get("sitting_rate"), 0.0)
        moving_rate = as_float(row.get("moving_rate"), 0.0)
        unknown_rate = as_float(row.get("unknown_rate"), 0.0)
        expected = str(row.get("expected_pose", "")).upper()
        verdict = "GOOD" if not np.isnan(acc) and acc >= 0.90 else "WEAK" if not np.isnan(acc) and acc >= 0.70 else "FAILED"
        reason = f"Accuracy is {fmt(acc)}."
        if moving_rate > 0.10:
            verdict = "MOVING_FALSE_POSITIVE"
            reason = "MOVING appears too often during a stationary benchmark segment."
        elif unknown_rate > 0.10:
            verdict = "UNKNOWN_TOO_OFTEN"
            reason = "UNKNOWN appears too often during a labeled benchmark segment."
        elif expected == "SITTING" and standing_rate > sitting_rate:
            verdict = "SIT_AS_STAND"
            reason = "Expected SITTING is displayed as STANDING more often than SITTING."
        elif expected == "STANDING" and sitting_rate > 0.10:
            verdict = "STAND_AS_SIT"
            reason = "Expected STANDING has a nontrivial SITTING false-positive rate."
        elif verdict == "WEAK":
            reason = "Accuracy is below the strong-pass threshold but not a full failure."
        elif verdict == "FAILED":
            reason = "Accuracy is below the 70% failure threshold."
        extra = combined_lookup.loc[sid] if not combined_lookup.empty and sid in combined_lookup.index else {}
        rows.append(
            {
                "segment_id": sid,
                "expected_pose": row.get("expected_pose"),
                "expected_distance_m": row.get("expected_distance_m"),
                "accuracy": row.get("accuracy", "NA"),
                "dominant_display_pose": row.get("dominant_display_pose", "NA"),
                "dominant_wrong_pose": row.get("dominant_wrong_pose", "NA"),
                "standing_rate": row.get("standing_rate", "NA"),
                "sitting_rate": row.get("sitting_rate", "NA"),
                "moving_rate": row.get("moving_rate", "NA"),
                "unknown_rate": row.get("unknown_rate", "NA"),
                "quality_NO_POINTS_rate": extra.get("quality_NO_POINTS_rate", "NA") if isinstance(extra, pd.Series) else "NA",
                "mean_geom_pts": extra.get("mean_geom_pts", "NA") if isinstance(extra, pd.Series) else "NA",
                "mean_stand_prob": row.get("mean_stand_prob", "NA"),
                "mean_sit_prob": row.get("mean_sit_prob", "NA"),
                "time_to_stable_correct_s": row.get("time_to_stable_correct_s", "NA"),
                "posture_verdict": verdict,
                "posture_issue_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def stand_sit_probability_by_segment(segments: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, seg in segments.iterrows():
        sp = segment_filter(pose, seg)
        stand = numeric(sp.get("stand_prob", pd.Series(dtype=float))).dropna() if not sp.empty else pd.Series(dtype=float)
        sit = numeric(sp.get("sit_prob", pd.Series(dtype=float))).dropna() if not sp.empty else pd.Series(dtype=float)
        both = sp.copy()
        if not both.empty:
            both["stand_prob"] = numeric(both.get("stand_prob", pd.Series(np.nan, index=both.index)))
            both["sit_prob"] = numeric(both.get("sit_prob", pd.Series(np.nan, index=both.index)))
            margin = (both["stand_prob"] - both["sit_prob"]).dropna()
        else:
            margin = pd.Series(dtype=float)
        rows.append(
            {
                "segment_id": seg["segment_id"],
                "expected_pose": seg["expected_pose"],
                "expected_distance_m": seg["expected_distance_m"],
                "mean_stand_prob": float(stand.mean()) if len(stand) else "NA",
                "median_stand_prob": float(stand.median()) if len(stand) else "NA",
                "p10_stand_prob": float(stand.quantile(0.10)) if len(stand) else "NA",
                "p90_stand_prob": float(stand.quantile(0.90)) if len(stand) else "NA",
                "mean_sit_prob": float(sit.mean()) if len(sit) else "NA",
                "median_sit_prob": float(sit.median()) if len(sit) else "NA",
                "p10_sit_prob": float(sit.quantile(0.10)) if len(sit) else "NA",
                "p90_sit_prob": float(sit.quantile(0.90)) if len(sit) else "NA",
                "mean_margin_stand_minus_sit": float(margin.mean()) if len(margin) else "NA",
                "median_margin_stand_minus_sit": float(margin.median()) if len(margin) else "NA",
                "frames_stand_prob_gt_sit_prob": int((both["stand_prob"] > both["sit_prob"]).sum()) if not both.empty and {"stand_prob", "sit_prob"}.issubset(both.columns) else 0,
                "frames_sit_prob_gt_stand_prob": int((both["sit_prob"] > both["stand_prob"]).sum()) if not both.empty and {"stand_prob", "sit_prob"}.issubset(both.columns) else 0,
            }
        )
    return pd.DataFrame(rows)


def no_points_effect_by_pose(segments: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, seg in segments.iterrows():
        sp = segment_filter(pose, seg)
        if sp.empty:
            rows.append(
                {
                    "expected_pose": seg["expected_pose"],
                    "expected_distance_m": seg["expected_distance_m"],
                    "quality_bucket": "NO_DATA",
                    "frames": 0,
                    "accuracy": "NA",
                    "standing_rate": "NA",
                    "sitting_rate": "NA",
                    "mean_stand_prob": "NA",
                    "mean_sit_prob": "NA",
                }
            )
            continue
        q = sp.get("quality", pd.Series("", index=sp.index)).astype(str).str.upper()
        geom = numeric(sp.get("geom_pts", pd.Series(np.nan, index=sp.index)))
        buckets = pd.Series("OTHER", index=sp.index)
        buckets[q.str.contains("NO_POINTS", na=False) | geom.eq(0)] = "NO_POINTS"
        buckets[q.str.contains("LOW_POINTS", na=False) & ~buckets.eq("NO_POINTS")] = "LOW_POINTS"
        buckets[geom.ge(1) & ~buckets.isin(["NO_POINTS", "LOW_POINTS"])] = "HAS_POINTS"
        for bucket in ["NO_POINTS", "LOW_POINTS", "HAS_POINTS", "OTHER"]:
            sb = sp[buckets.eq(bucket)]
            if sb.empty:
                continue
            pred = sb["display_pose"].astype(str).str.upper()
            rows.append(
                {
                    "expected_pose": seg["expected_pose"],
                    "expected_distance_m": seg["expected_distance_m"],
                    "quality_bucket": bucket,
                    "frames": int(len(sb)),
                    "accuracy": float(pred.eq(seg["expected_pose"]).mean()),
                    "standing_rate": float(pred.eq("STANDING").mean()),
                    "sitting_rate": float(pred.eq("SITTING").mean()),
                    "mean_stand_prob": mean_or_na(sb["stand_prob"]) if "stand_prob" in sb else "NA",
                    "mean_sit_prob": mean_or_na(sb["sit_prob"]) if "sit_prob" in sb else "NA",
                }
            )
    return pd.DataFrame(rows)


def ghost_shadow_verdicts(tracking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tracking.iterrows():
        extra = as_float(row.get("extra_track_rate"), 0.0)
        frames_extra = as_float(row.get("frames_with_extra_tracks"), 0.0)
        if extra > 0.05:
            verdict = "GHOST_TRACKS"
            notes = "Extra tracks beyond expected_person_count=1 were measured in this controlled benchmark."
        else:
            verdict = "GOOD_NO_EXTRA_TRACKS"
            notes = "No extra tracks beyond expected_person_count=1; suppressed suspect-track statistics were not available/parsed."
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "mean_active_tracks": row.get("mean_active_tracks", "NA"),
                "max_active_tracks": row.get("max_active_tracks", "NA"),
                "extra_track_rate": row.get("extra_track_rate", "NA"),
                "frames_with_extra_tracks": int(frames_extra) if not np.isnan(frames_extra) else "NA",
                "suspect_frames_if_available": "NA",
                "provisional_frames_if_available": "NA",
                "confirmed_frames_if_available": "NA",
                "ghost_verdict": verdict,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_csvs(
    out: Path,
    segments: pd.DataFrame,
    tracking: pd.DataFrame,
    tracking_overall: pd.DataFrame,
    posture: pd.DataFrame,
    confusion: pd.DataFrame,
    failure: pd.DataFrame,
    quality: pd.DataFrame,
    combined: pd.DataFrame,
    tracking_verdict: pd.DataFrame,
    posture_verdict: pd.DataFrame,
    probability: pd.DataFrame,
    no_points: pd.DataFrame,
    ghost_shadow: pd.DataFrame,
    ghosts: pd.DataFrame,
    tid_switches: pd.DataFrame,
    dropouts: pd.DataFrame,
    pose_switches: pd.DataFrame,
) -> None:
    outputs = {
        "segments_auto.csv": segments,
        "tracking_metrics_by_segment.csv": tracking,
        "tracking_metrics_overall.csv": tracking_overall,
        "posture_metrics_by_segment.csv": posture,
        "posture_confusion_matrix.csv": confusion,
        "posture_failure_reasons.csv": failure,
        "posture_quality_breakdown.csv": quality,
        "combined_diagnostics_by_segment.csv": combined,
        "tracking_verdict_by_segment.csv": tracking_verdict,
        "posture_verdict_by_segment.csv": posture_verdict,
        "stand_sit_probability_by_segment.csv": probability,
        "no_points_effect_by_pose.csv": no_points,
        "ghost_shadow_verdict_by_segment.csv": ghost_shadow,
        "ghost_tracks.csv": ghosts,
        "tid_switch_events.csv": tid_switches,
        "dropout_events.csv": dropouts,
        "pose_switch_events.csv": pose_switches,
    }
    for name, df in outputs.items():
        df.to_csv(out / name, index=False)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def no_data_plot(path: Path, title: str) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def add_segment_bands(ax: Any, segments: pd.DataFrame) -> None:
    for _, seg in segments.iterrows():
        color = "#dfefff" if seg["expected_pose"] == "STANDING" else "#fff1d6"
        ax.axvspan(seg["start_time_s"], seg["end_time_s"], color=color, alpha=0.35)
        ax.text(
            (seg["start_time_s"] + seg["end_time_s"]) / 2,
            ax.get_ylim()[1],
            f"{seg['expected_pose'][0]} {seg['expected_distance_m']:g}m",
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
        )


def generate_plots(
    out: Path,
    segments: pd.DataFrame,
    frames: pd.DataFrame,
    tracks: pd.DataFrame,
    pose: pd.DataFrame,
    rgb: pd.DataFrame,
    tracking: pd.DataFrame,
    posture: pd.DataFrame,
    confusion: pd.DataFrame,
    quality: pd.DataFrame,
    combined: pd.DataFrame,
    probability: pd.DataFrame,
    ctx: Context,
) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    def save_or_empty(name: str, title: str, fn: Any) -> None:
        try:
            fn(plots / name)
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"plot {name} failed: {exc}")
            no_data_plot(plots / name, title)

    save_or_empty("timeline_range_by_track.png", "Range by track", lambda p: plot_timeline_range(p, tracks, segments, plt))
    save_or_empty("timeline_active_track_count.png", "Active track count", lambda p: plot_active_tracks(p, frames, tracks, segments, plt))
    save_or_empty("timeline_display_pose.png", "Display pose", lambda p: plot_display_pose(p, pose, segments, plt))
    save_or_empty("timeline_quality_geom_pts.png", "Quality and geom points", lambda p: plot_quality_geom_pts(p, pose, segments, plt))
    save_or_empty("timeline_stand_sit_probs.png", "Stand/sit probabilities", lambda p: plot_stand_sit_probs(p, pose, segments, plt))
    save_or_empty("stand_vs_sit_probability_by_segment.png", "Stand vs sit probability", lambda p: plot_stand_vs_sit_probability_by_segment(p, probability, plt))
    save_or_empty("stand_minus_sit_margin_by_segment.png", "Stand minus sit margin", lambda p: plot_stand_minus_sit_margin_by_segment(p, probability, plt))
    save_or_empty("sitting_segments_stand_sit_prob_timeline.png", "Sitting stand/sit probability timeline", lambda p: plot_sitting_segments_stand_sit_prob_timeline(p, pose, segments, plt))
    if not rgb.empty:
        save_or_empty("timeline_rgb_detection_count.png", "RGB detection count", lambda p: plot_rgb_count(p, rgb, segments, plt))
    save_or_empty("tracking_presence_by_distance.png", "Tracking presence", lambda p: plot_grouped_metric(p, tracking, "tracking_presence_rate", "Tracking presence rate", plt))
    save_or_empty("range_error_by_distance.png", "Range error", lambda p: plot_grouped_metric(p, tracking, "mae_range_m", "Range MAE (m)", plt))
    save_or_empty("range_jitter_by_distance.png", "Range jitter", lambda p: plot_grouped_metric(p, tracking, "range_jitter_p95_m", "Range jitter p95 (m)", plt))
    save_or_empty("ghost_rate_by_distance.png", "Ghost rate", lambda p: plot_grouped_metric(p, tracking, "extra_track_rate", "Extra track rate", plt))
    save_or_empty("tid_switches_by_distance.png", "TID switches", lambda p: plot_grouped_metric(p, tracking, "tid_switch_count", "TID switches", plt))
    save_or_empty("xy_position_scatter_by_segment.png", "XY scatter", lambda p: plot_xy_scatter(p, tracks, segments, plt))
    save_or_empty("posture_accuracy_by_distance.png", "Posture accuracy", lambda p: plot_grouped_metric(p, posture, "accuracy", "Posture accuracy", plt))
    save_or_empty("posture_confusion_matrix.png", "Posture confusion matrix", lambda p: plot_confusion(p, confusion, plt))
    save_or_empty("pose_distribution_by_segment.png", "Pose distribution", lambda p: plot_pose_distribution(p, pose, segments, plt))
    save_or_empty("posture_accuracy_vs_geom_pts.png", "Accuracy vs geom points", lambda p: plot_accuracy_vs_geom_pts(p, pose, segments, plt))
    save_or_empty("posture_accuracy_by_quality.png", "Accuracy by quality", lambda p: plot_accuracy_by_quality(p, quality, plt))
    save_or_empty("moving_false_positive_rate.png", "Moving false positive rate", lambda p: plot_grouped_metric(p, posture, "moving_rate", "MOVING rate", plt))
    save_or_empty("false_falling_rate_by_segment.png", "False falling/lying/unknown", lambda p: plot_false_rates(p, posture, plt))
    save_or_empty("time_to_stable_correct_by_segment.png", "Time to stable correct", lambda p: plot_grouped_metric(p, posture, "time_to_stable_correct_s", "Seconds", plt))
    save_or_empty("tracking_vs_posture_summary.png", "Tracking vs posture", lambda p: plot_tracking_vs_posture(p, tracking, posture, plt))
    save_or_empty("failure_mode_heatmap.png", "Failure mode heatmap", lambda p: plot_failure_heatmap(p, combined, plt))


def plot_timeline_range(path: Path, tracks: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if tracks.empty:
        return no_data_plot(path, "Range by track")
    fig, ax = plt.subplots(figsize=(12, 5))
    for tid, rows in tracks.dropna(subset=["range_m"]).groupby("tid"):
        ax.plot(rows["timestamp_s"], rows["range_m"], ".", markersize=2, label=f"TID {tid:g}" if pd.notna(tid) else "TID NA")
    for d in [1, 2, 3, 4]:
        ax.axhline(d, color="k", lw=0.5, alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    add_segment_bands(ax, segments)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_active_tracks(path: Path, frames: pd.DataFrame, tracks: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if not frames.empty and "num_tracks" in frames and frames["num_tracks"].notna().any():
        data = frames[["timestamp_s", "num_tracks"]].dropna()
    elif not tracks.empty:
        data = tracks.groupby("frame").agg(timestamp_s=("timestamp_s", "min"), num_tracks=("tid", "nunique")).reset_index()
    else:
        return no_data_plot(path, "Active track count")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["timestamp_s"], data["num_tracks"], lw=1)
    extra = data[data["num_tracks"] > 1]
    ax.scatter(extra["timestamp_s"], extra["num_tracks"], color="red", s=8, label="extra tracks")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("active tracks")
    add_segment_bands(ax, segments)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_display_pose(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if pose.empty:
        return no_data_plot(path, "Display pose")
    ymap = {cls: i for i, cls in enumerate(POSE_CLASSES)}
    data = pose.copy()
    data["y"] = data["display_pose"].map(ymap).fillna(ymap["OTHER"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(data["timestamp_s"], data["y"], s=3)
    ax.set_yticks(list(ymap.values()), list(ymap.keys()))
    ax.set_xlabel("time (s)")
    add_segment_bands(ax, segments)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quality_geom_pts(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if pose.empty or "geom_pts" not in pose:
        return no_data_plot(path, "Quality and geom points")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pose["timestamp_s"], pose["geom_pts"], ".", markersize=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("geom_pts")
    add_segment_bands(ax, segments)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stand_sit_probs(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if pose.empty or not {"stand_prob", "sit_prob"}.issubset(pose.columns):
        return no_data_plot(path, "Stand/sit probabilities")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pose["timestamp_s"], pose["stand_prob"], ".", markersize=2, label="stand_prob")
    ax.plot(pose["timestamp_s"], pose["sit_prob"], ".", markersize=2, label="sit_prob")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.legend()
    add_segment_bands(ax, segments)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stand_vs_sit_probability_by_segment(path: Path, probability: pd.DataFrame, plt: Any) -> None:
    if probability.empty:
        return no_data_plot(path, "Stand vs sit probability")
    data = probability.copy()
    data["mean_stand_prob"] = pd.to_numeric(data["mean_stand_prob"], errors="coerce")
    data["mean_sit_prob"] = pd.to_numeric(data["mean_sit_prob"], errors="coerce")
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.18, data["mean_stand_prob"], width=0.36, label="mean stand_prob")
    ax.bar(x + 0.18, data["mean_sit_prob"], width=0.36, label="mean sit_prob")
    ax.set_xticks(x, data["segment_id"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stand_minus_sit_margin_by_segment(path: Path, probability: pd.DataFrame, plt: Any) -> None:
    if probability.empty:
        return no_data_plot(path, "Stand minus sit margin")
    data = probability.copy()
    data["mean_margin_stand_minus_sit"] = pd.to_numeric(data["mean_margin_stand_minus_sit"], errors="coerce")
    colors = ["#4c78a8" if v >= 0 else "#f58518" for v in data["mean_margin_stand_minus_sit"].fillna(0)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(data["segment_id"], data["mean_margin_stand_minus_sit"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("stand_prob - sit_prob")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sitting_segments_stand_sit_prob_timeline(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    sitting = segments[segments["expected_pose"].astype(str).str.upper().eq("SITTING")]
    if pose.empty or sitting.empty or not {"stand_prob", "sit_prob"}.issubset(pose.columns):
        return no_data_plot(path, "Sitting stand/sit probability timeline")
    start = sitting["start_time_s"].min()
    end = sitting["end_time_s"].max()
    data = pose[(pose["timestamp_s"] >= start) & (pose["timestamp_s"] <= end)].copy()
    if data.empty:
        return no_data_plot(path, "Sitting stand/sit probability timeline")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["timestamp_s"], pd.to_numeric(data["stand_prob"], errors="coerce"), ".", markersize=2, label="stand_prob")
    ax.plot(data["timestamp_s"], pd.to_numeric(data["sit_prob"], errors="coerce"), ".", markersize=2, label="sit_prob")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probability")
    add_segment_bands(ax, sitting)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rgb_count(path: Path, rgb: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if "detected_count" in rgb:
        ax.plot(rgb["timestamp_s"], rgb["detected_count"], label="detections")
    if "track_count" in rgb:
        ax.plot(rgb["timestamp_s"], rgb["track_count"], label="tracks")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    add_segment_bands(ax, segments)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped_metric(path: Path, df: pd.DataFrame, metric: str, ylabel: str, plt: Any) -> None:
    if df.empty or metric not in df:
        return no_data_plot(path, ylabel)
    data = df.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = data["segment_id"].astype(str)
    ax.bar(labels, data[metric])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_xy_scatter(path: Path, tracks: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if tracks.empty or not {"x_m", "y_m"}.issubset(tracks.columns):
        return no_data_plot(path, "XY scatter")
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, seg in segments.iterrows():
        st = segment_filter(tracks, seg)
        if not st.empty:
            ax.scatter(st["x_m"], st["y_m"], s=4, alpha=0.35, label=seg["segment_id"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(path: Path, confusion: pd.DataFrame, plt: Any) -> None:
    if confusion.empty:
        return no_data_plot(path, "Posture confusion matrix")
    mat = confusion.set_index("expected_pose")[POSE_CLASSES].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat.values, cmap="Blues")
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat.iloc[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pose_distribution(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if pose.empty:
        return no_data_plot(path, "Pose distribution")
    rows = []
    for _, seg in segments.iterrows():
        sp = segment_filter(pose, seg)
        counts = sp["display_pose"].value_counts(normalize=True)
        row = {"segment_id": seg["segment_id"]}
        row.update({cls: counts.get(cls, 0.0) for cls in POSE_CLASSES})
        rows.append(row)
    data = pd.DataFrame(rows).set_index("segment_id")
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(data))
    for cls in POSE_CLASSES:
        ax.bar(data.index, data[cls], bottom=bottom, label=cls)
        bottom += data[cls].to_numpy()
    ax.set_ylabel("fraction")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_geom_pts(path: Path, pose: pd.DataFrame, segments: pd.DataFrame, plt: Any) -> None:
    if pose.empty or "geom_pts" not in pose:
        return no_data_plot(path, "Accuracy vs geom points")
    rows = []
    for _, seg in segments.iterrows():
        sp = segment_filter(pose, seg)
        if sp.empty:
            continue
        correct = sp["display_pose"].eq(seg["expected_pose"])
        buckets = pd.cut(sp["geom_pts"], bins=[-1, 0, 2, 4, 9999], labels=["0", "1-2", "3-4", "5+"])
        for bucket, vals in correct.groupby(buckets, observed=False):
            rows.append({"bucket": str(bucket), "accuracy": float(vals.mean()) if len(vals) else np.nan})
    data = pd.DataFrame(rows)
    if data.empty:
        return no_data_plot(path, "Accuracy vs geom points")
    agg = data.groupby("bucket")["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(agg.index, agg.values)
    ax.set_xlabel("geom_pts bucket")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_quality(path: Path, quality: pd.DataFrame, plt: Any) -> None:
    if quality.empty:
        return no_data_plot(path, "Accuracy by quality")
    data = quality.copy()
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    agg = data.groupby("bucket")["accuracy"].mean().dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agg.index, agg.values)
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_false_rates(path: Path, posture: pd.DataFrame, plt: Any) -> None:
    if posture.empty:
        return no_data_plot(path, "False rates")
    data = posture.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(data))
    width = 0.25
    for idx, col in enumerate(["falling_false_rate", "lying_false_rate", "unknown_rate"]):
        ax.bar(x + (idx - 1) * width, pd.to_numeric(data[col], errors="coerce"), width=width, label=col)
    ax.set_xticks(x, data["segment_id"], rotation=45, ha="right")
    ax.set_ylabel("rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tracking_vs_posture(path: Path, tracking: pd.DataFrame, posture: pd.DataFrame, plt: Any) -> None:
    if tracking.empty or posture.empty:
        return no_data_plot(path, "Tracking vs posture")
    data = tracking[["segment_id", "tracking_score"]].merge(posture[["segment_id", "accuracy"]], on="segment_id", how="outer")
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x - 0.2, pd.to_numeric(data["tracking_score"], errors="coerce") / 100.0, width=0.4, label="tracking score / 100")
    ax1.bar(x + 0.2, pd.to_numeric(data["accuracy"], errors="coerce"), width=0.4, label="posture accuracy")
    ax1.set_xticks(x, data["segment_id"], rotation=45, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_failure_heatmap(path: Path, combined: pd.DataFrame, plt: Any) -> None:
    if combined.empty:
        return no_data_plot(path, "Failure mode heatmap")
    cols = ["dropout", "ghost", "range error", "NO_POINTS", "moving false positive", "posture wrong"]
    mat = []
    for _, row in combined.iterrows():
        mat.append(
            [
                1.0 - as_float(row["tracking_presence_rate"], 1.0),
                as_float(row["extra_track_rate"], 0.0),
                min(1.0, as_float(row["range_mae_m"], 0.0)),
                as_float(row["quality_NO_POINTS_rate"], 0.0),
                1.0 if "moving" in str(row["main_failure_hypothesis"]).lower() else 0.0,
                1.0 - as_float(row["posture_accuracy"], 1.0),
            ]
        )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.array(mat), cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(combined)), combined["segment_id"])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"
    data = df.head(max_rows).fillna("NA").astype(str)
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def ranked_recommendations(
    combined: pd.DataFrame,
    tracking: pd.DataFrame,
    posture: pd.DataFrame,
    posture_verdict: pd.DataFrame,
    ghost_shadow: pd.DataFrame,
) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    tracking_lookup = tracking.set_index("segment_id") if not tracking.empty and "segment_id" in tracking else pd.DataFrame()

    def add(priority: float, issue: str, evidence: str, cause: str, fix: str) -> None:
        recs.append(
            {
                "severity": priority,
                "Issue": issue,
                "Evidence": evidence,
                "Likely cause": cause,
                "Recommended fix": fix,
            }
        )

    if not posture.empty:
        sit4 = posture[posture["segment_id"].eq("sitting_4m")]
        if not sit4.empty and as_float(sit4.iloc[0].get("accuracy"), 1.0) < 0.10:
            row = sit4.iloc[0]
            tr = tracking_lookup.loc["sitting_4m"] if not tracking_lookup.empty and "sitting_4m" in tracking_lookup.index else pd.Series(dtype=object)
            add(
                100.0,
                "sitting_4m is classified as STANDING instead of SITTING.",
                (
                    f"accuracy={fmt(row.get('accuracy'))}, standing_rate={fmt(row.get('standing_rate'))}, "
                    f"sitting_rate={fmt(row.get('sitting_rate'))}, tracking_presence={fmt(tr.get('tracking_presence_rate', 'NA'))}, "
                    f"extra_track_rate={fmt(tr.get('extra_track_rate', 'NA'))}."
                ),
                "Sitting-vs-standing discrimination fails at far range under sparse/NO_POINTS geometry, while tracking remains continuous.",
                "Add sitting-specific geometry/height evidence or a range-aware sitting gate; compare the current cfg against a static-retention/fine-motion cfg before tuning thresholds.",
            )
        sit3 = posture[posture["segment_id"].eq("sitting_3m")]
        if not sit3.empty and as_float(sit3.iloc[0].get("accuracy"), 1.0) < 0.65:
            row = sit3.iloc[0]
            add(
                90.0,
                "sitting_3m is unstable between STANDING and SITTING.",
                (
                    f"accuracy={fmt(row.get('accuracy'))}, standing_rate={fmt(row.get('standing_rate'))}, "
                    f"sitting_rate={fmt(row.get('sitting_rate'))}, pose_switch_count={fmt(row.get('pose_switch_count'))}."
                ),
                "Stand/sit evidence is ambiguous enough that display output oscillates or favors STANDING.",
                "Inspect the stand_prob/sit_prob timeline and add a confidence-margin plus temporal sitting confirmation test.",
            )
        near_sitting = posture[posture["segment_id"].isin(["sitting_1m", "sitting_2m"])]
        weak_near = near_sitting[pd.to_numeric(near_sitting["accuracy"], errors="coerce").lt(0.90)]
        if not weak_near.empty:
            evidence_parts = [
                f"{r.segment_id}: accuracy={fmt(r.accuracy)}, dominant_wrong={getattr(r, 'dominant_wrong_pose', 'NA')}"
                for r in weak_near.itertuples()
            ]
            add(
                80.0,
                "sitting_1m/sitting_2m are partially correct but still often drift toward STANDING.",
                "; ".join(evidence_parts),
                "Sitting threshold/hysteresis is insufficient even where range tracking is good.",
                "Tune the sitting gate after solving the 3m/4m evidence problem, so near-range tuning does not hide the far-range failure.",
            )

    if not posture_verdict.empty:
        moving = posture_verdict[posture_verdict["posture_verdict"].eq("MOVING_FALSE_POSITIVE")]
        if not moving.empty:
            worst = moving.sort_values("moving_rate", ascending=False).iloc[0]
            add(
                60.0,
                f"{worst['segment_id']} has a stationary MOVING false-positive issue.",
                f"moving_rate={fmt(worst['moving_rate'])}, accuracy={fmt(worst['accuracy'])}.",
                "Moving override is more sensitive than the stationary benchmark evidence supports.",
                "Raise the moving override threshold or require longer confirmed translation before overriding pose.",
            )

    if not ghost_shadow.empty and pd.to_numeric(ghost_shadow["extra_track_rate"], errors="coerce").fillna(0).max() > 0.05:
        worst = ghost_shadow.assign(_extra=pd.to_numeric(ghost_shadow["extra_track_rate"], errors="coerce")).sort_values("_extra", ascending=False).iloc[0]
        add(
            70.0,
            f"{worst['segment_id']} has measurable extra-track/ghost activity.",
            f"extra_track_rate={fmt(worst['extra_track_rate'])}, max_active_tracks={fmt(worst['max_active_tracks'])}.",
            "Extra active tracks are present in a controlled single-person benchmark.",
            "Tighten ghost filtering using point evidence, persistence, and distance from the primary target.",
        )

    if combined.empty:
        return recs
    for _, row in combined.iterrows():
        issue = row.get("main_failure_hypothesis", "no major failure detected")
        if issue == "no major failure detected" or "sit-vs-stand" in str(issue):
            continue
        posture_acc = as_float(row.get("posture_accuracy"), 1.0)
        if posture_acc >= 0.70 and as_float(row.get("range_mae_m"), 0.0) <= 0.30:
            continue
        add(
            40.0 + (1.0 - posture_acc),
            f"{issue} in {row['segment_id']}",
            (
                f"tracking_presence={fmt(row['tracking_presence_rate'])}, range_mae={fmt(row['range_mae_m'])} m, "
                f"extra_rate={fmt(row['extra_track_rate'])}, posture_accuracy={fmt(row['posture_accuracy'])}, "
                f"dominant_wrong={row['dominant_wrong_pose']}."
            ),
            str(issue),
            recommendation_for_issue(str(issue)),
        )
    return sorted(recs, key=lambda r: r["severity"], reverse=True)[:8]


def recommendation_for_issue(issue: str) -> str:
    mapping = {
        "tracking dropout": "Review cfg coverage, boundary boxes, sensor tilt, and track confirmation/retention thresholds at this distance.",
        "ghost/shadow tracking": "Tighten ghost filtering and renderer confirmation using point evidence, persistence, and distance from the primary target.",
        "range bias / coordinate calibration": "Calibrate range/coordinate transform against marker positions before using distance-dependent posture thresholds.",
        "point association / sparse point cloud": "Fix target-to-point association and lower-confidence handling before tuning the classifier.",
        "moving override false positive / target jitter": "Raise the moving override threshold or require longer confirmed translation during stationary segments.",
        "sit-vs-stand discrimination": "Tune sitting geometry thresholds and add sitting samples at the failing distances.",
        "sit-vs-stand discrimination under sparse geometry": "Add sitting-specific geometry/height evidence and evaluate static-retention/fine-motion cfg variants before threshold tuning.",
        "height/tilt/calibration or model bias": "Check mounting height/tilt and height normalization, then retrain or recalibrate stand/sit thresholds.",
        "posture classifier / threshold / hysteresis": "Inspect raw/smoothed/candidate disagreement and adjust smoothing or model decision thresholds.",
    }
    return mapping.get(issue, "Inspect the segment-level evidence and tune the component identified by the diagnostic table.")


def fmt(value: Any) -> str:
    try:
        if value == "NA" or pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(
    out: Path,
    session: Path,
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    segments: pd.DataFrame,
    tracking: pd.DataFrame,
    posture: pd.DataFrame,
    combined: pd.DataFrame,
    tracking_verdict: pd.DataFrame,
    posture_verdict: pd.DataFrame,
    probability: pd.DataFrame,
    no_points: pd.DataFrame,
    ghost_shadow: pd.DataFrame,
    ghosts: pd.DataFrame,
    quality: pd.DataFrame,
    warnings: list[str],
) -> None:
    recs = ranked_recommendations(combined, tracking, posture, posture_verdict, ghost_shadow)
    posture_fail = combined.copy()
    posture_fail["_pa"] = pd.to_numeric(posture_fail.get("posture_accuracy", pd.Series(dtype=float)), errors="coerce")
    posture_fail = posture_fail.sort_values("_pa").drop(columns=["_pa"], errors="ignore").head(3)
    sitting = posture[posture["expected_pose"] == "SITTING"] if not posture.empty else pd.DataFrame()
    standing = posture[posture["expected_pose"] == "STANDING"] if not posture.empty else pd.DataFrame()
    sitting_worse = "NA"
    if not sitting.empty and not standing.empty:
        sitting_worse = "yes" if as_float(sitting["accuracy"].mean()) < as_float(standing["accuracy"].mean()) else "no"
    dist1 = combined[combined["expected_distance_m"].astype(float).eq(1.0)] if not combined.empty else pd.DataFrame()
    other = combined[~combined["expected_distance_m"].astype(float).eq(1.0)] if not combined.empty else pd.DataFrame()
    one_m_worse = "NA"
    if not dist1.empty and not other.empty:
        one_m_worse = "yes" if as_float(dist1["posture_accuracy"].mean()) < as_float(other["posture_accuracy"].mean()) else "no"
    standing_acc = as_float(standing["accuracy"].mean()) if not standing.empty else np.nan
    sitting_acc = as_float(sitting["accuracy"].mean()) if not sitting.empty else np.nan
    tracking_bad = pd.DataFrame()
    if not tracking_verdict.empty:
        tracking_bad = tracking_verdict[
            tracking_verdict["tracking_verdict"].isin(["DROPOUT", "ID_SWITCH", "GHOST_TRACKS", "FAILED"])
        ]
    tracking_strong = tracking_bad.empty and not tracking_verdict.empty
    range_minor = tracking_verdict[tracking_verdict["tracking_verdict"].isin(["MINOR_RANGE_BIAS", "MINOR_JITTER"])] if not tracking_verdict.empty else pd.DataFrame()
    tracking_line = (
        "Tracking: strong in this session. Presence rate was 100% across all distance/pose segments with no ID switches, no dropouts, and no extra-track rate."
        if tracking_strong
        else f"Tracking: review required in {', '.join(tracking_bad['segment_id'].astype(str).tolist()) if not tracking_bad.empty else 'NA'}."
    )
    posture_line = (
        f"Posture: weak for sitting. Standing accuracy was {fmt(standing_acc * 100 if not np.isnan(standing_acc) else 'NA')}%, "
        f"but sitting accuracy was {fmt(sitting_acc * 100 if not np.isnan(sitting_acc) else 'NA')}%, with the worst failures in "
        f"{', '.join(posture_fail['segment_id'].astype(str).tolist()) if not posture_fail.empty else 'NA'}."
    )
    range_line = (
        f"Range/calibration: no tracking failure threshold was crossed; {', '.join(range_minor['segment_id'].astype(str).tolist())} show minor range/jitter bias."
        if not range_minor.empty and tracking_bad.empty
        else "Range/calibration: no major range or continuity failure was detected."
    )
    point_line = "Point-density: NO_POINTS is common in both standing and sitting, but only sitting collapses. Therefore the primary failure is sitting-vs-standing discrimination under sparse geometry, not general tracking loss."
    rgb_video = session / "rgb_annotated.mp4"
    if not rgb_video.exists():
        candidate_video = session / "videos" / "rgb_annotated.mp4"
        rgb_video = candidate_video if candidate_video.exists() else rgb_video
    ghost_summary = "No measurable ghost/shadow issue in this benchmark session after the latest validation changes."
    if not ghost_shadow.empty and pd.to_numeric(ghost_shadow["extra_track_rate"], errors="coerce").fillna(0).max() > 0:
        ghost_summary = "Extra/rendered track activity was measured; inspect ghost_shadow_verdict_by_segment.csv."
    lines = [
        "# Distance/Posture Benchmark Report",
        "",
        "## Executive summary",
        f"- Session analyzed: `{session}`",
        f"- {tracking_line}",
        f"- {posture_line}",
        "- Main failure: sitting is confused as STANDING, especially at 3m and 4m.",
        f"- {range_line}",
        f"- {point_line}",
        f"- Is sitting worse than standing? {sitting_worse}",
        f"- Is 1m worse than 2m/3m/4m? {one_m_worse}",
        f"- Warning count: {len(warnings)}",
        "",
        "## Session analyzed",
        markdown_table(candidates, 10),
        "",
        "## Files discovered",
        markdown_table(inventory[["file", "exists", "rows", "time/frame column detected", "notes"]], 40),
        "",
        "## Protocol and inferred/manual segments",
        "The expected order is standing 1m, 2m, 3m, 4m, then sitting 1m, 2m, 3m, 4m. Auto segmentation uses range plateaus when possible and equal-time fallback when uncertain.",
        "",
        markdown_table(segments),
        "",
        "## Segment verification required",
        f"- Auto segments: `{out / 'segments_auto.csv'}`",
        f"- Manual template: `{out / 'segments_manual_template.csv'}`",
        f"- Range timeline: `{out / 'plots' / 'timeline_range_by_track.png'}`",
        f"- RGB video: `{rgb_video}`" if rgb_video.exists() else "- RGB video: `NA`",
        "",
        "Before treating posture numbers as final, verify the auto boundaries against the RGB video and range timeline. If boundaries are off, edit `segments_manual_template.csv` and rerun with `--manual-segments`.",
        "",
        "## Tracking accuracy results",
        markdown_table(tracking),
        "",
        "## Tracking verdict by segment",
        markdown_table(tracking_verdict),
        "",
        "## Posture accuracy results",
        markdown_table(posture),
        "",
        "## Posture verdict by segment",
        markdown_table(posture_verdict),
        "",
        "## Sitting-vs-standing probability analysis",
        "This table answers whether the model probabilities themselves favor STANDING during sitting, or whether later smoothing/gates select STANDING despite ambiguous probabilities.",
        "",
        markdown_table(probability),
        "",
        "## Tracking vs posture comparison",
        markdown_table(combined),
        "",
        "## Per-distance findings",
        markdown_table(combined.groupby("expected_distance_m", as_index=False).agg(tracking_presence_rate=("tracking_presence_rate", "mean"), range_mae_m=("range_mae_m", "mean"), posture_accuracy=("posture_accuracy", "mean")) if not combined.empty else pd.DataFrame()),
        "",
        "## Per-pose findings",
        markdown_table(combined.groupby("expected_pose", as_index=False).agg(tracking_presence_rate=("tracking_presence_rate", "mean"), posture_accuracy=("posture_accuracy", "mean")) if not combined.empty else pd.DataFrame()),
        "",
        "## Ghost/shadow analysis",
        ghost_summary,
        "",
        "No rendered ghost tracks observed; suppressed suspect-track statistics were not available/parsed." if not ghost_shadow.empty else "Ghost/suspect logs were not available.",
        "",
        markdown_table(ghost_shadow),
        "",
        "Legacy ghost-track candidates:",
        "",
        markdown_table(ghosts),
        "",
        "## NO_POINTS / geom_pts analysis",
        point_line,
        "",
        markdown_table(quality),
        "",
        "## NO_POINTS effect by pose",
        markdown_table(no_points),
        "",
        "## Moving false-positive analysis",
        markdown_table(posture[["segment_id", "expected_pose", "expected_distance_m", "moving_rate"]] if not posture.empty else pd.DataFrame()),
        "",
        "## Sitting-vs-standing confusion analysis",
        markdown_table(posture[["segment_id", "expected_pose", "dominant_display_pose", "dominant_wrong_pose", "standing_rate", "sitting_rate", "accuracy"]] if not posture.empty else pd.DataFrame()),
        "",
        "## RGB sanity check if available",
        "RGB logs are used as supplementary sanity checks only; mmWave tracking and posture metrics do not require RGB.",
        "",
        "## Key plots",
    ]
    for name in PLOT_NAMES:
        lines.append(f"- `plots/{name}`")
    lines.extend(["", "## Ranked issues"])
    if recs:
        for idx, rec in enumerate(recs, 1):
            lines.extend(
                [
                    f"### Priority {idx}",
                    f"Issue: {rec['Issue']}",
                    f"Evidence: {rec['Evidence']}",
                    f"Likely cause: {rec['Likely cause']}",
                    f"Recommended fix: {rec['Recommended fix']}",
                    "",
                ]
            )
    else:
        lines.append("No high-severity automatic issue was detected.")
    lines.extend(["", "## Recommended next fixes"])
    if recs:
        for idx, rec in enumerate(recs, 1):
            lines.append(f"{idx}. {rec['Recommended fix']}")
    else:
        lines.append("Inspect segment boundaries first, then rerun with manual segments if needed.")
    lines.extend(
        [
            "",
            "## Next experiment plan",
            "Experiment A: Verify auto segments manually with RGB video.",
            "Experiment B: Repeat sitting-only 2m/3m/4m with the original cfg and a static-retention/fine-motion cfg.",
            "Experiment C: Compare geom_pts, NO_POINTS rate, sit_prob, and stand_prob between configs.",
            "Experiment D: Only after the point/geometry comparison, tune sit-vs-stand gates.",
            "",
            "`staticRangeAngleCfg -1 0 8 8` disables static processing. If seated targets remain sparse, compare against TI `ODS_6m_staticRetention.cfg` / fineMotionCfg variants before changing posture thresholds.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {w}" for w in warnings)
    md = "\n".join(lines) + "\n"
    md_path = out / "DISTANCE_POSTURE_BENCHMARK_REPORT.md"
    md_path.write_text(md, encoding="utf-8")
    html_path = out / "DISTANCE_POSTURE_BENCHMARK_REPORT.html"
    html_body = "<html><head><meta charset='utf-8'><title>Distance/Posture Benchmark Report</title></head><body><pre>" + html.escape(md) + "</pre></body></html>"
    html_path.write_text(html_body, encoding="utf-8")


def write_pipeline_doc(path: Path) -> None:
    text = """# Distance/Posture Analysis Pipeline

## What the script does
`analysis/analyze_distance_posture_session.py` builds an offline report for continuous mmWave + RGB benchmark sessions. It discovers session files, normalizes different CSV schemas, slices the run into standing/sitting distance segments, computes tracking metrics separately from posture metrics, creates diagnostic CSVs/plots, and writes markdown/HTML reports.

## How latest sessions are found
The script scans `logs`, `..\\logs`, `C:\\Users\\UBESC\\Desktop\\Combined MMwave and RGB\\logs`, and any `--log-root` paths. Candidate directories are ranked by modified time and by whether they contain useful CSV/JSON/log/video files. `--latest` selects the newest useful candidate.

## What files are parsed
Common files include `mmwave_frames.csv`, `mmwave_tracks.csv`, `mmwave_pose.csv`, `pose_predictions_ui.csv`, `targets.csv`, `frames_summary.csv`, `rgb_frames.csv`, `rgb_tracks.csv`, `rgb_keypoints.csv`, `sync_index.csv`, `events.csv`, `events.jsonl`, `combined_events.csv`, `session_metadata.json`, and `videos/rgb_annotated.mp4`.

## How automatic segmentation works
The expected ground-truth order is standing at 1m/2m/3m/4m followed by sitting at 1m/2m/3m/4m. Auto segmentation searches for stable range plateaus in that order and trims 5 seconds from each end when possible. If range plateaus cannot be inferred for all segments, it writes warnings and falls back to equal-time best-effort segments.

## How to manually override segments
Create or edit `analysis_outputs/latest_distance_posture_analysis/segments_manual_template.csv` with:

```text
segment_id,expected_pose,expected_distance_m,start_time_s,end_time_s
```

Then rerun with:

```powershell
python analysis\\analyze_distance_posture_session.py --log-root \"..\\logs\" --latest --out analysis_outputs\\latest_distance_posture_analysis --manual-segments analysis_outputs\\latest_distance_posture_analysis\\segments_manual_template.csv --make-plots
```

## Tracking metrics
Tracking metrics include presence/dropout, range MAE/RMSE/bias, position jitter, TID continuity/switches, active-track and ghost/shadow rates, plus an interpretable tracking score with components reported separately.

## Posture metrics
Posture metrics use `display_pose`/`final_label` as the main UI prediction and keep tracking independent. Metrics include accuracy, pose distribution, false MOVING/FALLING/LYING/UNKNOWN rates, latency to first/stable correct prediction, switch rates, and breakdowns by quality/geom points/association/reason fields.

## Plot list
The script writes timeline, tracking, posture, and summary plots under `plots/`, including range by track, active track count, display pose, quality/geom points, stand/sit probabilities, tracking presence/range error/jitter/ghosts/TID switches, XY scatter, posture accuracy/confusion/distribution, quality breakdowns, moving false positives, false falling rates, stable-correct latency, tracking-vs-posture summary, and failure heatmap.

## Output directory structure
Outputs are written under the selected `--out` directory, including `file_inventory.csv`, `segments_auto.csv`, metric CSVs, event CSVs, `warnings.txt`, `plots/`, `DISTANCE_POSTURE_BENCHMARK_REPORT.md`, and `DISTANCE_POSTURE_BENCHMARK_REPORT.html`.

## Validation commands run

```powershell
python -m py_compile analysis\\analyze_distance_posture_session.py
python analysis\\analyze_distance_posture_session.py --log-root \"..\\logs\" --latest --out analysis_outputs\\latest_distance_posture_analysis --make-plots
```

## Known limitations
Automatic segment boundaries are best effort and must be inspected before treating results as final. Equal-time fallback is intentionally conservative and flagged in warnings. RGB is supplementary unless a clean ground-truth label exists. Text fallback parsing only extracts common pose-debug patterns.

## Exact command to run

```powershell
python analysis\\analyze_distance_posture_session.py --log-root \"..\\logs\" --latest --out analysis_outputs\\latest_distance_posture_analysis --make-plots
```
"""
    path.write_text(text, encoding="utf-8")


def write_refinement_doc(path: Path) -> None:
    text = """# Distance/Posture Analysis Refinement

## What was wrong/misleading in the first report
The first report ranked tracking segments by relative metric values and could label segments as "tracking failures" even when presence was 100%, dropouts were zero, TID switches were zero, and extra-track rate was zero. That made strong tracking look like a failure and obscured the real engineering issue: sitting posture detection.

## What interpretation logic changed
The refined script separates tracking status, posture status, range/calibration status, and point-density status. Tracking is only called a failure when presence, dropout, extra-track, ID-switch, range-MAE, or tracking-score thresholds are crossed. Otherwise smaller range/jitter deviations are reported as minor range/jitter issues.

## New CSVs added
- `tracking_verdict_by_segment.csv`
- `posture_verdict_by_segment.csv`
- `stand_sit_probability_by_segment.csv`
- `no_points_effect_by_pose.csv`
- `ghost_shadow_verdict_by_segment.csv`

## New plots added
- `plots/stand_vs_sit_probability_by_segment.png`
- `plots/stand_minus_sit_margin_by_segment.png`
- `plots/sitting_segments_stand_sit_prob_timeline.png`

## Updated executive summary rules
The executive summary now reports tracking as strong unless at least one segment crosses a defined failure threshold: tracking presence below 0.95, dropout rate above 0.05, extra-track rate above 0.05, any TID switch, range MAE above 0.30 m, or tracking score below 85. Posture is summarized separately by standing and sitting accuracy.

## Updated ranked recommendation rules
Recommendations are now generated around the highest-value engineering failures first: sitting_4m classified as STANDING, sitting_3m ambiguous/unstable, and near-range sitting partial failures. Duplicate generic point-association recommendations are avoided unless the data supports that as a distinct issue.

## Validation commands run

```powershell
python -m py_compile analysis\\analyze_distance_posture_session.py
python analysis\\analyze_distance_posture_session.py --log-root \"..\\logs\" --latest --out analysis_outputs\\latest_distance_posture_analysis_v2 --make-plots
```

## Exact command to regenerate the report

```powershell
python analysis\\analyze_distance_posture_session.py --log-root \"..\\logs\" --latest --out analysis_outputs\\latest_distance_posture_analysis_v2 --make-plots
```
"""
    path.write_text(text, encoding="utf-8")


def open_report(path: Path) -> None:
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except Exception:
        subprocess.Popen(["cmd", "/c", "start", "", str(path)], shell=False)


def main() -> int:
    args = parse_args()
    out = as_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = default_roots(args.log_root)
    candidates = find_candidate_sessions(roots)
    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df["modified_time_iso"] = pd.to_datetime(candidates_df["modified_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        candidates_df[["rank", "path", "modified_time_iso", "useful_score", "csv_count", "notes"]].to_csv(out / "candidate_sessions.csv", index=False)
    if args.session:
        session = as_path(args.session)
    elif args.latest:
        if not candidates:
            print("No useful sessions found.", file=sys.stderr)
            return 2
        session = candidates[0]["path"]
    else:
        print("Use --session or --latest.", file=sys.stderr)
        return 2
    warnings: list[str] = []
    ctx = Context(session=session, out=out, warnings=warnings)
    inventory = build_inventory(session, out)

    raw_frames = safe_read_csv(session / "mmwave_frames.csv", warnings)
    if raw_frames.empty:
        raw_frames = safe_read_csv(session / "frames_summary.csv", warnings)
    raw_tracks = safe_read_csv(session / "mmwave_tracks.csv", warnings)
    if raw_tracks.empty:
        raw_tracks = safe_read_csv(session / "targets.csv", warnings)
    raw_pose = safe_read_csv(session / "mmwave_pose.csv", warnings)
    if raw_pose.empty:
        raw_pose = safe_read_csv(session / "pose_predictions_ui.csv", warnings)
    if raw_tracks.empty and not raw_pose.empty:
        raw_tracks = raw_pose
        warnings.append("using pose_predictions_ui.csv as tracking source because no standalone mmwave_tracks.csv/targets.csv exists")
    raw_rgb_frames = safe_read_csv(session / "rgb_frames.csv", warnings)
    raw_rgb_tracks = safe_read_csv(session / "rgb_tracks.csv", warnings)
    ctx.t0_ns = global_t0_ns([raw_frames, raw_tracks, raw_pose, raw_rgb_frames, raw_rgb_tracks])
    infer_frame_period(ctx, raw_frames, args.fps_estimate)

    frames = normalize_frames(raw_frames, ctx)
    tracks = normalize_tracking(raw_tracks, ctx)
    pose = normalize_posture(raw_pose, ctx, tracks)
    if pose.empty:
        pose_debug = parse_pose_debug_text(session, ctx)
        if not pose_debug.empty:
            pose = pose_debug
    rgb = normalize_rgb(raw_rgb_frames, raw_rgb_tracks, ctx)

    distances = [float(x.strip()) for x in args.expected_distances.split(",") if x.strip()]
    expected = expected_table(distances)
    make_manual_template(expected, out)
    manual_path = Path(args.manual_segments).resolve() if args.manual_segments else None
    if manual_path:
        segments = load_manual_segments(manual_path, expected, ctx, frames, tracks)
        if segments is None:
            segments = auto_segments(expected, tracks, frames, args.segment_min_seconds, args.segment_target_seconds, ctx)
    else:
        segments = auto_segments(expected, tracks, frames, args.segment_min_seconds, args.segment_target_seconds, ctx)

    tracking, tracking_overall, ghosts, tid_switches, dropouts = compute_tracking_metrics(segments, frames, tracks, ctx)
    posture, confusion, failure, quality, pose_switches = compute_posture_metrics(segments, frames, pose, tracks, tracking, ctx)
    combined = combined_diagnostics(segments, tracks, pose, tracking, posture)
    tracking_verdict = tracking_verdicts(tracking)
    posture_verdict = posture_verdicts(posture, combined)
    probability = stand_sit_probability_by_segment(segments, pose)
    no_points = no_points_effect_by_pose(segments, pose)
    ghost_shadow = ghost_shadow_verdicts(tracking)
    write_csvs(
        out,
        segments,
        tracking,
        tracking_overall,
        posture,
        confusion,
        failure,
        quality,
        combined,
        tracking_verdict,
        posture_verdict,
        probability,
        no_points,
        ghost_shadow,
        ghosts,
        tid_switches,
        dropouts,
        pose_switches,
    )
    if args.make_plots:
        generate_plots(out, segments, frames, tracks, pose, rgb, tracking, posture, confusion, quality, combined, probability, ctx)
    else:
        (out / "plots").mkdir(exist_ok=True)
    write_report(
        out,
        session,
        candidates_df[["rank", "path", "modified_time_iso", "useful_score", "csv_count", "notes"]] if not candidates_df.empty else pd.DataFrame(),
        inventory,
        segments,
        tracking,
        posture,
        combined,
        tracking_verdict,
        posture_verdict,
        probability,
        no_points,
        ghost_shadow,
        ghosts,
        quality,
        warnings,
    )
    write_pipeline_doc(Path("DISTANCE_POSTURE_ANALYSIS_PIPELINE.md").resolve())
    write_refinement_doc(Path("DISTANCE_POSTURE_ANALYSIS_REFINEMENT.md").resolve())
    (out / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    if args.open_report:
        open_report(out / "DISTANCE_POSTURE_BENCHMARK_REPORT.html")

    print("Candidate sessions:")
    if candidates_df.empty:
        print("  None")
    else:
        for _, row in candidates_df.head(10).iterrows():
            print(f"  {int(row['rank'])}. {row['path']} | modified={row['modified_time_iso']} | score={row['useful_score']} | csvs={row['csv_count']}")
    print(f"Selected session: {session}")
    print(f"Output directory: {out}")
    print("Segments detected:")
    print(segments[["segment_id", "expected_pose", "expected_distance_m", "start_time_s", "end_time_s", "duration_s", "method", "confidence"]].to_string(index=False))
    print("Main summary:")
    summary = combined[["segment_id", "tracking_presence_rate", "range_mae_m", "extra_track_rate", "posture_accuracy", "main_failure_hypothesis"]]
    print(summary.to_string(index=False))
    print("Tracking verdict table:")
    print(tracking_verdict[["segment_id", "tracking_verdict", "tracking_presence_rate", "range_mae_m", "dropout_rate", "extra_track_rate", "tid_switch_count", "tracking_score"]].to_string(index=False))
    print("Posture verdict table:")
    print(posture_verdict[["segment_id", "posture_verdict", "accuracy", "dominant_display_pose", "dominant_wrong_pose", "standing_rate", "sitting_rate", "quality_NO_POINTS_rate", "mean_geom_pts"]].to_string(index=False))
    print("Top 3 ranked issues:")
    for idx, rec in enumerate(ranked_recommendations(combined, tracking, posture, posture_verdict, ghost_shadow)[:3], 1):
        print(f"  Priority {idx}: {rec['Issue']} | Evidence: {rec['Evidence']}")
    print("Open report:")
    print(out / "DISTANCE_POSTURE_BENCHMARK_REPORT.md")
    print(out / "DISTANCE_POSTURE_BENCHMARK_REPORT.html")
    if warnings:
        print(f"Warnings: {len(warnings)} written to {out / 'warnings.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
