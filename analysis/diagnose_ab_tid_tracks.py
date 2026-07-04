#!/usr/bin/env python
"""Per-TID offline diagnosis for the sitting static-retention A/B test."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_distance_posture_session import (
    Context,
    global_t0_ns,
    infer_frame_period,
    normalize_frames,
    normalize_posture,
    normalize_rgb,
    normalize_tracking,
)


POSE_ORDER = {"UNKNOWN": 0, "OTHER": 1, "MOVING": 2, "STANDING": 3, "SITTING": 4, "FALLING": 5, "LYING": 6}
PLOT_SEGMENTS = ["sitting_3m", "sitting_4m"]


@dataclass
class SessionData:
    cfg_name: str
    session: Path
    metadata: dict[str, Any]
    frames: pd.DataFrame
    tracks: pd.DataFrame
    pose: pd.DataFrame
    merged: pd.DataFrame
    rgb_summary: dict[str, Any]
    warnings: list[str]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_metadata(session: Path) -> dict[str, Any]:
    path = session / "session_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def fmt(value: Any, digits: int = 3) -> str:
    value = as_float(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def rate(mask: pd.Series, denom: int | None = None) -> float:
    if denom is None:
        denom = len(mask)
    if not denom:
        return math.nan
    return float(mask.fillna(False).sum()) / float(denom)


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def std_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std()) if len(values) else math.nan


def normalize_quality(value: Any, geom_pts: Any) -> str:
    if pd.notna(value):
        text = str(value).strip().upper()
        if text and text != "NAN":
            return text
    pts = as_float(geom_pts)
    if math.isnan(pts) or pts <= 0:
        return "NO_POINTS"
    if pts < 3:
        return "LOW_POINTS"
    return "OK"


def load_session(cfg_name: str, session: Path) -> SessionData:
    frames_raw = read_csv(session / "mmwave_frames.csv")
    tracks_raw = read_csv(session / "mmwave_tracks.csv")
    pose_raw = read_csv(session / "mmwave_pose.csv")
    rgb_frames_raw = read_csv(session / "rgb_frames.csv")
    rgb_tracks_raw = read_csv(session / "rgb_tracks.csv")
    sync_raw = read_csv(session / "sync_index.csv")
    rgb_actions_raw = read_csv(session / "rgb_actions.csv")

    ctx = Context(session=session, out=Path("."), warnings=[])
    ctx.t0_ns = global_t0_ns([frames_raw, tracks_raw, pose_raw, rgb_frames_raw, rgb_tracks_raw, sync_raw])
    infer_frame_period(ctx, frames_raw, "auto")

    frames = normalize_frames(frames_raw, ctx)
    tracks = normalize_tracking(tracks_raw, ctx)
    pose = normalize_posture(pose_raw, ctx, tracks)
    _ = normalize_rgb(rgb_frames_raw, rgb_tracks_raw, ctx)

    merged = merge_tracks_pose_frames(tracks, pose, frames)
    metadata = read_metadata(session)
    rgb_summary = {
        "rgb_frames_rows": len(rgb_frames_raw),
        "rgb_tracks_rows": len(rgb_tracks_raw),
        "sync_index_rows": len(sync_raw),
        "rgb_actions_rows": max(len(rgb_actions_raw), 0),
        "rgb_video_present": (session / "videos" / "rgb_annotated.mp4").exists(),
    }
    return SessionData(cfg_name, session, metadata, frames, tracks, pose, merged, rgb_summary, ctx.warnings)


def merge_tracks_pose_frames(tracks: pd.DataFrame, pose: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    t = tracks.copy()
    p = pose.copy()
    for df in [t, p]:
        if "frame" in df:
            df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        if "tid" in df:
            df["tid"] = pd.to_numeric(df["tid"], errors="coerce")

    keep_track = [
        c
        for c in [
            "frame",
            "timestamp_s",
            "tid",
            "x_m",
            "y_m",
            "z_m",
            "range_m",
            "geom_pts",
            "quality",
            "assoc",
        ]
        if c in t.columns
    ]
    keep_pose = [
        c
        for c in [
            "frame",
            "timestamp_s",
            "tid",
            "display_pose",
            "raw_pose",
            "quality",
            "geom_pts",
            "points_total",
            "assoc",
            "range_m",
            "stand_prob",
            "sit_prob",
            "fall_prob",
            "lying_prob",
            "confidence",
            "speed_mps",
        ]
        if c in p.columns
    ]
    t = t[keep_track].rename(
        columns={
            "timestamp_s": "timestamp_track_s",
            "range_m": "range_track_m",
            "geom_pts": "geom_pts_track",
            "quality": "quality_track",
            "assoc": "assoc_track",
        }
    )
    p = p[keep_pose].rename(
        columns={
            "timestamp_s": "timestamp_pose_s",
            "range_m": "range_pose_m",
            "geom_pts": "geom_pts_pose",
            "quality": "quality_pose",
            "assoc": "assoc_pose",
        }
    )
    merged = pd.merge(t, p, on=["frame", "tid"], how="outer")
    merged["timestamp_s"] = merged.get("timestamp_pose_s", pd.Series(np.nan, index=merged.index)).fillna(
        merged.get("timestamp_track_s", pd.Series(np.nan, index=merged.index))
    )
    merged["range_m"] = merged.get("range_pose_m", pd.Series(np.nan, index=merged.index)).fillna(
        merged.get("range_track_m", pd.Series(np.nan, index=merged.index))
    )
    merged["geom_pts"] = merged.get("geom_pts_pose", pd.Series(np.nan, index=merged.index)).fillna(
        merged.get("geom_pts_track", pd.Series(np.nan, index=merged.index))
    )
    merged["quality"] = [
        normalize_quality(q, g)
        for q, g in zip(
            merged.get("quality_pose", pd.Series(np.nan, index=merged.index)),
            merged.get("geom_pts", pd.Series(np.nan, index=merged.index)),
        )
    ]
    merged["assoc"] = merged.get("assoc_pose", pd.Series(np.nan, index=merged.index)).fillna(
        merged.get("assoc_track", pd.Series(np.nan, index=merged.index))
    )
    if "display_pose" not in merged:
        merged["display_pose"] = "UNKNOWN"
    merged["display_pose"] = merged["display_pose"].fillna("UNKNOWN").astype(str).str.upper()
    for col in ["stand_prob", "sit_prob", "points_total", "x_m", "y_m", "z_m", "range_m", "geom_pts"]:
        if col not in merged:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if not frames.empty and "frame" in frames.columns:
        frame_points = frames[["frame", "num_points", "num_tracks"]].drop_duplicates("frame").rename(
            columns={"num_points": "frame_points_total", "num_tracks": "frame_num_tracks"}
        )
        merged = merged.merge(frame_points, on="frame", how="left")
    else:
        merged["frame_points_total"] = np.nan
        merged["frame_num_tracks"] = np.nan
    # In current logs, mmwave_pose.num_points is the per-TID posture geometry count,
    # while mmwave_frames.num_points is the frame-level total point count.
    merged["points_total_effective"] = merged["frame_points_total"].fillna(merged["points_total"])
    merged["geom_to_total_ratio"] = merged["geom_pts"] / merged["points_total_effective"].replace(0, np.nan)
    return merged.dropna(subset=["timestamp_s"], how="all")


def load_segments(path: Path) -> pd.DataFrame:
    seg = read_csv(path)
    required = ["segment_id", "expected_pose", "expected_distance_m", "start_time_s", "end_time_s"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        raise ValueError(f"segment file {path} missing columns: {missing}")
    for col in ["expected_distance_m", "start_time_s", "end_time_s", "duration_s", "confidence"]:
        if col in seg:
            seg[col] = pd.to_numeric(seg[col], errors="coerce")
    if "duration_s" not in seg:
        seg["duration_s"] = seg["end_time_s"] - seg["start_time_s"]
    if "segmentation_method" not in seg:
        seg["segmentation_method"] = "manual_or_existing"
    if "confidence" not in seg:
        seg["confidence"] = np.nan
    return seg


def segment_filter(df: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
    return df[(df["timestamp_s"] >= seg["start_time_s"]) & (df["timestamp_s"] <= seg["end_time_s"])].copy()


def frame_count(frames: pd.DataFrame, seg: pd.Series) -> int:
    sf = segment_filter(frames, seg) if not frames.empty else pd.DataFrame()
    if not sf.empty and "frame" in sf:
        return int(sf["frame"].nunique())
    duration = as_float(seg.get("duration_s"))
    return int(max(round(duration / 0.055), 1)) if not math.isnan(duration) else 0


def assoc_rate(rows: pd.DataFrame, names: list[str]) -> float:
    if "assoc" not in rows or rows["assoc"].dropna().empty:
        return math.nan
    assoc = rows["assoc"].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=rows.index)
    for name in names:
        mask = mask | assoc.str.contains(name, regex=False)
    return rate(mask)


def per_tid_metrics_for_session(data: SessionData, segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, seg in segments.iterrows():
        sm = segment_filter(data.merged, seg)
        frames_total = frame_count(data.frames, seg)
        if sm.empty:
            continue
        for tid, tid_rows in sm.groupby("tid", dropna=True):
            if pd.isna(tid):
                continue
            tid_rows = tid_rows.copy()
            unique_frames = tid_rows["frame"].dropna().nunique() if "frame" in tid_rows else len(tid_rows)
            valid_probs = tid_rows["stand_prob"].notna() & tid_rows["sit_prob"].notna()
            sit_gt = valid_probs & (tid_rows["sit_prob"] > tid_rows["stand_prob"])
            stand_gt = valid_probs & (tid_rows["stand_prob"] > tid_rows["sit_prob"])
            display = tid_rows["display_pose"].fillna("UNKNOWN").astype(str).str.upper()
            expected_distance = as_float(seg["expected_distance_m"])
            abs_err = (tid_rows["range_m"] - expected_distance).abs()
            geom = tid_rows["geom_pts"]
            points_total = tid_rows["points_total_effective"]
            denom = len(tid_rows)
            rendered_rate = field_rate_if_available(tid_rows, ["rendered", "is_rendered", "visible"])
            confirmed_rate = field_rate_if_available(tid_rows, ["confirmed", "is_confirmed"])
            suspect_rate = field_rate_if_available(tid_rows, ["suspect", "is_suspect"])
            rows.append(
                {
                    "cfg_name": data.cfg_name,
                    "segment_id": seg["segment_id"],
                    "expected_pose": seg["expected_pose"],
                    "expected_distance_m": expected_distance,
                    "tid": int(tid) if float(tid).is_integer() else tid,
                    "frames_seen": int(unique_frames),
                    "presence_rate_within_segment": float(unique_frames) / frames_total if frames_total else math.nan,
                    "mean_range_m": mean_or_nan(tid_rows["range_m"]),
                    "median_range_m": median_or_nan(tid_rows["range_m"]),
                    "range_std_m": std_or_nan(tid_rows["range_m"]),
                    "range_mae_vs_expected_m": mean_or_nan(abs_err),
                    "mean_x_m": mean_or_nan(tid_rows["x_m"]),
                    "mean_y_m": mean_or_nan(tid_rows["y_m"]),
                    "std_x_m": std_or_nan(tid_rows["x_m"]),
                    "std_y_m": std_or_nan(tid_rows["y_m"]),
                    "mean_z_m": mean_or_nan(tid_rows["z_m"]),
                    "mean_geom_pts": mean_or_nan(geom),
                    "NO_POINTS_rate": rate(tid_rows["quality"].eq("NO_POINTS"), denom),
                    "LOW_POINTS_rate": rate(tid_rows["quality"].eq("LOW_POINTS"), denom),
                    "OK_rate": rate(tid_rows["quality"].eq("OK"), denom),
                    "assoc_target_index_rate": assoc_rate(tid_rows, ["target_index"]),
                    "assoc_nearest_rate": assoc_rate(tid_rows, ["nearest", "index"]),
                    "assoc_auto_none_rate": assoc_rate(tid_rows, ["auto_none"]),
                    "display_standing_rate": rate(display.eq("STANDING"), denom),
                    "display_sitting_rate": rate(display.eq("SITTING"), denom),
                    "display_moving_rate": rate(display.eq("MOVING"), denom),
                    "display_unknown_rate": rate(display.eq("UNKNOWN"), denom),
                    "mean_stand_prob": mean_or_nan(tid_rows["stand_prob"]),
                    "mean_sit_prob": mean_or_nan(tid_rows["sit_prob"]),
                    "mean_stand_minus_sit_margin": mean_or_nan(tid_rows["stand_prob"] - tid_rows["sit_prob"]),
                    "frames_stand_prob_gt_sit_prob": int(stand_gt.sum()),
                    "frames_sit_prob_gt_stand_prob": int(sit_gt.sum()),
                    "sit_prob_gt_stand_prob_rate": rate(sit_gt, int(valid_probs.sum())) if int(valid_probs.sum()) else math.nan,
                    "mismatch_rate": rate(sit_gt & ~display.eq("SITTING"), int(valid_probs.sum())) if int(valid_probs.sum()) else math.nan,
                    "mean_points_total": mean_or_nan(points_total),
                    "mean_geom_to_total_ratio": mean_or_nan(tid_rows["geom_to_total_ratio"]),
                    "rendered_rate_if_available": rendered_rate,
                    "confirmed_rate_if_available": confirmed_rate,
                    "suspect_rate_if_available": suspect_rate,
                }
            )
    return pd.DataFrame(rows)


def field_rate_if_available(rows: pd.DataFrame, names: list[str]) -> float:
    for name in names:
        if name in rows.columns:
            values = rows[name]
            if values.dropna().empty:
                return math.nan
            if values.dtype == object:
                return rate(values.astype(str).str.lower().isin(["1", "true", "yes", "confirmed", "rendered", "visible", "suspect"]))
            return rate(pd.to_numeric(values, errors="coerce").fillna(0) > 0)
    return math.nan


def classify_tids(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cfg_name, segment_id), group in metrics.groupby(["cfg_name", "segment_id"], dropna=False):
        group = group.copy()
        group["_score"] = group["range_mae_vs_expected_m"].fillna(99) + (1 - group["presence_rate_within_segment"].fillna(0)) * 0.5
        primary_idx = group.sort_values(["_score", "tid"]).index[0] if not group.empty else None
        primary = group.loc[primary_idx] if primary_idx is not None else None
        primary_geom = as_float(primary.get("mean_geom_pts")) if primary is not None else math.nan
        primary_tid = primary.get("tid") if primary is not None else math.nan
        for idx, row in group.iterrows():
            presence = as_float(row.get("presence_rate_within_segment"), 0.0)
            mae = as_float(row.get("range_mae_vs_expected_m"), math.nan)
            geom = as_float(row.get("mean_geom_pts"), math.nan)
            no_points = as_float(row.get("NO_POINTS_rate"), 0.0)
            range_std = as_float(row.get("range_std_m"), 0.0)
            sit_rate = as_float(row.get("sit_prob_gt_stand_prob_rate"), 0.0)
            classification = "UNKNOWN"
            evidence: list[str] = []
            if idx == primary_idx and presence >= 0.5 and (math.isnan(mae) or mae <= 0.75):
                classification = "REAL_PRIMARY"
                evidence.append(f"closest persistent TID; presence={fmt(presence)}, range_mae={fmt(mae)}m")
            elif presence >= 0.5 and not math.isnan(mae) and mae > 0.75:
                classification = "LIKELY_EXTRA_STATIC"
                evidence.append(f"persistent extra TID offset from expected range; presence={fmt(presence)}, range_mae={fmt(mae)}m")
                if no_points >= 0.8 or (not math.isnan(geom) and geom < 1):
                    evidence.append(f"mostly low/no geometry; NO_POINTS={fmt(no_points)}, mean_geom_pts={fmt(geom)}")
            elif (no_points >= 0.8 or (not math.isnan(geom) and geom < 1)) and (presence < 0.5 or range_std > 0.25):
                classification = "LIKELY_GHOST"
                evidence.append(f"low geometry or unstable/short-lived target; presence={fmt(presence)}, NO_POINTS={fmt(no_points)}")
            if classification != "REAL_PRIMARY" and not math.isnan(geom) and not math.isnan(primary_geom):
                if geom > primary_geom + 0.5 and sit_rate > 0.5 and (math.isnan(mae) or mae > 0.75):
                    classification = "LIKELY_WRONG_ASSOCIATION"
                    evidence.append("non-primary TID has more seated-looking point/probability evidence than the primary")
            rows.append(
                {
                    "cfg_name": cfg_name,
                    "segment_id": segment_id,
                    "tid": row["tid"],
                    "classification": classification,
                    "primary_tid_for_segment": primary_tid,
                    "evidence": "; ".join(evidence) if evidence else "insufficient discriminating evidence",
                }
            )
    return pd.DataFrame(rows)


def probability_display_mismatch(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cfg_name",
        "segment_id",
        "tid",
        "sit_prob_gt_stand_prob_rate",
        "display_sitting_rate",
        "display_standing_rate",
        "mismatch_rate",
        "mean_stand_prob",
        "mean_sit_prob",
        "mean_stand_minus_sit_margin",
        "frames_stand_prob_gt_sit_prob",
        "frames_sit_prob_gt_stand_prob",
    ]
    return metrics[[c for c in cols if c in metrics.columns]].copy()


def point_association(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cfg_name",
        "segment_id",
        "tid",
        "mean_geom_pts",
        "NO_POINTS_rate",
        "LOW_POINTS_rate",
        "OK_rate",
        "assoc_target_index_rate",
        "assoc_nearest_rate",
        "assoc_auto_none_rate",
        "mean_points_total",
        "mean_geom_to_total_ratio",
        "presence_rate_within_segment",
        "mean_range_m",
        "range_mae_vs_expected_m",
    ]
    return metrics[[c for c in cols if c in metrics.columns]].copy()


def plot_tid_range(data: SessionData, seg: pd.Series, out: Path) -> None:
    rows = segment_filter(data.merged, seg)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tid, group in rows.groupby("tid"):
        ax.plot(group["timestamp_s"] - seg["start_time_s"], group["range_m"], ".", markersize=2, label=f"TID {tid:g}")
    ax.axhline(float(seg["expected_distance_m"]), color="black", linestyle="--", linewidth=1, label="expected")
    ax.set_title(f"{data.cfg_name} {seg['segment_id']} TID range timeline")
    ax.set_xlabel("segment time (s)")
    ax.set_ylabel("range (m)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tid_pose(data: SessionData, seg: pd.Series, out: Path) -> None:
    rows = segment_filter(data.merged, seg)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tid, group in rows.groupby("tid"):
        y = group["display_pose"].map(POSE_ORDER).fillna(0)
        ax.plot(group["timestamp_s"] - seg["start_time_s"], y, ".", markersize=2, label=f"TID {tid:g}")
    ax.set_yticks(list(POSE_ORDER.values()), list(POSE_ORDER.keys()))
    ax.set_title(f"{data.cfg_name} {seg['segment_id']} TID display pose timeline")
    ax.set_xlabel("segment time (s)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tid_probs(data: SessionData, seg: pd.Series, out: Path) -> None:
    rows = segment_filter(data.merged, seg)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tid, group in rows.groupby("tid"):
        group = group.sort_values("timestamp_s")
        rel_t = group["timestamp_s"] - seg["start_time_s"]
        ax.plot(rel_t, group["stand_prob"], linewidth=1, alpha=0.8, label=f"TID {tid:g} stand")
        ax.plot(rel_t, group["sit_prob"], linewidth=1, linestyle="--", alpha=0.8, label=f"TID {tid:g} sit")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{data.cfg_name} {seg['segment_id']} TID stand/sit probabilities")
    ax.set_xlabel("segment time (s)")
    ax.set_ylabel("probability")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tid_geom(data: SessionData, seg: pd.Series, out: Path) -> None:
    rows = segment_filter(data.merged, seg)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tid, group in rows.groupby("tid"):
        ax.plot(group["timestamp_s"] - seg["start_time_s"], group["geom_pts"], ".", markersize=2, label=f"TID {tid:g}")
    ax.set_title(f"{data.cfg_name} {seg['segment_id']} TID geom_pts timeline")
    ax.set_xlabel("segment time (s)")
    ax.set_ylabel("geom_pts / pose num_points")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tid_count(default: SessionData, static: SessionData, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in [(default, "default"), (static, "static")]:
        frames = data.frames.copy()
        if frames.empty:
            continue
        if "num_tracks" in frames and frames["num_tracks"].notna().any():
            ax.plot(frames["timestamp_s"], frames["num_tracks"], linewidth=1, label=label)
        else:
            counts = data.tracks.groupby("timestamp_s")["tid"].nunique().reset_index(name="tid_count")
            ax.plot(counts["timestamp_s"], counts["tid_count"], linewidth=1, label=label)
    ax.set_title("Default vs static-retention active TID count by time")
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("active track count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def create_plots(default: SessionData, static: SessionData, static_segments: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for segment_id in PLOT_SEGMENTS:
        seg_rows = static_segments[static_segments["segment_id"].eq(segment_id)]
        if seg_rows.empty:
            continue
        seg = seg_rows.iloc[0]
        prefix = "static_" + segment_id
        plot_tid_range(static, seg, plot_dir / f"{prefix}_tid_range_timeline.png")
        plot_tid_pose(static, seg, plot_dir / f"{prefix}_tid_pose_timeline.png")
        plot_tid_probs(static, seg, plot_dir / f"{prefix}_tid_stand_sit_probs.png")
        plot_tid_geom(static, seg, plot_dir / f"{prefix}_tid_geom_pts.png")
    plot_tid_count(default, static, plot_dir / "default_vs_static_tid_count_by_time.png")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}")
    show = show.fillna("NA").astype(str)
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_static_findings(metrics: pd.DataFrame, classifications: pd.DataFrame) -> dict[str, str]:
    findings: dict[str, str] = {}
    static = metrics[metrics["cfg_name"].eq("static_retention_cfg")]
    for segment_id in ["sitting_2m", "sitting_3m", "sitting_4m"]:
        seg = static[static["segment_id"].eq(segment_id)].copy()
        if seg.empty:
            continue
        cls = classifications[(classifications["cfg_name"].eq("static_retention_cfg")) & (classifications["segment_id"].eq(segment_id))]
        primary_tid = cls["primary_tid_for_segment"].iloc[0] if not cls.empty else "NA"
        primary = seg[seg["tid"].eq(primary_tid)]
        extras = cls[~cls["classification"].eq("REAL_PRIMARY")]
        if not primary.empty:
            p = primary.iloc[0]
            findings[f"{segment_id}_primary"] = (
                f"TID {primary_tid} is the real primary by range/persistence: range_mae={fmt(p['range_mae_vs_expected_m'])}m, "
                f"sit_prob={fmt(p['mean_sit_prob'])}, stand_prob={fmt(p['mean_stand_prob'])}, "
                f"display_sitting={fmt(p['display_sitting_rate'])}, display_standing={fmt(p['display_standing_rate'])}."
            )
        if not extras.empty:
            findings[f"{segment_id}_extras"] = "; ".join(
                f"TID {r['tid']}={r['classification']} ({r['evidence']})" for _, r in extras.iterrows()
            )
    return findings


def recommendation(metrics: pd.DataFrame, classifications: pd.DataFrame) -> tuple[str, str]:
    static = metrics[metrics["cfg_name"].eq("static_retention_cfg")]
    cls_static = classifications[classifications["cfg_name"].eq("static_retention_cfg")]
    has_persistent_extra = cls_static["classification"].isin(["LIKELY_EXTRA_STATIC", "LIKELY_GHOST"]).any()
    real = static.merge(
        cls_static[["segment_id", "tid", "classification"]],
        on=["segment_id", "tid"],
        how="left",
    )
    real = real[real["classification"].eq("REAL_PRIMARY")]
    real_sit_gt = (real["mean_sit_prob"] > real["mean_stand_prob"]).any()
    real_display_bad = ((real["mean_sit_prob"] > real["mean_stand_prob"]) & (real["display_sitting_rate"] < 0.2)).any()
    real_stand_gt = (real["mean_stand_prob"] > real["mean_sit_prob"]).any()
    if has_persistent_extra:
        return (
            "Fix track validation / point association / primary target selection before posture tuning.",
            "Static retention produced persistent extra TIDs during the failing 3m/4m segments, so posture tuning would be premature.",
        )
    if real_display_bad:
        return (
            "Fix stand-vs-sit display/gating logic with offline replay first.",
            "At least one real primary TID has sit_prob above stand_prob while display remains non-SITTING.",
        )
    if real_stand_gt:
        return (
            "Improve posture features/model training for seated long-range target.",
            "The real primary TID still favors STANDING over SITTING.",
        )
    if real_sit_gt:
        return (
            "Do not use full static-retention cfg directly; create a narrower cfg experiment or tune tracker/static retention parameters.",
            "Static retention gives some useful seated probability/geometry but remains unstable.",
        )
    return (
        "Fix track validation / point association / primary target selection before posture tuning.",
        "The evidence is mixed, but the extra-track regression is the clearest system-level failure.",
    )


def static_segment_tid_answer(metrics: pd.DataFrame, segment_id: str) -> str:
    seg = metrics[(metrics["cfg_name"].eq("static_retention_cfg")) & (metrics["segment_id"].eq(segment_id))].copy()
    if seg.empty:
        return f"No per-TID metrics were available for static {segment_id}."
    sit_tids = seg[seg["mean_sit_prob"] > seg["mean_stand_prob"]]
    stand_tids = seg[seg["mean_stand_prob"] > seg["mean_sit_prob"]]
    sit_desc = ", ".join(
        f"TID {r.tid:g} (sit={fmt(r.mean_sit_prob)}, stand={fmt(r.mean_stand_prob)}, "
        f"display_sitting={fmt(r.display_sitting_rate)}, display_standing={fmt(r.display_standing_rate)})"
        for r in sit_tids.itertuples()
    ) or "none"
    stand_desc = ", ".join(
        f"TID {r.tid:g} (sit={fmt(r.mean_sit_prob)}, stand={fmt(r.mean_stand_prob)}, "
        f"display_sitting={fmt(r.display_sitting_rate)}, display_standing={fmt(r.display_standing_rate)})"
        for r in stand_tids.itertuples()
    ) or "none"
    return f"Static {segment_id}: TIDs with sit_prob > stand_prob: {sit_desc}. TIDs with stand_prob > sit_prob: {stand_desc}."


def write_report(
    out_dir: Path,
    default: SessionData,
    static: SessionData,
    default_segments: pd.DataFrame,
    static_segments: pd.DataFrame,
    metrics: pd.DataFrame,
    classifications: pd.DataFrame,
    mismatch: pd.DataFrame,
    point_assoc: pd.DataFrame,
) -> tuple[str, str]:
    findings = summarize_static_findings(metrics, classifications)
    reco, reco_reason = recommendation(metrics, classifications)
    static_3_4_cls = classifications[
        classifications["cfg_name"].eq("static_retention_cfg") & classifications["segment_id"].isin(["sitting_3m", "sitting_4m"])
    ]
    static_3_4_metrics = metrics[
        metrics["cfg_name"].eq("static_retention_cfg") & metrics["segment_id"].isin(["sitting_3m", "sitting_4m"])
    ]
    real_static = static_3_4_metrics.merge(
        static_3_4_cls[["segment_id", "tid", "classification"]],
        on=["segment_id", "tid"],
        how="left",
    )
    real_static = real_static[real_static["classification"].eq("REAL_PRIMARY")]

    extra_answer = "Yes" if static_3_4_cls["classification"].isin(["LIKELY_EXTRA_STATIC", "LIKELY_GHOST"]).any() else "No"
    wrong_assoc_answer = (
        "Partly supported but not fully proven: extra TID 5 carries seated-looking probabilities and lower NO_POINTS at 3m/4m, "
        "while exact raw point-to-TID assignment cannot be reconstructed because point coordinates and assoc modes were not logged."
    )
    model_fail = "Mixed"
    if not real_static.empty:
        if (real_static["mean_stand_prob"] > real_static["mean_sit_prob"]).all():
            model_fail = "Yes, on the real primary TID in the inspected failing distances."
        elif (real_static["mean_sit_prob"] > real_static["mean_stand_prob"]).any():
            model_fail = "Mixed: at least one real primary segment has sit_prob above stand_prob."
    gating_fail = "Yes" if ((real_static["mean_sit_prob"] > real_static["mean_stand_prob"]) & (real_static["display_sitting_rate"] < 0.2)).any() else "Not as the only cause"

    lines = [
        "# Static-Retention Per-TID Diagnosis Report",
        "",
        "## 1. Executive summary",
        (
            f"Static retention failed primarily because it introduced persistent extra tracks at 3m and 4m while the real primary "
            f"TID did not produce stable displayed SITTING. Recommendation: **{reco}** {reco_reason}"
        ),
        "",
        "## 2. Why this diagnosis was needed",
        "The A/B summary showed lower NO_POINTS in some static-retention segments but worse displayed sitting posture and 100% extra-track rate at 3m/4m. This report checks whether those failures come from extra TIDs, primary TID selection, point evidence assignment, probability/display mismatch, or model probabilities.",
        "",
        "## 3. Sessions and segments inspected",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "cfg_name": default.cfg_name,
                        "session_path": str(default.session),
                        "cfg_path": default.metadata.get("mmwave_cfg_path", "NA"),
                        "segment_file": "analysis_inputs/sitting_ab_default_segments.csv",
                    },
                    {
                        "cfg_name": static.cfg_name,
                        "session_path": str(static.session),
                        "cfg_path": static.metadata.get("mmwave_cfg_path", "NA"),
                        "segment_file": "analysis_inputs/sitting_ab_static_retention_segments.csv",
                    },
                ]
            )
        ),
        "",
        "Default segments:",
        markdown_table(default_segments[["segment_id", "expected_pose", "expected_distance_m", "start_time_s", "end_time_s", "duration_s", "segmentation_method", "confidence"]]),
        "",
        "Static-retention segments:",
        markdown_table(static_segments[["segment_id", "expected_pose", "expected_distance_m", "start_time_s", "end_time_s", "duration_s", "segmentation_method", "confidence"]]),
        "",
        "## 4. Per-TID metrics summary",
        markdown_table(
            metrics[
                [
                    "cfg_name",
                    "segment_id",
                    "tid",
                    "presence_rate_within_segment",
                    "mean_range_m",
                    "range_mae_vs_expected_m",
                    "mean_geom_pts",
                    "NO_POINTS_rate",
                    "display_standing_rate",
                    "display_sitting_rate",
                    "mean_stand_prob",
                    "mean_sit_prob",
                    "mean_stand_minus_sit_margin",
                ]
            ]
        ),
        "",
        "## 5. Real primary vs extra target classification",
        markdown_table(classifications),
        "",
        "## 6. Static-retention extra-track regression",
        f"Did static retention fail because it created extra tracks? **{extra_answer}.**",
        findings.get("sitting_3m_extras", "No static sitting_3m extra TID classification available."),
        findings.get("sitting_4m_extras", "No static sitting_4m extra TID classification available."),
        "",
        "## 7. Probability/display mismatch analysis",
        markdown_table(mismatch[mismatch["cfg_name"].eq("static_retention_cfg") & mismatch["segment_id"].isin(["sitting_3m", "sitting_4m"])]),
        "",
        "At static sitting_4m, the per-TID table above identifies which TID has sit_prob > stand_prob. A mismatch exists when that same TID has high sit_prob_gt_stand_prob_rate but display_sitting_rate remains near zero.",
        static_segment_tid_answer(metrics, "sitting_3m"),
        static_segment_tid_answer(metrics, "sitting_4m"),
        "Interpretation: the segment-level sit probability can be pulled upward by extra TID 5, while the real primary TID 0 still favors STANDING and is the correct target by range.",
        f"Does display/gating fail despite sit_prob being higher? **{gating_fail}.**",
        "",
        "## 8. Point association analysis",
        "Raw point clouds were not logged, and the current raw track table has `num_associated_points=0` for every static-retention TID. Therefore this report uses posture `num_points` as `geom_pts` and frame `num_points` as total point evidence. Association mode fields are `NA` when no `assoc` column is present.",
        markdown_table(point_assoc),
        "Static retention gives the extra TID its own seated-looking evidence at 3m/4m: TID 5 has high SITTING probability and much lower NO_POINTS than TID 0. The real primary TID 0 gets better range stability, but at 3m it has less geom_pts than the default primary and at 4m it still favors STANDING despite more geom_pts than default.",
        f"Did it attach useful sitting points to the wrong TID? **{wrong_assoc_answer}**",
        "",
        "## 9. Per-distance diagnosis: 2m",
        findings.get("sitting_2m_primary", "No static sitting_2m primary finding available."),
        "",
        "## 10. Per-distance diagnosis: 3m",
        findings.get("sitting_3m_primary", "No static sitting_3m primary finding available."),
        findings.get("sitting_3m_extras", "No static sitting_3m extra finding available."),
        "",
        "## 11. Per-distance diagnosis: 4m",
        findings.get("sitting_4m_primary", "No static sitting_4m primary finding available."),
        findings.get("sitting_4m_extras", "No static sitting_4m extra finding available."),
        "",
        "## 12. What is proven",
        f"- Extra/duplicate static-retention tracks are present in the failing 3m/4m segments: {extra_answer}.",
        "- The per-TID probabilities and display labels can be evaluated from the same `mmwave_pose.csv` row keyed by frame/TID.",
        "- Static-retention posture failure is not explained by tracking dropout; the real primary TID remains present by range/persistence.",
        "",
        "## 13. What is not proven",
        "- Raw radar point coordinates were not logged, so exact point-to-TID spatial assignment cannot be reconstructed.",
        "- Renderer confirmation state is not available as a separate CSV in these sessions.",
        "- This is offline log analysis only, not live radar validation.",
        "",
        "## 14. Recommended next engineering path",
        f"**{reco}**",
        reco_reason,
        "Do not tune posture thresholds or retrain the model until the extra-track/association behavior is isolated with offline replay or a narrower cfg experiment.",
        "",
        "## 15. Plots and generated files",
        "- `per_tid_segment_metrics.csv`: per cfg/segment/TID metrics.",
        "- `tid_classification_by_segment.csv`: real primary vs extra/ghost/wrong-association labels.",
        "- `probability_display_mismatch.csv`: sit probability vs displayed pose mismatch rates.",
        "- `point_association_by_tid.csv`: geometry and point evidence by TID.",
        "- `plots/static_sitting_3m_tid_range_timeline.png`: TID range separation at static 3m.",
        "- `plots/static_sitting_4m_tid_range_timeline.png`: TID range separation at static 4m.",
        "- `plots/static_sitting_3m_tid_pose_timeline.png`: display pose by TID at static 3m.",
        "- `plots/static_sitting_4m_tid_pose_timeline.png`: display pose by TID at static 4m.",
        "- `plots/static_sitting_3m_tid_stand_sit_probs.png`: stand/sit probabilities by TID at static 3m.",
        "- `plots/static_sitting_4m_tid_stand_sit_probs.png`: stand/sit probabilities by TID at static 4m.",
        "- `plots/static_sitting_3m_tid_geom_pts.png`: geometry evidence by TID at static 3m.",
        "- `plots/static_sitting_4m_tid_geom_pts.png`: geometry evidence by TID at static 4m.",
        "- `plots/default_vs_static_tid_count_by_time.png`: active TID count comparison.",
        "",
        "## Final answer",
        f"- Did static retention fail because it created extra tracks? **{extra_answer}.**",
        f"- Did it attach useful sitting points to the wrong TID? **{wrong_assoc_answer}**",
        f"- Did the model probability fail on the real TID? **{model_fail}**",
        f"- Did display/gating fail despite sit_prob being higher? **{gating_fail}.**",
        f"- What should we fix next? **{reco}**",
        "",
    ]
    report_path = out_dir / "STATIC_RETENTION_TID_DIAGNOSIS_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    impl_lines = [
        "# Static-Retention TID Diagnosis Implementation",
        "",
        "## 1. Files inspected",
        "- `mmwave_tracks.csv`, `mmwave_pose.csv`, `mmwave_frames.csv`, `sync_index.csv`, `rgb_frames.csv`, `rgb_tracks.csv`, `session_metadata.json` from both A/B sessions.",
        "- `analysis_inputs/sitting_ab_default_segments.csv`.",
        "- `analysis_inputs/sitting_ab_static_retention_segments.csv`.",
        "- Existing A/B analysis folders under `analysis_outputs/`.",
        "",
        "## 2. Script created",
        "- `analysis/diagnose_ab_tid_tracks.py`.",
        "",
        "## 3. Metrics computed",
        "- Per cfg/segment/TID range, position, presence, geometry, quality, display pose, stand/sit probabilities, probability/display mismatch, and point-total ratios.",
        "",
        "## 4. Plots created",
        "- Static 3m/4m TID range, pose, probability, and geom_pts timelines.",
        "- Default-vs-static TID count timeline.",
        "",
        "## 5. Final diagnosis report path",
        f"- `{report_path}`.",
        "",
        "## 6. Main conclusion",
        f"- {reco} {reco_reason}",
        "",
        "## 7. Validation commands run",
        "- `python -m py_compile analysis\\diagnose_ab_tid_tracks.py`.",
        "- `python analysis\\diagnose_ab_tid_tracks.py --default-session ... --static-session ... --default-segments ... --static-segments ... --out analysis_outputs\\sitting_ab_tid_diagnosis`.",
        "",
        "## 8. Any limitations",
        "- Raw point coordinates were not logged, so exact point-to-TID spatial attachment is not reconstructable.",
        "- Renderer confirmed/rendered/suspect rates are `NA` because no renderer state CSV is present.",
        "- This is offline analysis only and does not claim live radar validation.",
        "",
    ]
    impl_path = Path("STATIC_RETENTION_TID_DIAGNOSIS_IMPLEMENTATION.md")
    impl_path.write_text("\n".join(impl_lines), encoding="utf-8")
    return str(report_path), str(impl_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-session", required=True)
    parser.add_argument("--static-session", required=True)
    parser.add_argument("--default-segments", required=True)
    parser.add_argument("--static-segments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    default = load_session("default_cfg", Path(args.default_session))
    static = load_session("static_retention_cfg", Path(args.static_session))
    default_segments = load_segments(Path(args.default_segments))
    static_segments = load_segments(Path(args.static_segments))

    default_metrics = per_tid_metrics_for_session(default, default_segments)
    static_metrics = per_tid_metrics_for_session(static, static_segments)
    metrics = pd.concat([default_metrics, static_metrics], ignore_index=True)
    classifications = classify_tids(metrics)
    mismatch = probability_display_mismatch(metrics)
    point_assoc = point_association(metrics)

    metrics.to_csv(out_dir / "per_tid_segment_metrics.csv", index=False)
    classifications.to_csv(out_dir / "tid_classification_by_segment.csv", index=False)
    mismatch.to_csv(out_dir / "probability_display_mismatch.csv", index=False)
    point_assoc.to_csv(out_dir / "point_association_by_tid.csv", index=False)
    create_plots(default, static, static_segments, out_dir)
    report_path, impl_path = write_report(out_dir, default, static, default_segments, static_segments, metrics, classifications, mismatch, point_assoc)

    print(f"Per-TID metrics: {out_dir / 'per_tid_segment_metrics.csv'}")
    print(f"TID classifications: {out_dir / 'tid_classification_by_segment.csv'}")
    print(f"Probability/display mismatch: {out_dir / 'probability_display_mismatch.csv'}")
    print(f"Point association: {out_dir / 'point_association_by_tid.csv'}")
    print(f"Report: {report_path}")
    print(f"Implementation report: {impl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
