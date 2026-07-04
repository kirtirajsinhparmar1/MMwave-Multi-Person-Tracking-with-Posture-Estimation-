#!/usr/bin/env python
"""Build an offline second-stage posture filter dataset from labeled sessions."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WINDOWS_S = [0.0, 0.5, 1.0, 2.0, 3.0]
LABEL_COLS = {"expected_pose", "expected_subpose", "expected_distance_m", "session_id", "segment_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def normalize_time(df: pd.DataFrame) -> pd.Series:
    for col in ("time", "timestamp_s", "time_s", "elapsed_s"):
        if col in df.columns:
            vals = numeric(df[col])
            if vals.notna().any():
                if vals.max(skipna=True) > 1e6:
                    vals = vals - vals.min(skipna=True)
                return vals.astype(float)
            dt = pd.to_datetime(df[col], errors="coerce", utc=True)
            if dt.notna().any():
                return (dt - dt.min()).dt.total_seconds().astype(float)
    for col in ("host_monotonic_ns", "monotonic_ns"):
        if col in df.columns:
            vals = numeric(df[col])
            if vals.notna().any():
                return ((vals - vals.min(skipna=True)) / 1e9).astype(float)
    for col in ("host_wall_time_iso", "timestamp", "wall_time"):
        if col in df.columns:
            vals = pd.to_datetime(df[col], errors="coerce", utc=True)
            if vals.notna().any():
                return (vals - vals.min()).dt.total_seconds().astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def pose_label(value: Any) -> str:
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
    return text if text in {"STANDING", "SITTING", "MOVING", "LYING", "FALLING", "UNKNOWN"} else "OTHER"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def md_table(data: pd.DataFrame | pd.Series) -> str:
    if isinstance(data, pd.Series):
        df = data.rename("count").reset_index()
    else:
        df = data.copy()
    if df.empty:
        return "No rows."
    cols = [str(c) for c in df.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def find_pose_file(session: Path) -> Path:
    local = Path("logs") / session.name / "pose_predictions_ui.csv"
    for path in (local, session / "pose_predictions_ui.csv", session / "mmwave_pose.csv"):
        if path.exists() and path.stat().st_size > 0:
            return path
    return session / "mmwave_pose.csv"


def normalize_pose(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out["timestamp_s"] = normalize_time(df)
    for target, names in {
        "frame": ["frame", "mmwave_frame_num", "frame_num"],
        "tid": ["tid", "target_id", "track_id"],
        "display_pose": ["final_display_pose", "displayed_label", "display_pose", "final_label", "pose"],
        "raw_pose": ["ml_label", "raw_pose", "postureml", "raw_label"],
        "quality": ["quality", "quality_flag", "geom_quality"],
        "geom_pts": ["geom_pts", "num_points", "num_associated_points"],
        "range_m": ["range_m", "range", "distance_m"],
        "z_m": ["z_m", "z"],
        "speed_mps": ["speed_mps", "horizontal_speed", "speed"],
        "stand_prob": ["stand_prob", "prob_standing", "prob_STANDING"],
        "sit_prob": ["sit_prob", "prob_sitting", "prob_SITTING"],
        "move_prob": ["move_prob", "prob_moving", "prob_MOVING"],
        "lie_prob": ["lie_prob", "lying_prob", "prob_lying", "prob_LYING"],
        "fall_prob": ["fall_prob", "prob_falling", "prob_FALLING"],
        "relative_gate_trigger": ["sitting_relative_gate_triggered", "sitting_relative_gate_trigger", "relative_gate_trigger"],
        "relative_gate_passed": ["sitting_relative_gate_passed", "relative_gate_passed"],
        "standing_veto_block": ["sitting_relative_gate_blocked_standing_veto", "standing_veto_block"],
        "ui_visible": ["ui_visible", "display_confirmed", "is_rendered"],
    }.items():
        col = first_col(df, names)
        out[target] = df[col] if col else np.nan
    for col in ["frame", "tid", "geom_pts", "range_m", "z_m", "speed_mps", "stand_prob", "sit_prob", "move_prob", "lie_prob", "fall_prob"]:
        out[col] = numeric(out[col])
    for col in ("relative_gate_trigger", "relative_gate_passed", "standing_veto_block", "ui_visible"):
        out[col] = out[col].map(boolish)
    out["display_pose"] = out["display_pose"].map(pose_label)
    out["raw_pose"] = out["raw_pose"].map(pose_label)
    return out.dropna(subset=["timestamp_s"])


def boolish(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "passed", "visible"}:
        return 1.0
    if text in {"0", "false", "no", "n", "blocked", "hidden"}:
        return 0.0
    try:
        return 1.0 if float(text) != 0.0 else 0.0
    except ValueError:
        return np.nan


def normalize_tracks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out["timestamp_s"] = normalize_time(df)
    for target, names in {
        "frame": ["frame", "mmwave_frame_num", "frame_num"],
        "tid": ["tid", "target_id", "track_id"],
        "x_m": ["x_m", "x"],
        "y_m": ["y_m", "y"],
        "z_m": ["z_m", "z"],
        "vx_mps": ["vx_mps", "vx"],
        "vy_mps": ["vy_mps", "vy"],
        "range_m": ["range_m", "range", "distance_m"],
        "geom_pts": ["geom_pts", "num_associated_points", "num_points"],
    }.items():
        col = first_col(df, names)
        out[target] = df[col] if col else np.nan
    for col in out.columns:
        out[col] = numeric(out[col])
    if out["range_m"].isna().all() and out["x_m"].notna().any() and out["y_m"].notna().any():
        out["range_m"] = np.sqrt(out["x_m"] ** 2 + out["y_m"] ** 2)
    out["speed_mps_track"] = np.sqrt(out["vx_mps"] ** 2 + out["vy_mps"] ** 2)
    return out.dropna(subset=["timestamp_s"])


def merge_pose_tracks(pose: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    if pose.empty and tracks.empty:
        return pd.DataFrame()
    if pose.empty:
        out = tracks.copy()
        out["display_pose"] = "UNKNOWN"
        return out
    out = pose.copy()
    if not tracks.empty and {"frame", "tid"}.issubset(out.columns) and {"frame", "tid"}.issubset(tracks.columns):
        extra = tracks[["frame", "tid", "range_m", "z_m", "geom_pts", "speed_mps_track"]].dropna(subset=["frame", "tid"]).drop_duplicates(["frame", "tid"])
        out = out.merge(extra, on=["frame", "tid"], how="left", suffixes=("", "_track"))
        for col in ("range_m", "z_m", "geom_pts"):
            alt = f"{col}_track"
            if alt in out.columns:
                out[col] = out[col].fillna(out[alt])
        if "speed_mps_track" in out.columns:
            out["speed_mps"] = out["speed_mps"].fillna(out["speed_mps_track"])
        out = out.drop(columns=[c for c in out.columns if c.endswith("_track")], errors="ignore")
    return out


def cfg_family(row: pd.Series) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("session_id", "cfg_path", "notes")).lower()
    if "static" in text:
        return "static_retention"
    if "relative_gate" in text or "refined" in text:
        return "refined_gate_default_cfg"
    if "default" in text:
        return "default_cfg"
    return "unknown_cfg"


def rate(series: pd.Series, value: str) -> float:
    if series.empty:
        return np.nan
    return float((series.astype(str) == value).mean())


def switch_count(series: pd.Series) -> int:
    vals = [v for v in series.astype(str).tolist() if v and v != "nan"]
    return int(sum(1 for a, b in zip(vals, vals[1:]) if a != b))


def dominant(series: pd.Series) -> str:
    vals = [str(v) for v in series.dropna().tolist() if str(v)]
    return Counter(vals).most_common(1)[0][0] if vals else "UNKNOWN"


def summarize_window(win: pd.DataFrame, all_tracks: pd.DataFrame, seg: pd.Series, registry_row: pd.Series, window_s: float) -> dict[str, Any]:
    dist = float(seg["expected_distance_m"])
    quality = win.get("quality", pd.Series(dtype=object)).astype(str).str.upper()
    display = win.get("display_pose", pd.Series(dtype=object)).map(pose_label)
    track_win = all_tracks[(all_tracks["timestamp_s"] >= float(win["timestamp_s"].min())) & (all_tracks["timestamp_s"] <= float(win["timestamp_s"].max()))] if not all_tracks.empty and not win.empty else pd.DataFrame()
    duration = max(float(win["timestamp_s"].max() - win["timestamp_s"].min()) if len(win) > 1 else 0.055, 0.055)
    expected_frames = max(1.0, duration / 0.055)
    row = {
        "range_m_mean": win["range_m"].mean() if "range_m" in win else np.nan,
        "range_m_std": win["range_m"].std() if "range_m" in win else np.nan,
        "range_error_m_mean": (win["range_m"] - dist).abs().mean() if "range_m" in win else np.nan,
        "z_mean": win["z_m"].mean() if "z_m" in win else np.nan,
        "z_std": win["z_m"].std() if "z_m" in win else np.nan,
        "speed_mean": win["speed_mps"].mean() if "speed_mps" in win else np.nan,
        "speed_std": win["speed_mps"].std() if "speed_mps" in win else np.nan,
        "stand_prob_mean": win["stand_prob"].mean() if "stand_prob" in win else np.nan,
        "sit_prob_mean": win["sit_prob"].mean() if "sit_prob" in win else np.nan,
        "move_prob_mean_if_available": win["move_prob"].mean() if "move_prob" in win else np.nan,
        "lie_prob_mean_if_available": win["lie_prob"].mean() if "lie_prob" in win else np.nan,
        "fall_prob_mean_if_available": win["fall_prob"].mean() if "fall_prob" in win else np.nan,
        "stand_prob_std": win["stand_prob"].std() if "stand_prob" in win else np.nan,
        "sit_prob_std": win["sit_prob"].std() if "sit_prob" in win else np.nan,
        "sit_minus_stand_mean": (win["sit_prob"] - win["stand_prob"]).mean() if {"sit_prob", "stand_prob"}.issubset(win.columns) else np.nan,
        "sit_minus_stand_std": (win["sit_prob"] - win["stand_prob"]).std() if {"sit_prob", "stand_prob"}.issubset(win.columns) else np.nan,
        "display_standing_rate": rate(display, "STANDING"),
        "display_sitting_rate": rate(display, "SITTING"),
        "display_moving_rate": rate(display, "MOVING"),
        "display_unknown_rate": rate(display, "UNKNOWN"),
        "NO_POINTS_rate": float(quality.str.contains("NO_POINTS", na=False).mean()) if len(quality) else np.nan,
        "LOW_POINTS_rate": float(quality.str.contains("LOW_POINTS", na=False).mean()) if len(quality) else np.nan,
        "OK_rate": float(quality.str.contains("OK", na=False).mean()) if len(quality) else np.nan,
        "geom_pts_mean": win["geom_pts"].mean() if "geom_pts" in win else np.nan,
        "geom_pts_std": win["geom_pts"].std() if "geom_pts" in win else np.nan,
        "geom_pts_ge_1_rate": float((win["geom_pts"] >= 1).mean()) if "geom_pts" in win else np.nan,
        "geom_pts_ge_3_rate": float((win["geom_pts"] >= 3).mean()) if "geom_pts" in win else np.nan,
        "pose_switch_count": switch_count(display),
        "tracking_presence_rate": min(1.0, len(track_win) / expected_frames) if not track_win.empty else np.nan,
        "pose_presence_rate": min(1.0, len(win) / expected_frames),
        "ui_visible_rate_if_available": win["ui_visible"].mean() if "ui_visible" in win else np.nan,
        "relative_gate_trigger_rate_if_available": win["relative_gate_trigger"].mean() if "relative_gate_trigger" in win else np.nan,
        "relative_gate_passed_rate_if_available": win["relative_gate_passed"].mean() if "relative_gate_passed" in win else np.nan,
        "standing_veto_block_rate_if_available": win["standing_veto_block"].mean() if "standing_veto_block" in win else np.nan,
        "distance_m": dist,
        "expected_subpose": seg["expected_subpose"],
        "cfg_family": cfg_family(registry_row),
        "session_id": seg["session_id"],
        "segment_id": seg["segment_id"],
        "expected_pose": seg["expected_pose"],
        "expected_distance_m": dist,
        "window_s": window_s,
        "window_start_time_s": float(win["timestamp_s"].min()) if not win.empty else np.nan,
        "window_end_time_s": float(win["timestamp_s"].max()) if not win.empty else np.nan,
        "tid": dominant(win["tid"]) if "tid" in win else "UNKNOWN",
        "baseline_display_pose": dominant(display),
        "baseline_raw_prob_pose": raw_prob_pose(win),
    }
    return row


def raw_prob_pose(win: pd.DataFrame) -> str:
    means = {
        "STANDING": win["stand_prob"].mean() if "stand_prob" in win else np.nan,
        "SITTING": win["sit_prob"].mean() if "sit_prob" in win else np.nan,
        "LYING": win["lie_prob"].mean() if "lie_prob" in win else np.nan,
        "FALLING": win["fall_prob"].mean() if "fall_prob" in win else np.nan,
    }
    finite = {k: v for k, v in means.items() if pd.notna(v)}
    return max(finite, key=finite.get) if finite else "UNKNOWN"


def build_examples_for_segment(data: pd.DataFrame, tracks: pd.DataFrame, seg: pd.Series, registry_row: pd.Series) -> list[dict[str, Any]]:
    start = float(seg["start_time_s"])
    end = float(seg["end_time_s"])
    seg_data = data[(data["timestamp_s"] >= start) & (data["timestamp_s"] <= end)].sort_values("timestamp_s")
    if seg_data.empty:
        return []
    counts = seg_data["tid"].value_counts(dropna=True) if "tid" in seg_data else pd.Series(dtype=int)
    dominant_tid = counts.index[0] if len(counts) else np.nan
    if pd.notna(dominant_tid):
        seg_data = seg_data[seg_data["tid"] == dominant_tid]
    examples: list[dict[str, Any]] = []
    times = seg_data["timestamp_s"].dropna().tolist()
    for window_s in WINDOWS_S:
        if window_s == 0.0:
            for _, row in seg_data.iterrows():
                examples.append(summarize_window(pd.DataFrame([row]), tracks, seg, registry_row, window_s))
            continue
        stride = max(0.25, min(window_s / 2.0, 1.0))
        cursor = start
        while cursor + window_s <= end + 1e-6:
            win = seg_data[(seg_data["timestamp_s"] >= cursor) & (seg_data["timestamp_s"] <= cursor + window_s)]
            if len(win) >= 2 or (times and window_s <= 0.5 and len(win) >= 1):
                examples.append(summarize_window(win, tracks, seg, registry_row, window_s))
            cursor += stride
    return examples


def write_dataset_summary(examples: pd.DataFrame, out_dir: Path, registry: pd.DataFrame) -> None:
    lines = [
        "# Posture Filter Dataset Summary",
        "",
        f"number of sessions: {registry['session_id'].nunique()}",
        f"number of segments: {examples[['session_id', 'segment_id']].drop_duplicates().shape[0] if not examples.empty else 0}",
        f"number of examples: {len(examples)}",
        "",
        "## Examples Per Class",
        "",
        md_table(examples["expected_pose"].value_counts(dropna=False)) if not examples.empty else "No examples.",
        "",
        "## Examples Per Subpose",
        "",
        md_table(examples["expected_subpose"].value_counts(dropna=False)) if not examples.empty else "No examples.",
        "",
        "## Examples Per Distance",
        "",
        md_table(examples["expected_distance_m"].value_counts(dropna=False).sort_index()) if not examples.empty else "No examples.",
        "",
        "## Examples Per Cfg Family",
        "",
        md_table(examples["cfg_family"].value_counts(dropna=False)) if not examples.empty else "No examples.",
        "",
        "## Warnings",
        "",
    ]
    if examples.empty:
        lines.append("- No examples were generated.")
    else:
        counts = examples["expected_pose"].value_counts()
        if len(counts) > 1 and counts.max() / max(counts.min(), 1) > 2.0:
            lines.append(f"- Class imbalance warning: largest/smallest class ratio is {counts.max() / max(counts.min(), 1):.2f}.")
        sub_counts = examples["expected_subpose"].value_counts()
        if len(sub_counts) > 1 and sub_counts.max() / max(sub_counts.min(), 1) > 2.0:
            lines.append(f"- Subpose imbalance warning: largest/smallest subpose ratio is {sub_counts.max() / max(sub_counts.min(), 1):.2f}.")
        if examples["session_id"].nunique() < 3:
            lines.append("- Few sessions available; grouped validation has high uncertainty.")
        if not lines[-1].startswith("-"):
            lines.append("- No major imbalance warnings beyond the protocol imbalance inherent in available sessions.")
    (out_dir / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_registry_analysis_summary(analysis_root: Path, examples: pd.DataFrame) -> None:
    analysis_root.mkdir(parents=True, exist_ok=True)
    lines = ["# Registry Analysis Summary", "", "Corrected analyses were run using the registry segment files.", ""]
    if examples.empty:
        lines.append("No dataset examples were available to summarize.")
    else:
        grouped = examples.groupby(["session_id", "expected_subpose", "expected_distance_m"], dropna=False)
        rows = []
        for key, group in grouped:
            session_id, subpose, dist = key
            expected = group["expected_pose"].iloc[0]
            display = group["baseline_display_pose"].map(pose_label)
            rows.append(
                {
                    "session_id": session_id,
                    "expected_subpose": subpose,
                    "distance_m": dist,
                    "examples": len(group),
                    "baseline_display_accuracy": float((display == expected).mean()),
                    "tracking_presence_rate": group["tracking_presence_rate"].mean(),
                    "pose_presence_rate": group["pose_presence_rate"].mean(),
                    "range_error_m_mean": group["range_error_m_mean"].mean(),
                }
            )
        summary = pd.DataFrame(rows)
        summary.to_csv(analysis_root / "registry_dataset_summary_metrics.csv", index=False)
        lines.append(md_table(summary))
    (analysis_root / "REGISTRY_ANALYSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_examples: list[dict[str, Any]] = []
    for _, reg in registry.iterrows():
        session = Path(str(reg["session_path"]))
        segments = read_csv(Path(str(reg["segment_file"])))
        if segments.empty:
            continue
        segments = segments.copy()
        segments["start_time_s"] = numeric(segments["start_time_s"])
        segments["end_time_s"] = numeric(segments["end_time_s"])
        pose = normalize_pose(read_csv(find_pose_file(session)))
        tracks = normalize_tracks(read_csv(session / "mmwave_tracks.csv"))
        data = merge_pose_tracks(pose, tracks)
        for _, seg in segments.dropna(subset=["start_time_s", "end_time_s"]).iterrows():
            all_examples.extend(build_examples_for_segment(data, tracks, seg, reg))
    examples = pd.DataFrame(all_examples)
    examples.to_csv(out_dir / "posture_filter_examples.csv", index=False)
    write_dataset_summary(examples, out_dir, registry)
    write_registry_analysis_summary(Path(args.analysis_root), examples)
    print(f"Wrote examples: {out_dir / 'posture_filter_examples.csv'} ({len(examples)} rows)")
    print(f"Wrote dataset summary: {out_dir / 'dataset_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
