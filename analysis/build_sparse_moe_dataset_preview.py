#!/usr/bin/env python
"""Preview Sparse-MoE tensor construction from associated point logs.

This is an offline smoke check only. It builds in-memory sample tensors from
`mmwave_associated_points.csv` and writes shape/manifest reports. It does not
train, export, or integrate a runtime model.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posturenet_v2_common import REPO_ROOT, ensure_dir


POINT_FEATURES = [
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "height_above_ground_m",
    "point_range_m",
    "point_doppler_mps",
    "point_snr",
    "valid_mask",
]

TRACK_FEATURES = [
    "target_range_m",
    "target_z_m",
    "target_vx_mps",
    "target_vy_mps",
    "target_vz_mps",
    "speed_mps",
    "geom_pts_for_tid",
    "NO_POINTS_flag",
    "LOW_POINTS_flag",
    "OK_flag",
    "track_age_norm",
    "ui_visible_flag",
]

SPARSITY_FEATURES = [
    "range_band_code",
    "point_count_mean",
    "point_count_std",
    "NO_POINTS_rate_window",
    "LOW_POINTS_rate_window",
    "SNR_mean",
    "SNR_std",
    "valid_frame_rate",
    "target_range_mean",
    "target_range_std",
]

OLD_MODEL_FEATURES = [
    "old_stand_prob",
    "old_sit_prob",
    "old_move_prob",
    "old_lie_prob",
    "old_fall_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--out", default=str(REPO_ROOT / "analysis_outputs" / "sparse_moe_dataset_preview"))
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=200)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def numeric_value(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, default)
        if value is None or value == "":
            return default
        value = float(value)
        return default if math.isnan(value) else value
    except Exception:
        return default


def discover_point_logs(logs_root: Path) -> list[Path]:
    roots = [logs_root, REPO_ROOT / "logs", REPO_ROOT.parent / "logs"]
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("mmwave_associated_points.csv"):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return sorted(found)


def range_band_code(range_m: float) -> tuple[int, str]:
    if math.isnan(range_m):
        return -1, "UNKNOWN"
    if range_m <= 3.0:
        return 0, "NEAR"
    if range_m <= 5.0:
        return 1, "FAR"
    return 2, "EDGE"


def quality_flags(text: Any) -> tuple[float, float, float]:
    quality = str(text or "").upper()
    no_points = 1.0 if "NO_POINTS" in quality else 0.0
    low_points = 1.0 if "LOW_POINTS" in quality or "LOW QUALITY" in quality or "LOW_QUALITY" in quality else 0.0
    ok = 1.0 if "OK" in quality or "POINT_GEOMETRY" in quality or "GOOD" in quality else 0.0
    return no_points, low_points, ok


def build_window(
    tid_rows: pd.DataFrame,
    frames: list[int],
    window_frames: list[int],
    max_points: int,
    window_index: int,
) -> dict[str, Any]:
    point_tensor = np.zeros((len(window_frames), max_points, len(POINT_FEATURES)), dtype=np.float32)
    track_tensor = np.zeros((len(window_frames), len(TRACK_FEATURES)), dtype=np.float32)
    old_model_tensor = np.zeros((len(window_frames), len(OLD_MODEL_FEATURES)), dtype=np.float32)
    point_counts: list[float] = []
    snr_values: list[float] = []
    target_ranges: list[float] = []
    no_points_flags: list[float] = []
    low_points_flags: list[float] = []
    valid_frames = 0

    frame_col = find_column(tid_rows, ["frame", "mmwave_frame_num"])
    if frame_col is None:
        raise ValueError("associated point log is missing frame column")

    for frame_offset, frame in enumerate(window_frames):
        frame_rows = tid_rows[pd.to_numeric(tid_rows[frame_col], errors="coerce") == frame].copy()
        if frame_rows.empty:
            point_counts.append(0.0)
            no_points_flags.append(1.0)
            low_points_flags.append(0.0)
            continue

        valid = frame_rows.copy()
        valid_col = find_column(valid, ["is_valid_point"])
        if valid_col is not None:
            valid = valid[pd.to_numeric(valid[valid_col], errors="coerce").fillna(0) != 0]
        quality_col = find_column(valid, ["point_quality", "quality_label_for_tid"])
        if quality_col is not None:
            valid = valid[~valid[quality_col].astype(str).str.upper().eq("NO_POINTS")]

        if "point_snr" in valid.columns:
            valid = valid.assign(_snr=pd.to_numeric(valid["point_snr"], errors="coerce").fillna(-9999.0))
            valid = valid.sort_values("_snr", ascending=False)
        selected = valid.head(max_points)
        point_counts.append(float(len(selected)))
        if len(selected) > 0:
            valid_frames += 1

        for point_offset, (_, point) in enumerate(selected.iterrows()):
            values = [
                numeric_value(point, "relative_x_m"),
                numeric_value(point, "relative_y_m"),
                numeric_value(point, "relative_z_m"),
                numeric_value(point, "height_above_ground_m"),
                numeric_value(point, "point_range_m"),
                numeric_value(point, "point_doppler_mps"),
                numeric_value(point, "point_snr"),
                1.0,
            ]
            point_tensor[frame_offset, point_offset, :] = np.asarray(values, dtype=np.float32)
            snr_values.append(values[6])

        first = frame_rows.iloc[0]
        vx = numeric_value(first, "target_vx_mps")
        vy = numeric_value(first, "target_vy_mps")
        vz = numeric_value(first, "target_vz_mps")
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        no_points, low_points, ok = quality_flags(first.get("quality_label_for_tid", first.get("point_quality", "")))
        geom_pts = numeric_value(first, "geom_pts_for_tid", float(len(selected)))
        if geom_pts < 5:
            low_points = max(low_points, 1.0)
        target_range = numeric_value(first, "target_range_m", math.nan)
        if not math.isnan(target_range):
            target_ranges.append(target_range)
        no_points_flags.append(no_points if len(selected) > 0 else 1.0)
        low_points_flags.append(low_points)
        track_tensor[frame_offset, :] = np.asarray(
            [
                0.0 if math.isnan(target_range) else target_range,
                numeric_value(first, "target_z_m"),
                vx,
                vy,
                vz,
                speed,
                geom_pts,
                no_points if len(selected) > 0 else 1.0,
                low_points,
                ok,
                float(window_index + frame_offset) / max(1.0, float(len(frames))),
                0.0 if str(first.get("old_display_pose", "")).upper() in {"", "UNKNOWN", "WARMUP"} else 1.0,
            ],
            dtype=np.float32,
        )
        old_model_tensor[frame_offset, :] = np.asarray(
            [
                numeric_value(first, "old_model_stand_prob"),
                numeric_value(first, "old_model_sit_prob"),
                numeric_value(first, "old_model_move_prob"),
                numeric_value(first, "old_model_lie_prob"),
                numeric_value(first, "old_model_fall_prob"),
            ],
            dtype=np.float32,
        )

    mean_range = float(np.mean(target_ranges)) if target_ranges else math.nan
    band_code, band_name = range_band_code(mean_range)
    sparsity_tensor = np.asarray(
        [
            float(band_code),
            float(np.mean(point_counts)) if point_counts else 0.0,
            float(np.std(point_counts)) if point_counts else 0.0,
            float(np.mean(no_points_flags)) if no_points_flags else 1.0,
            float(np.mean(low_points_flags)) if low_points_flags else 0.0,
            float(np.mean(snr_values)) if snr_values else 0.0,
            float(np.std(snr_values)) if snr_values else 0.0,
            float(valid_frames) / max(1.0, float(len(window_frames))),
            0.0 if math.isnan(mean_range) else mean_range,
            float(np.std(target_ranges)) if target_ranges else 0.0,
        ],
        dtype=np.float32,
    )
    return {
        "point_tensor": point_tensor,
        "track_tensor": track_tensor,
        "sparsity_tensor": sparsity_tensor,
        "old_model_tensor": old_model_tensor,
        "range_band": band_name,
        "point_count_mean": float(sparsity_tensor[1]),
        "valid_frame_rate": float(sparsity_tensor[7]),
    }


def session_id_from_path(path: Path, df: pd.DataFrame) -> str:
    if "session_id" in df.columns and df["session_id"].notna().any():
        return str(df["session_id"].dropna().iloc[0])
    return path.parent.name


def infer_label_from_session_id(session_id: str) -> tuple[str, str, str]:
    text = session_id.lower()
    if "standing" in text:
        return "STANDING", "STANDING", "session_name_heuristic"
    if "leanback" in text or "lean_back" in text:
        return "SITTING", "SITTING_LEAN_BACK", "session_name_heuristic"
    if "upright" in text:
        return "SITTING", "SITTING_UPRIGHT", "session_name_heuristic"
    if "leanforward" in text or "lean_forward" in text:
        return "SITTING", "SITTING_LEAN_FORWARD", "session_name_heuristic"
    if "sitting" in text:
        return "SITTING", "SITTING", "session_name_heuristic"
    return "UNKNOWN", "UNKNOWN", "none"


def write_no_logs(out_dir: Path) -> None:
    ensure_dir(out_dir)
    (out_dir / "sample_window_manifest.csv").write_text(
        "window_id,session_id,tid,start_frame,end_frame,range_band,point_count_mean,valid_frame_rate,label_source\n",
        encoding="utf-8",
    )
    (out_dir / "tensor_shape_report.md").write_text(
        "# Sparse-MoE Dataset Preview\n\n"
        "No full point-cloud logs found yet. Run associated-point smoke test first.\n",
        encoding="utf-8",
    )


def write_report(
    out_dir: Path,
    point_logs: list[Path],
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Sparse-MoE Dataset Preview",
        "",
        "This report verifies tensor construction only. No model was trained.",
        "",
        "## Inputs",
        "",
        f"- Associated point logs found: {len(point_logs)}",
        f"- Window frames: {args.window_frames}",
        f"- Stride frames: {args.stride_frames}",
        f"- Max points per TID/frame: {args.max_points}",
        "",
        "## Tensor Shapes",
        "",
        f"- `point_tensor`: `[num_windows, {args.window_frames}, {args.max_points}, {len(POINT_FEATURES)}]`",
        f"- `track_tensor`: `[num_windows, {args.window_frames}, {len(TRACK_FEATURES)}]`",
        f"- `sparsity_tensor`: `[num_windows, {len(SPARSITY_FEATURES)}]`",
        f"- `old_model_tensor`: `[num_windows, {args.window_frames}, {len(OLD_MODEL_FEATURES)}]`",
        "",
        "## Feature Lists",
        "",
        "- Point features: " + ", ".join(f"`{name}`" for name in POINT_FEATURES),
        "- Track features: " + ", ".join(f"`{name}`" for name in TRACK_FEATURES),
        "- Sparsity features: " + ", ".join(f"`{name}`" for name in SPARSITY_FEATURES),
        "- Old model features: " + ", ".join(f"`{name}`" for name in OLD_MODEL_FEATURES),
        "",
        "## Result",
        "",
        f"- Windows constructed: {len(manifest)}",
    ]
    if not manifest.empty:
        lines.extend(
            [
                f"- Sessions represented: {manifest['session_id'].nunique()}",
                f"- TIDs represented: {manifest[['session_id', 'tid']].drop_duplicates().shape[0]}",
                f"- Mean point_count_mean: {manifest['point_count_mean'].mean():.3f}",
                f"- Mean valid_frame_rate: {manifest['valid_frame_rate'].mean():.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Label Note",
            "",
            "The preview does not require cleaned labels. If a session name exposes a posture, the manifest marks that as a heuristic label source; otherwise labels remain `UNKNOWN`.",
        ]
    )
    (out_dir / "tensor_shape_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    point_logs = discover_point_logs(Path(args.logs_root))
    if not point_logs:
        write_no_logs(out_dir)
        print("Full point logs found: no")
        print("Tensor preview built: no")
        return 0

    manifest_rows: list[dict[str, Any]] = []
    window_id = 0
    for path in point_logs:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            manifest_rows.append({"window_id": window_id, "session_id": path.parent.name, "notes": f"read failed: {exc}"})
            window_id += 1
            continue
        frame_col = find_column(df, ["frame", "mmwave_frame_num"])
        if frame_col is None or "tid" not in df.columns:
            continue
        session_id = session_id_from_path(path, df)
        coarse_label, subtype_label, label_source = infer_label_from_session_id(session_id)
        for tid_value in sorted(pd.to_numeric(df["tid"], errors="coerce").dropna().unique()):
            tid = int(tid_value)
            tid_rows = df[pd.to_numeric(df["tid"], errors="coerce") == tid].copy()
            frames = sorted(int(v) for v in pd.to_numeric(tid_rows[frame_col], errors="coerce").dropna().unique())
            if len(frames) < args.window_frames:
                continue
            for start_idx in range(0, len(frames) - args.window_frames + 1, args.stride_frames):
                if len(manifest_rows) >= args.max_windows:
                    break
                window_frames = frames[start_idx : start_idx + args.window_frames]
                tensors = build_window(tid_rows, frames, window_frames, args.max_points, start_idx)
                expected_point_shape = (args.window_frames, args.max_points, len(POINT_FEATURES))
                expected_track_shape = (args.window_frames, len(TRACK_FEATURES))
                expected_old_shape = (args.window_frames, len(OLD_MODEL_FEATURES))
                if tensors["point_tensor"].shape != expected_point_shape:
                    raise RuntimeError(f"point tensor shape mismatch: {tensors['point_tensor'].shape}")
                if tensors["track_tensor"].shape != expected_track_shape:
                    raise RuntimeError(f"track tensor shape mismatch: {tensors['track_tensor'].shape}")
                if tensors["old_model_tensor"].shape != expected_old_shape:
                    raise RuntimeError(f"old model tensor shape mismatch: {tensors['old_model_tensor'].shape}")
                manifest_rows.append(
                    {
                        "window_id": window_id,
                        "session_id": session_id,
                        "source_path": str(path),
                        "tid": tid,
                        "start_frame": window_frames[0],
                        "end_frame": window_frames[-1],
                        "frames_observed": len(window_frames),
                        "point_tensor_shape": str(tensors["point_tensor"].shape),
                        "track_tensor_shape": str(tensors["track_tensor"].shape),
                        "sparsity_tensor_shape": str(tensors["sparsity_tensor"].shape),
                        "old_model_tensor_shape": str(tensors["old_model_tensor"].shape),
                        "range_band": tensors["range_band"],
                        "point_count_mean": round(tensors["point_count_mean"], 4),
                        "valid_frame_rate": round(tensors["valid_frame_rate"], 4),
                        "coarse_pose": coarse_label,
                        "subtype_pose": subtype_label,
                        "label_source": label_source,
                    }
                )
                window_id += 1
            if len(manifest_rows) >= args.max_windows:
                break
        if len(manifest_rows) >= args.max_windows:
            break

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "sample_window_manifest.csv", index=False)
    write_report(out_dir, point_logs, manifest, args)
    print("Full point logs found: yes")
    print(f"Tensor preview built: {'yes' if len(manifest) else 'no'}")
    print(f"Windows constructed: {len(manifest)}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
