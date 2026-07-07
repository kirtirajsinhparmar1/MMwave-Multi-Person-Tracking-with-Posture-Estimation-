"""Build bounded RadarPostureNet-v2 datasets from cleaned posture sessions.

The builder intentionally separates labels from runtime posture predictions:
protocol segment labels are targets; old model probabilities and displayed
labels are only input features or baselines.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clean_and_label_posture_sessions import load_session_tables, rows_in_segment
from posturenet_v2_common import (
    REPO_ROOT,
    cfg_family,
    ensure_dir,
    normalized_label,
    position_code,
    read_json,
    safe_div,
    to_float,
    to_int,
    write_csv,
)


POINT_FILE_NAMES = {
    "points.csv",
    "raw_points.csv",
    "mmwave_points.csv",
    "mmwave_point_cloud.csv",
    "point_cloud.csv",
    "pointcloud.csv",
}
FEATURE_FILE_NAMES = {"features_22.csv", "features_176.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cleaned-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = find_column(df, candidates)
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def mean_std(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 0.0
    return float(clean.mean()), float(clean.std(ddof=0))


def rate(values: pd.Series, wanted: str) -> float:
    labels = values.map(normalized_label)
    return safe_div(float((labels == wanted).sum()), float(len(labels))) if len(labels) else 0.0


def switch_count(labels: pd.Series) -> int:
    normalized = labels.map(normalized_label)
    normalized = normalized[normalized != "UNKNOWN"].reset_index(drop=True)
    if len(normalized) < 2:
        return 0
    return int((normalized != normalized.shift()).sum() - 1)


def quality_rates(pose: pd.DataFrame) -> tuple[float, float, float]:
    quality_col = find_column(pose, ["quality_flag", "quality", "geom_quality"])
    if quality_col is None or pose.empty:
        return 0.0, 0.0, 0.0
    values = pose[quality_col].astype(str).str.upper()
    no_points = safe_div(float(values.str.contains("NO_POINTS").sum()), float(len(values)))
    low_points = safe_div(float(values.str.contains("LOW_POINTS|LOW QUALITY|LOW_QUALITY").sum()), float(len(values)))
    ok = safe_div(float(values.str.contains("OK|POINT_GEOMETRY|GOOD").sum()), float(len(values)))
    point_counts = numeric_series(pose, ["num_points", "geom_pts"])
    if point_counts.notna().any():
        low_points = max(low_points, safe_div(float((point_counts < 5).sum()), float(point_counts.notna().sum())))
    return no_points, low_points, ok


def pointcloud_audit_for_session(session_id: str, session_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "session_path": str(session_path),
        "metadata_mmwave_log_points": "",
        "point_files": "",
        "feature_files": "",
        "has_raw_point_rows": False,
        "has_xyz_snr_doppler": False,
        "has_track_indexes": False,
        "has_associated_tid": False,
        "has_features_176": False,
        "can_reconstruct_target_centered_tensors": False,
        "notes": "",
    }
    if not session_path.exists():
        row["notes"] = "session folder missing"
        return row

    metadata = read_json(session_path / "session_metadata.json")
    if "mmwave_log_points" in metadata:
        row["metadata_mmwave_log_points"] = str(metadata.get("mmwave_log_points"))

    point_files: list[str] = []
    feature_files: list[str] = []
    notes: list[str] = []
    for path in sorted(session_path.glob("*.csv")):
        lower_name = path.name.lower()
        if lower_name in FEATURE_FILE_NAMES:
            feature_files.append(path.name)
            if lower_name == "features_176.csv":
                row["has_features_176"] = True
        if lower_name.startswith("rgb_"):
            continue
        if lower_name not in POINT_FILE_NAMES and "point" not in lower_name:
            continue
        point_files.append(path.name)
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception as exc:
            notes.append(f"{path.name}: could not read header: {exc}")
            continue
        lower_header = {str(col).lower() for col in header}
        has_xyz = (
            {"x", "y", "z"}.issubset(lower_header)
            or {"x_m", "y_m", "z_m"}.issubset(lower_header)
            or {"point_x", "point_y", "point_z"}.issubset(lower_header)
        )
        has_signal = bool(lower_header & {"snr", "snr_db", "doppler", "doppler_mps", "velocity", "v"})
        has_track_index = bool(lower_header & {"trackindex", "track_index", "target_index", "targetindex"})
        has_tid = bool(lower_header & {"tid", "track_id", "target_id", "assigned_tid"})
        row["has_raw_point_rows"] = True
        row["has_xyz_snr_doppler"] = bool(row["has_xyz_snr_doppler"] or (has_xyz and has_signal))
        row["has_track_indexes"] = bool(row["has_track_indexes"] or has_track_index)
        row["has_associated_tid"] = bool(row["has_associated_tid"] or has_tid)

    row["point_files"] = ";".join(point_files)
    row["feature_files"] = ";".join(feature_files)
    row["can_reconstruct_target_centered_tensors"] = bool(
        row["has_raw_point_rows"]
        and row["has_xyz_snr_doppler"]
        and (row["has_track_indexes"] or row["has_associated_tid"])
    )
    if not row["can_reconstruct_target_centered_tensors"]:
        notes.append("missing per-point xyz/signal rows with point-to-target association")
    row["notes"] = "; ".join(notes)
    return row


def expected_frame_count(frames: pd.DataFrame, start: float, end: float) -> int:
    frame_rows = rows_in_segment(frames, start, end)
    if not frame_rows.empty:
        return max(1, len(frame_rows))
    return max(1, int((end - start) * 18.0))


def window_presence(df: pd.DataFrame, start: float, end: float, tid: int | None, expected_frames: int) -> tuple[pd.DataFrame, float]:
    seg = rows_in_segment(df, start, end)
    if tid is not None and "tid" in seg.columns:
        seg = seg[pd.to_numeric(seg["tid"], errors="coerce") == tid].copy()
    if seg.empty:
        return seg, 0.0
    if "mmwave_frame_num" in seg.columns:
        frames = pd.to_numeric(seg["mmwave_frame_num"], errors="coerce").dropna().nunique()
    else:
        frames = len(seg)
    return seg, safe_div(float(frames), float(expected_frames))


def reliability_label(
    tracking_presence: float,
    pose_presence: float,
    no_points_rate: float,
    low_points_rate: float,
    segment_quality: str,
) -> str:
    if tracking_presence < 0.3 or pose_presence < 0.3 or str(segment_quality).upper() == "LOW":
        return "LOW_VISIBILITY"
    if tracking_presence < 0.7 or pose_presence < 0.7 or no_points_rate > 0.1 or low_points_rate > 0.3:
        return "DEGRADED"
    return "OK"


def build_window_features(
    row: pd.Series,
    session_row: pd.Series,
    tables: dict[str, pd.DataFrame],
    window_size_s: float,
    window_start_s: float,
    window_end_s: float,
    window_index: int,
) -> dict[str, Any] | None:
    tid = to_int(row.get("assigned_tid"))
    if tid is None:
        return None

    expected_frames = expected_frame_count(tables["frames"], window_start_s, window_end_s)
    pose, pose_presence = window_presence(tables["pose"], window_start_s, window_end_s, tid, expected_frames)
    tracks, tracking_presence = window_presence(tables["tracks"], window_start_s, window_end_s, tid, expected_frames)
    if pose.empty and tracks.empty:
        return None

    stand = numeric_series(pose, ["prob_standing", "stand_prob", "standing_prob", "prob_STANDING"])
    sit = numeric_series(pose, ["prob_sitting", "sit_prob", "sitting_prob", "prob_SITTING"])
    move = numeric_series(pose, ["prob_moving", "move_prob", "moving_prob", "prob_MOVING"])
    lie = numeric_series(pose, ["prob_lying", "lie_prob", "lying_prob", "prob_LYING"])
    fall = numeric_series(pose, ["prob_falling", "fall_prob", "falling_prob", "prob_FALLING"])
    stand_mean, stand_std = mean_std(stand)
    sit_mean, sit_std = mean_std(sit)
    sit_minus_stand = sit.reset_index(drop=True) - stand.reset_index(drop=True)
    diff_mean, diff_std = mean_std(sit_minus_stand)

    range_mean, range_std = mean_std(numeric_series(tracks, ["range_m"]))
    z_mean, z_std = mean_std(numeric_series(tracks, ["target_z_m", "z_m"]))
    speed_series = numeric_series(pose, ["speed_mps"])
    if speed_series.empty or not speed_series.notna().any():
        speed_series = numeric_series(tracks, ["speed_mps_calc", "speed_mps"])
    speed_mean, speed_std = mean_std(speed_series)
    geom_points = numeric_series(pose, ["num_points", "geom_pts"])
    if geom_points.empty or not geom_points.notna().any():
        geom_points = numeric_series(tracks, ["num_associated_points"])
    geom_mean, geom_std = mean_std(geom_points)
    no_points, low_points, ok_rate = quality_rates(pose)

    display_col = find_column(pose, ["final_label", "final_display_pose", "displayed_label", "ml_label"])
    display_labels = pose[display_col] if display_col else pd.Series(dtype=object)
    display_standing = rate(display_labels, "STANDING")
    display_sitting = rate(display_labels, "SITTING")
    display_moving = rate(display_labels, "MOVING")
    display_unknown = rate(display_labels, "UNKNOWN") if len(display_labels) else 1.0
    switches = switch_count(display_labels)

    all_track_rows = rows_in_segment(tables["tracks"], window_start_s, window_end_s)
    tid_counts = Counter()
    if not all_track_rows.empty and "tid" in all_track_rows.columns:
        tid_counts = Counter(pd.to_numeric(all_track_rows["tid"], errors="coerce").dropna().astype(int).tolist())
    significant_tids = [count for count in tid_counts.values() if count >= max(3, 0.1 * sum(tid_counts.values()))]
    tid_switch_count = max(0, len(significant_tids) - 1)
    ui_visible_rate = 1.0 - display_unknown if len(display_labels) else 0.0
    disappearance_rate = max(0.0, 1.0 - min(tracking_presence, pose_presence))
    rel_label = reliability_label(
        tracking_presence,
        pose_presence,
        no_points,
        low_points,
        str(row.get("segment_quality") or ""),
    )

    cfg = cfg_family(str(session_row.get("cfg_path") or ""))
    return {
        "window_id": f"{row['session_id']}::{row['segment_id']}::{row['person_slot']}::{window_size_s:.0f}s::{window_index}",
        "session_id": row["session_id"],
        "segment_id": row["segment_id"],
        "person_slot": row["person_slot"],
        "assigned_tid": tid,
        "window_size_s": window_size_s,
        "window_start_s": round(window_start_s, 3),
        "window_end_s": round(window_end_s, 3),
        "expected_pose": row["expected_pose"],
        "expected_subpose": row["expected_subpose"],
        "expected_distance_m": to_float(row.get("expected_distance_m"), 0.0),
        "expected_position": row["expected_position"],
        "people_count": to_int(session_row.get("people_count"), 1) or 1,
        "label_confidence": row.get("label_confidence", ""),
        "assignment_confidence": row.get("assignment_confidence", ""),
        "visibility_reliability_label": rel_label,
        "stand_prob_mean": stand_mean,
        "sit_prob_mean": sit_mean,
        "move_prob_mean_if_available": mean_std(move)[0],
        "lie_prob_mean_if_available": mean_std(lie)[0],
        "fall_prob_mean_if_available": mean_std(fall)[0],
        "stand_prob_std": stand_std,
        "sit_prob_std": sit_std,
        "sit_minus_stand_mean": diff_mean,
        "sit_minus_stand_std": diff_std,
        "range_m_mean": range_mean,
        "range_m_std": range_std,
        "target_z_mean": z_mean,
        "target_z_std": z_std,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "geom_pts_mean": geom_mean,
        "geom_pts_std": geom_std,
        "NO_POINTS_rate": no_points,
        "LOW_POINTS_rate": low_points,
        "OK_rate": ok_rate,
        "display_standing_rate": display_standing,
        "display_sitting_rate": display_sitting,
        "display_moving_rate": display_moving,
        "display_unknown_rate": display_unknown,
        "pose_switch_count": switches,
        "tracking_presence_rate": tracking_presence,
        "pose_presence_rate": pose_presence,
        "ui_visible_rate": ui_visible_rate,
        "disappearance_rate": disappearance_rate,
        "tid_switch_count": tid_switch_count,
        "expected_position_encoded": position_code(str(row["expected_position"])),
        "cfg_family": cfg,
        "cfg_family_default": 1 if cfg == "default" else 0,
        "cfg_family_static_retention": 1 if cfg == "static_retention" else 0,
        "old_model_probs_if_available": 1 if stand.notna().any() or sit.notna().any() else 0,
    }


def generate_windows(
    quality: pd.DataFrame,
    registry: pd.DataFrame,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    registry_by_session = {str(row["session_id"]): row for _, row in registry.iterrows()}

    for session_id, session_quality in quality.groupby("session_id", sort=False):
        session_row = registry_by_session[str(session_id)]
        session_path = Path(str(session_row.get("session_path") or ""))
        tables = load_session_tables(session_path) if session_path.exists() else {
            "pose": pd.DataFrame(),
            "tracks": pd.DataFrame(),
            "frames": pd.DataFrame(),
        }
        session_windows = 0
        skipped_no_tid = 0
        skipped_no_rows = 0
        for _, qrow in session_quality.iterrows():
            if to_int(qrow.get("assigned_tid")) is None:
                skipped_no_tid += 1
                continue
            start = to_float(qrow.get("start_time_s"), 0.0)
            end = to_float(qrow.get("end_time_s"), 0.0)
            if end <= start:
                skipped_no_rows += 1
                continue
            for size in [1.0, 2.0, 3.0]:
                stride = size / 2.0
                cursor = start
                index = 0
                while cursor + size <= end + 1e-6:
                    features = build_window_features(qrow, session_row, tables, size, cursor, cursor + size, index)
                    if features is None:
                        skipped_no_rows += 1
                    else:
                        windows.append(features)
                        session_windows += 1
                    cursor += stride
                    index += 1
        manifest.append(
            {
                "session_id": session_id,
                "session_path": str(session_path),
                "person_instances": len(session_quality),
                "segments": session_quality["segment_id"].nunique(),
                "windows": session_windows,
                "skipped_person_instances_no_tid": skipped_no_tid,
                "skipped_empty_windows": skipped_no_rows,
            }
        )
    return windows, manifest


def write_dataset_report(
    out: Path,
    registry: pd.DataFrame,
    quality: pd.DataFrame,
    windows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> None:
    full_possible = any(bool(row["can_reconstruct_target_centered_tensors"]) for row in point_rows)
    lite_available = len(windows) > 0
    counts_by_size = Counter(str(row["window_size_s"]) for row in windows)
    counts_by_label = Counter(str(row["expected_pose"]) for row in windows)
    with (out / "DATASET_BUILD_REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 Dataset Build Report\n\n")
        handle.write("Ground truth labels come from user-provided segment protocols and filled segment files. Displayed posture was not used as a target.\n\n")
        handle.write(f"Sessions in registry: {len(registry)}\n\n")
        handle.write(f"Person-instances in cleaned segments: {len(quality)}\n\n")
        handle.write(f"Lite windows written: {len(windows)}\n\n")
        handle.write(f"Full point-cloud tensors possible: {'yes' if full_possible else 'no'}\n\n")
        handle.write(f"Lite dataset available: {'yes' if lite_available else 'no'}\n\n")
        handle.write("## Windows By Size\n\n")
        for size, count in sorted(counts_by_size.items()):
            handle.write(f"- {size}s: {count}\n")
        handle.write("\n## Windows By Coarse Label\n\n")
        for label, count in sorted(counts_by_label.items()):
            handle.write(f"- {label}: {count}\n")
        handle.write("\n## Session Manifest\n\n")
        handle.write("| session_id | person_instances | segments | windows | skipped_no_tid | skipped_empty_windows |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in manifest:
            handle.write(
                f"| {row['session_id']} | {row['person_instances']} | {row['segments']} | "
                f"{row['windows']} | {row['skipped_person_instances_no_tid']} | {row['skipped_empty_windows']} |\n"
            )
        handle.write("\n## Data Modality Decision\n\n")
        if full_possible:
            handle.write(
                "At least one session contains point rows with xyz/signal data and point-to-target association. "
                "A full tensor build path can be enabled for those sessions, but mixed-modality training should still report missing sessions explicitly.\n"
            )
        else:
            handle.write(
                "No session contained the required per-point xyz/snr-or-doppler rows with point-to-TID association. "
                "The bounded pass therefore builds RadarPostureNet-v2-lite only. Full RadarPostureNet-v2 requires logging associated point-cloud rows per frame and TID.\n"
            )


def main() -> int:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    cleaned_root = Path(args.cleaned_root)
    out = ensure_dir(Path(args.out))
    quality_path = cleaned_root / "segment_quality.csv"
    if not quality_path.exists():
        raise FileNotFoundError(f"Missing cleaned quality file: {quality_path}")
    quality = pd.read_csv(quality_path)

    point_rows = []
    for _, session in registry.iterrows():
        session_id = str(session["session_id"])
        session_path = Path(str(session.get("session_path") or ""))
        point_rows.append(pointcloud_audit_for_session(session_id, session_path))

    windows, manifest = generate_windows(quality, registry, out)
    fieldnames = list(windows[0].keys()) if windows else [
        "window_id",
        "session_id",
        "segment_id",
        "assigned_tid",
        "expected_pose",
    ]
    write_csv(out / "posturenet_lite_windows.csv", windows, fieldnames)
    write_csv(out / "dataset_manifest.csv", manifest, list(manifest[0].keys()) if manifest else ["session_id"])
    write_csv(out / "pointcloud_availability_report.csv", point_rows, list(point_rows[0].keys()) if point_rows else ["session_id"])
    write_dataset_report(out, registry, quality, windows, point_rows, manifest)

    print(f"Sessions in registry: {len(registry)}")
    print(f"Person-instances in cleaned segments: {len(quality)}")
    print(f"Lite windows written: {len(windows)}")
    print(
        "Full point-cloud data available: "
        + ("yes" if any(row["can_reconstruct_target_centered_tensors"] for row in point_rows) else "no")
    )
    print("Lite dataset available: " + ("yes" if windows else "no"))
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
