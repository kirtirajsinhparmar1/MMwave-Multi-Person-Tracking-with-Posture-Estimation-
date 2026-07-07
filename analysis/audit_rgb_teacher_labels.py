"""Audit whether RGB sidecar logs can serve as an offline label teacher.

The audit is deliberately conservative. It reports whether RGB keypoints can
assist frame-level analysis, but it does not promote RGB heuristics to ground
truth without manual/high-confidence verification.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINED_ROOT = REPO_ROOT.parent
PARENT_LOGS = COMBINED_ROOT / "logs"
LOCAL_LOGS = REPO_ROOT / "logs"

JOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cleaned-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def find_session_path(session_id: str, registry_path: object) -> Path:
    candidates = []
    if registry_path:
        candidates.append(Path(str(registry_path)))
    candidates.append(PARENT_LOGS / session_id)
    candidates.append(LOCAL_LOGS / session_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_csv_if_exists(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def center_from_joints(group: pd.DataFrame, left_idx: int, right_idx: int) -> tuple[float, float, float]:
    joints = group[group["joint_index"].isin([left_idx, right_idx])]
    joints = joints[pd.to_numeric(joints["score"], errors="coerce").fillna(0.0) >= 0.20]
    if joints.empty:
        return math.nan, math.nan, 0.0
    return (
        float(pd.to_numeric(joints["x_norm"], errors="coerce").mean()),
        float(pd.to_numeric(joints["y_norm"], errors="coerce").mean()),
        float(pd.to_numeric(joints["score"], errors="coerce").mean()),
    )


def build_candidate_labels(session_id: str, keypoints: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    needed = {"rgb_frame_num", "rgb_track_id", "joint_index", "x_norm", "y_norm", "score"}
    if not needed.issubset(set(keypoints.columns)):
        return pd.DataFrame()

    kp = keypoints[list(needed)].copy()
    kp["joint_index"] = pd.to_numeric(kp["joint_index"], errors="coerce").astype("Int64")
    kp["score"] = pd.to_numeric(kp["score"], errors="coerce").fillna(0.0)
    kp["x_norm"] = pd.to_numeric(kp["x_norm"], errors="coerce")
    kp["y_norm"] = pd.to_numeric(kp["y_norm"], errors="coerce")

    bbox = pd.DataFrame()
    if {"rgb_frame_num", "rgb_track_id", "bbox_y1_px", "bbox_y2_px", "bbox_x1_px", "bbox_x2_px"}.issubset(
        tracks.columns
    ):
        bbox = tracks[
            ["rgb_frame_num", "rgb_track_id", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"]
        ].copy()
        for col in ["bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"]:
            bbox[col] = pd.to_numeric(bbox[col], errors="coerce")
        bbox["bbox_height_px"] = (bbox["bbox_y2_px"] - bbox["bbox_y1_px"]).abs()
        bbox["bbox_width_px"] = (bbox["bbox_x2_px"] - bbox["bbox_x1_px"]).abs()
        bbox = bbox[["rgb_frame_num", "rgb_track_id", "bbox_height_px", "bbox_width_px"]]

    rows: list[dict[str, object]] = []
    for (frame, track), group in kp.groupby(["rgb_frame_num", "rgb_track_id"], sort=False):
        shoulder_x, shoulder_y, shoulder_conf = center_from_joints(group, 5, 6)
        hip_x, hip_y, hip_conf = center_from_joints(group, 11, 12)
        knee_x, knee_y, knee_conf = center_from_joints(group, 13, 14)
        ankle_x, ankle_y, ankle_conf = center_from_joints(group, 15, 16)
        visible = int((group["score"] >= 0.20).sum())
        confidence = float(group["score"].mean()) if len(group) else 0.0

        bbox_h = math.nan
        bbox_w = math.nan
        if not bbox.empty:
            match = bbox[(bbox["rgb_frame_num"] == frame) & (bbox["rgb_track_id"] == track)]
            if not match.empty:
                bbox_h = float(match.iloc[0]["bbox_height_px"])
                bbox_w = float(match.iloc[0]["bbox_width_px"])

        torso_vertical = hip_y - shoulder_y if not math.isnan(hip_y) and not math.isnan(shoulder_y) else math.nan
        torso_lateral = hip_x - shoulder_x if not math.isnan(hip_x) and not math.isnan(shoulder_x) else math.nan
        torso_angle_deg = math.nan
        if not math.isnan(torso_vertical) and abs(torso_vertical) > 1e-6 and not math.isnan(torso_lateral):
            torso_angle_deg = math.degrees(math.atan2(torso_lateral, torso_vertical))
        leg_vertical = ankle_y - hip_y if not math.isnan(ankle_y) and not math.isnan(hip_y) else math.nan
        knee_hip_vertical = knee_y - hip_y if not math.isnan(knee_y) and not math.isnan(hip_y) else math.nan

        if confidence < 0.30 or shoulder_conf == 0.0 or hip_conf == 0.0:
            candidate = "UNKNOWN"
            heuristic_conf = 0.10
        elif not math.isnan(leg_vertical) and leg_vertical >= 0.42 and visible >= 10:
            candidate = "STANDING"
            heuristic_conf = min(0.80, 0.35 + confidence)
        elif not math.isnan(leg_vertical) and leg_vertical < 0.36 and not math.isnan(knee_hip_vertical):
            candidate = "SITTING"
            heuristic_conf = min(0.70, 0.25 + confidence)
        elif not math.isnan(knee_hip_vertical) and knee_hip_vertical < 0.18:
            candidate = "SITTING"
            heuristic_conf = min(0.55, 0.15 + confidence)
        else:
            candidate = "UNCERTAIN"
            heuristic_conf = min(0.45, 0.10 + confidence)

        rows.append(
            {
                "session_id": session_id,
                "rgb_frame_num": frame,
                "rgb_track_id": track,
                "bbox_height_px": bbox_h,
                "bbox_width_px": bbox_w,
                "shoulder_center_x_norm": shoulder_x,
                "shoulder_center_y_norm": shoulder_y,
                "hip_center_x_norm": hip_x,
                "hip_center_y_norm": hip_y,
                "knee_center_x_norm": knee_x,
                "knee_center_y_norm": knee_y,
                "ankle_center_x_norm": ankle_x,
                "ankle_center_y_norm": ankle_y,
                "torso_angle_deg": torso_angle_deg,
                "hip_to_shoulder_vertical_norm": torso_vertical,
                "ankle_to_hip_vertical_norm": leg_vertical,
                "knee_to_hip_vertical_norm": knee_hip_vertical,
                "visible_keypoints": visible,
                "mean_keypoint_score": confidence,
                "rgb_teacher_pose_candidate": candidate,
                "rgb_teacher_confidence": heuristic_conf,
                "rgb_teacher_subpose_candidate": "UNKNOWN",
            }
        )
    return pd.DataFrame(rows)


def audit_session(row: pd.Series) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], str]:
    session_id = str(row["session_id"])
    session_path = find_session_path(session_id, row.get("session_path"))
    keypoints_path = session_path / "rgb_keypoints.csv"
    tracks_path = session_path / "rgb_tracks.csv"
    frames_path = session_path / "rgb_frames.csv"
    sync_path = session_path / "sync_index.csv"
    video_files = list((session_path / "videos").glob("*.mp4")) + list((session_path / "videos").glob("*.avi"))

    keypoints = read_csv_if_exists(
        keypoints_path,
        ["rgb_frame_num", "rgb_track_id", "joint_index", "x_px", "y_px", "score", "x_norm", "y_norm"],
    )
    tracks = read_csv_if_exists(tracks_path)
    frames = read_csv_if_exists(frames_path)
    sync = read_csv_if_exists(sync_path)

    person_col = "rgb_track_id" if "rgb_track_id" in keypoints.columns else None
    frame_col = "rgb_frame_num" if "rgb_frame_num" in keypoints.columns else None
    score_col = "score" if "score" in keypoints.columns else None
    joint_col = "joint_index" if "joint_index" in keypoints.columns else None

    keypoint_rows = len(keypoints)
    keypoint_frames = int(keypoints[frame_col].nunique()) if frame_col else 0
    rgb_person_ids = int(keypoints[person_col].nunique()) if person_col else 0
    mean_score = float(pd.to_numeric(keypoints[score_col], errors="coerce").mean()) if score_col else 0.0
    p10_score = float(pd.to_numeric(keypoints[score_col], errors="coerce").quantile(0.10)) if score_col else 0.0
    low_score_rate = float((pd.to_numeric(keypoints[score_col], errors="coerce").fillna(0.0) < 0.20).mean()) if score_col and keypoint_rows else 0.0

    expected_groups = 0
    observed_groups = 0
    missing_keypoint_rate = 1.0
    if frame_col and person_col and keypoint_rows:
        observed_groups = keypoints[[frame_col, person_col]].drop_duplicates().shape[0]
        expected_groups = observed_groups * 17
        missing_keypoint_rate = max(0.0, 1.0 - float(keypoint_rows) / float(expected_groups)) if expected_groups else 1.0

    present_joints = set()
    if joint_col and score_col and keypoint_rows:
        visible_joints = keypoints[pd.to_numeric(keypoints[score_col], errors="coerce").fillna(0.0) >= 0.20]
        present_joints = set(pd.to_numeric(visible_joints[joint_col], errors="coerce").dropna().astype(int).tolist())
    has_shoulders = {5, 6}.issubset(present_joints)
    has_hips = {11, 12}.issubset(present_joints)
    has_knees = {13, 14}.issubset(present_joints)
    has_ankles = {15, 16}.issubset(present_joints)

    max_tracks_frame = 0
    if "num_tracks" in frames.columns and len(frames):
        max_tracks_frame = int(pd.to_numeric(frames["num_tracks"], errors="coerce").fillna(0).max())
    elif frame_col and person_col and keypoint_rows:
        max_tracks_frame = int(keypoints.groupby(frame_col)[person_col].nunique().max())

    two_person_distinguishable = False
    if as_float(row.get("people_count"), 1.0) >= 2 and {"rgb_frame_num", "rgb_track_id", "bbox_x1_px", "bbox_x2_px"}.issubset(tracks.columns):
        centers = tracks[["rgb_frame_num", "rgb_track_id", "bbox_x1_px", "bbox_x2_px"]].copy()
        centers["center_x"] = (
            pd.to_numeric(centers["bbox_x1_px"], errors="coerce")
            + pd.to_numeric(centers["bbox_x2_px"], errors="coerce")
        ) / 2.0
        spread = centers.groupby("rgb_frame_num")["center_x"].agg(["count", "max", "min"])
        two_person_distinguishable = bool(((spread["count"] >= 2) & ((spread["max"] - spread["min"]) > 30)).mean() > 0.25)

    candidate = build_candidate_labels(session_id, keypoints, tracks)
    candidate_conf_mean = float(candidate["rgb_teacher_confidence"].mean()) if not candidate.empty else 0.0
    candidate_known_rate = float((candidate["rgb_teacher_pose_candidate"].isin(["STANDING", "SITTING"])).mean()) if not candidate.empty else 0.0

    schema_text = [
        f"## {session_id}",
        "",
        f"- Session path: `{session_path}`",
        f"- rgb_keypoints.csv columns: {', '.join(map(str, keypoints.columns)) if not keypoints.empty else 'missing or empty'}",
        f"- rgb_tracks.csv columns: {', '.join(map(str, tracks.columns)) if not tracks.empty else 'missing or empty'}",
        f"- rgb_frames.csv columns: {', '.join(map(str, frames.columns)) if not frames.empty else 'missing or empty'}",
        f"- sync_index.csv columns: {', '.join(map(str, sync.columns)) if not sync.empty else 'missing or empty'}",
        "",
    ]

    availability = {
        "session_id": session_id,
        "session_path": str(session_path),
        "rgb_video_exists": bool(video_files),
        "rgb_keypoints_exists": keypoints_path.exists(),
        "rgb_tracks_exists": tracks_path.exists(),
        "rgb_frames_exists": frames_path.exists(),
        "sync_index_exists": sync_path.exists(),
        "rgb_keypoint_rows": keypoint_rows,
        "rgb_track_rows": len(tracks),
        "rgb_frame_rows": len(frames),
        "sync_rows": len(sync),
        "keypoint_format": "long_joint_index" if {"joint_index", "x_norm", "y_norm", "score"}.issubset(keypoints.columns) else "unknown",
        "person_ids_available": bool(person_col),
        "track_ids_available": bool(person_col),
        "mmwave_sync_available": {"latest_mmwave_frame_num", "latest_rgb_frame_num"}.issubset(sync.columns),
        "rgb_person_ids": rgb_person_ids,
        "max_rgb_tracks_per_frame": max_tracks_frame,
        "two_person_left_right_distinguishable": two_person_distinguishable,
        "shoulders_available": has_shoulders,
        "hips_available": has_hips,
        "knees_available": has_knees,
        "ankles_available": has_ankles,
        "torso_angle_computable": has_shoulders and has_hips,
        "body_height_ratio_computable": has_shoulders and has_hips and has_knees,
    }
    quality = {
        "session_id": session_id,
        "keypoint_frames": keypoint_frames,
        "keypoint_track_instances": observed_groups,
        "mean_keypoint_score": mean_score,
        "p10_keypoint_score": p10_score,
        "low_score_keypoint_rate": low_score_rate,
        "missing_keypoint_rate": missing_keypoint_rate,
        "candidate_label_rows": len(candidate),
        "candidate_known_pose_rate": candidate_known_rate,
        "candidate_confidence_mean": candidate_conf_mean,
        "rgb_teacher_status": "partial"
        if keypoint_rows and has_shoulders and has_hips and len(sync)
        else "not_usable",
    }
    return availability, candidate, quality, "\n".join(schema_text)


def determine_status(availability: pd.DataFrame, quality: pd.DataFrame) -> str:
    if availability.empty:
        return "no"
    partial_sessions = availability[
        (availability["rgb_video_exists"])
        & (availability["rgb_keypoints_exists"])
        & (availability["rgb_tracks_exists"])
        & (availability["sync_index_exists"])
        & (availability["shoulders_available"])
        & (availability["hips_available"])
    ]
    usable_sessions = partial_sessions[
        (partial_sessions["knees_available"]) & (partial_sessions["ankles_available"])
    ]
    good_quality = quality[
        (quality["mean_keypoint_score"] >= 0.35)
        & (quality["candidate_label_rows"] > 0)
        & (quality["candidate_known_pose_rate"] > 0.05)
    ]
    if len(usable_sessions) == len(availability) and len(good_quality) == len(quality):
        return "yes"
    if len(partial_sessions) > 0:
        return "partial"
    return "no"


def write_reports(out: Path, availability: pd.DataFrame, quality: pd.DataFrame, schema_sections: list[str], status: str) -> None:
    schema_lines = [
        "# RGB Keypoint Schema Report",
        "",
        "Observed RGB logs use a long keypoint schema where each row is one joint for one RGB track in one RGB frame. Joint indices match the common COCO order used by the logger: shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16.",
        "",
    ]
    schema_lines.extend(schema_sections)
    (out / "rgb_keypoint_schema_report.md").write_text("\n".join(schema_lines) + "\n", encoding="utf-8")

    session_count = len(availability)
    video_count = int(availability["rgb_video_exists"].sum()) if not availability.empty else 0
    keypoint_count = int(availability["rgb_keypoints_exists"].sum()) if not availability.empty else 0
    sync_count = int(availability["sync_index_exists"].sum()) if not availability.empty else 0
    mean_score = float(quality["mean_keypoint_score"].mean()) if not quality.empty else 0.0
    known_rate = float(quality["candidate_known_pose_rate"].mean()) if not quality.empty else 0.0

    lines = [
        "# RGB Teacher Audit Report",
        "",
        f"RGB teacher status: {status.upper() if status != 'partial' else 'PARTIALLY USABLE'}",
        "",
        "The existing RGB files can assist offline label review, sync checks, transition discovery, and weak torso-angle analysis. They are not treated as ground truth by this report because lower-body keypoints are missing, camera perspective is not calibrated, two-person identity alignment is not verified, and lean-forward/lean-back direction requires manual review.",
        "",
        "## Summary",
        "",
        f"- Sessions audited: {session_count}",
        f"- Sessions with RGB video: {video_count}",
        f"- Sessions with RGB keypoints: {keypoint_count}",
        f"- Sessions with sync_index.csv: {sync_count}",
        f"- Mean keypoint score across sessions: {mean_score:.3f}",
        f"- Mean candidate STANDING/SITTING label rate: {known_rate:.3f}",
        "",
        "## Decision",
        "",
    ]
    if status == "partial":
        lines.append("RGB teacher is partially usable now: it can align synchronized video/keypoint tracks to mmWave time and support manual review or weak torso-angle analysis, but missing knee/ankle coverage prevents robust automatic frame-level posture/subpose labels.")
    elif status == "yes":
        lines.append("RGB teacher is usable now for high-confidence coarse labels, subject to validation against manual review.")
    else:
        lines.append("RGB teacher is not usable without more logging or manual video review.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Shoulders, hips, knees, and ankles are required for torso/body-ratio features.",
            "- Two-person sessions can often distinguish left/right tracks by bbox x-position, but this is not the same as verified mmWave TID alignment.",
            "- Lean-forward vs lean-back is not reliably inferred from front-facing RGB keypoints alone.",
        ]
    )
    (out / "RGB_TEACHER_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = ensure_dir(Path(args.out))
    registry = pd.read_csv(args.registry)

    availability_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    candidates: list[pd.DataFrame] = []
    schema_sections: list[str] = []

    for _, row in registry.iterrows():
        availability, candidate, quality, schema_text = audit_session(row)
        availability_rows.append(availability)
        quality_rows.append(quality)
        if not candidate.empty:
            candidates.append(candidate)
        schema_sections.append(schema_text)

    availability_df = pd.DataFrame(availability_rows)
    quality_df = pd.DataFrame(quality_rows)
    candidate_df = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    status = determine_status(availability_df, quality_df)

    availability_df.to_csv(out / "rgb_teacher_availability.csv", index=False)
    quality_df.to_csv(out / "rgb_teacher_quality_by_session.csv", index=False)
    candidate_df.to_csv(out / "rgb_teacher_candidate_labels.csv", index=False)
    write_reports(out, availability_df, quality_df, schema_sections, status)
    print(f"RGB teacher audit written to {out}")
    print(f"RGB teacher usable: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
