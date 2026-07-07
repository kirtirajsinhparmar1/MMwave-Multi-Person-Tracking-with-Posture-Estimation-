"""Generate offline RGB-assisted label candidates from the RGB teacher audit.

These labels are for analysis only. They are not runtime labels and they do
not replace the user-provided segment protocol ground truth.
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
DEFAULT_DATASET = REPO_ROOT / "analysis_outputs" / "posturenet_v2_dataset" / "posturenet_lite_windows.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cleaned-root", required=True)
    parser.add_argument("--rgb-audit", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def coarse_pose(value: object) -> str:
    text = norm_text(value)
    if "SIT" in text:
        return "SITTING"
    if "STAND" in text:
        return "STANDING"
    if "MOVE" in text or "WALK" in text:
        return "MOVING"
    if "LIE" in text:
        return "LYING"
    if "FALL" in text:
        return "FALLING"
    return "UNKNOWN"


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def display_pred(row: pd.Series) -> str:
    rates = {
        "STANDING": as_float(row.get("display_standing_rate"), 0.0),
        "SITTING": as_float(row.get("display_sitting_rate"), 0.0),
        "MOVING": as_float(row.get("display_moving_rate"), 0.0),
        "UNKNOWN": as_float(row.get("display_unknown_rate"), 0.0),
    }
    label, value = max(rates.items(), key=lambda item: item[1])
    return label if value > 0.0 else "UNKNOWN"


def build_sync_map(session_path: Path) -> pd.DataFrame:
    sync = read_csv(session_path / "sync_index.csv")
    needed = {"latest_rgb_frame_num", "latest_mmwave_frame_num"}
    if sync.empty or not needed.issubset(sync.columns):
        return pd.DataFrame(columns=["rgb_frame_num", "mmwave_frame", "mmwave_time_s"])
    work = sync.copy()
    work["latest_rgb_frame_num"] = pd.to_numeric(work["latest_rgb_frame_num"], errors="coerce")
    work["latest_mmwave_frame_num"] = pd.to_numeric(work["latest_mmwave_frame_num"], errors="coerce")
    work["latest_mmwave_monotonic_ns"] = pd.to_numeric(
        work.get("latest_mmwave_monotonic_ns"), errors="coerce"
    )
    work = work.dropna(subset=["latest_rgb_frame_num", "latest_mmwave_frame_num"])
    if work.empty:
        return pd.DataFrame(columns=["rgb_frame_num", "mmwave_frame", "mmwave_time_s"])
    first_mm_ns = work["latest_mmwave_monotonic_ns"].dropna().min()
    work["mmwave_time_s"] = (work["latest_mmwave_monotonic_ns"] - first_mm_ns) / 1e9
    mapped = (
        work.sort_values("latest_mmwave_frame_num")
        .groupby("latest_rgb_frame_num", as_index=False)
        .tail(1)
    )
    mapped = mapped.rename(
        columns={"latest_rgb_frame_num": "rgb_frame_num", "latest_mmwave_frame_num": "mmwave_frame"}
    )
    return mapped[["rgb_frame_num", "mmwave_frame", "mmwave_time_s"]]


def load_segments(cleaned_root: Path, session_id: str) -> pd.DataFrame:
    path = cleaned_root / "filled_segments" / f"{session_id}_segments_filled.csv"
    seg = read_csv(path)
    if seg.empty:
        return seg
    seg["start_time_s"] = pd.to_numeric(seg.get("start_time_s"), errors="coerce")
    seg["end_time_s"] = pd.to_numeric(seg.get("end_time_s"), errors="coerce")
    return seg


def assign_segment(time_s: float, segments: pd.DataFrame) -> dict[str, object]:
    if math.isnan(time_s) or segments.empty:
        return {
            "segment_id": "",
            "expected_pose": "UNKNOWN",
            "expected_subpose": "UNKNOWN",
            "expected_distance_m": "",
            "expected_position": "UNKNOWN",
            "segment_match_note": "no_time_or_segments",
            "seconds_to_segment_boundary": math.nan,
        }
    matches = segments[(segments["start_time_s"] <= time_s) & (segments["end_time_s"] >= time_s)]
    if matches.empty:
        return {
            "segment_id": "",
            "expected_pose": "UNKNOWN",
            "expected_subpose": "UNKNOWN",
            "expected_distance_m": "",
            "expected_position": "UNKNOWN",
            "segment_match_note": "outside_filled_segments",
            "seconds_to_segment_boundary": math.nan,
        }
    first = matches.iloc[0]
    boundary = min(abs(time_s - as_float(first.get("start_time_s"))), abs(as_float(first.get("end_time_s")) - time_s))
    note = "matched"
    if len(matches) > 1:
        note = "multiple_person_rows_same_time"
    return {
        "segment_id": first.get("segment_id", ""),
        "expected_pose": first.get("expected_pose", "UNKNOWN"),
        "expected_subpose": first.get("expected_subpose", "UNKNOWN"),
        "expected_distance_m": first.get("expected_distance_m", ""),
        "expected_position": first.get("expected_position", "UNKNOWN"),
        "segment_match_note": note,
        "seconds_to_segment_boundary": boundary,
    }


def load_windows() -> pd.DataFrame:
    windows = read_csv(DEFAULT_DATASET)
    if windows.empty:
        return windows
    windows["window_start_s"] = pd.to_numeric(windows.get("window_start_s"), errors="coerce")
    windows["window_end_s"] = pd.to_numeric(windows.get("window_end_s"), errors="coerce")
    windows["old_display_pred"] = windows.apply(display_pred, axis=1)
    return windows


def assign_old_display(time_s: float, session_id: str, windows: pd.DataFrame) -> str:
    if windows.empty or math.isnan(time_s):
        return "UNKNOWN"
    subset = windows[
        (windows["session_id"] == session_id)
        & (windows["window_start_s"] <= time_s)
        & (windows["window_end_s"] >= time_s)
    ]
    if subset.empty:
        return "UNKNOWN"
    return str(subset.iloc[0].get("old_display_pred", "UNKNOWN"))


def attach_old_display(session_id: str, rows: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["old_mmw_display_pose"] = "UNKNOWN"
    if windows.empty or "mmwave_time_s" not in out.columns:
        return out
    subset = windows[windows["session_id"] == session_id].copy()
    if subset.empty:
        return out
    subset = subset[["window_start_s", "window_end_s", "old_display_pred"]].dropna(
        subset=["window_start_s", "window_end_s"]
    )
    work = out.copy()
    work["_original_order"] = range(len(work))
    work["mmwave_time_s"] = pd.to_numeric(work["mmwave_time_s"], errors="coerce")
    valid = work[work["mmwave_time_s"].notna()].sort_values("mmwave_time_s")
    invalid = work[work["mmwave_time_s"].isna()]
    if not valid.empty:
        joined = pd.merge_asof(
            valid,
            subset.sort_values("window_start_s"),
            left_on="mmwave_time_s",
            right_on="window_start_s",
            direction="backward",
        )
        matched = joined["mmwave_time_s"] <= joined["window_end_s"]
        joined.loc[matched, "old_mmw_display_pose"] = joined.loc[matched, "old_display_pred"].fillna("UNKNOWN")
        joined.loc[~matched, "old_mmw_display_pose"] = "UNKNOWN"
        work = pd.concat([joined, invalid], ignore_index=True).sort_values("_original_order")
    return work.drop(columns=[col for col in ["_original_order", "window_start_s", "window_end_s", "old_display_pred"] if col in work.columns])


def write_report(out: Path, labels: pd.DataFrame, disagreements: pd.DataFrame) -> None:
    known = labels[labels["rgb_assisted_pose"].isin(["STANDING", "SITTING"])] if not labels.empty else pd.DataFrame()
    lines = [
        "# RGB Assisted Label Report",
        "",
        "RGB-assisted labels are offline candidates only. They are not ground truth and are not used by runtime posture decisions.",
        "",
        f"- Candidate rows: {len(labels)}",
        f"- Known STANDING/SITTING candidate rows: {len(known)}",
        f"- Disagreement rows: {len(disagreements)}",
        "",
        "The current RGB keypoints usually include shoulders and hips, but knees/ankles are absent in these sessions. Therefore most rows remain UNKNOWN or UNCERTAIN and should be used for manual review, sync checks, and transition discovery rather than automatic posture supervision.",
    ]
    (out / "RGB_ASSISTED_LABEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = ensure_dir(Path(args.out))
    registry = pd.read_csv(args.registry)
    candidates_path = Path(args.rgb_audit) / "rgb_teacher_candidate_labels.csv"
    candidates = read_csv(candidates_path)
    if candidates.empty:
        pd.DataFrame().to_csv(out / "rgb_assisted_frame_labels.csv", index=False)
        pd.DataFrame().to_csv(out / "rgb_mmw_disagreement_cases.csv", index=False)
        write_report(out, pd.DataFrame(), pd.DataFrame())
        print("No RGB teacher candidates found.")
        return 0

    cleaned_root = Path(args.cleaned_root)
    windows = load_windows()
    registry_paths = {str(row["session_id"]): row.get("session_path") for _, row in registry.iterrows()}

    rows: list[dict[str, object]] = []
    for session_id, group in candidates.groupby("session_id", sort=False):
        session_path = find_session_path(str(session_id), registry_paths.get(str(session_id)))
        sync_map = build_sync_map(session_path)
        segments = load_segments(cleaned_root, str(session_id))
        merged = group.merge(sync_map, how="left", on="rgb_frame_num") if not sync_map.empty else group.copy()
        if "mmwave_time_s" not in merged.columns:
            merged["mmwave_time_s"] = math.nan
        if "mmwave_frame" not in merged.columns:
            merged["mmwave_frame"] = math.nan
        merged = attach_old_display(str(session_id), merged, windows)
        for _, row in merged.iterrows():
            time_s = as_float(row.get("mmwave_time_s"))
            segment = assign_segment(time_s, segments)
            rgb_pose = coarse_pose(row.get("rgb_teacher_pose_candidate"))
            old_pose = coarse_pose(row.get("old_mmw_display_pose"))
            expected_pose = coarse_pose(segment["expected_pose"])
            rgb_known = rgb_pose in {"STANDING", "SITTING"}
            old_known = old_pose in {"STANDING", "SITTING"}
            near_boundary = as_float(segment.get("seconds_to_segment_boundary"), math.inf) <= 3.0
            out_row = {
                "session_id": session_id,
                "rgb_frame_num": row.get("rgb_frame_num"),
                "rgb_track_id": row.get("rgb_track_id"),
                "mmwave_frame": row.get("mmwave_frame"),
                "mmwave_time_s": time_s,
                "rgb_assisted_pose": rgb_pose,
                "rgb_assisted_subpose": row.get("rgb_teacher_subpose_candidate", "UNKNOWN"),
                "rgb_assisted_confidence": row.get("rgb_teacher_confidence"),
                "expected_pose": segment["expected_pose"],
                "expected_subpose": segment["expected_subpose"],
                "expected_distance_m": segment["expected_distance_m"],
                "expected_position": segment["expected_position"],
                "segment_id": segment["segment_id"],
                "segment_match_note": segment["segment_match_note"],
                "seconds_to_segment_boundary": segment["seconds_to_segment_boundary"],
                "old_mmw_display_pose": old_pose,
                "rgb_disagrees_segment_label": bool(rgb_known and expected_pose in {"STANDING", "SITTING"} and rgb_pose != expected_pose),
                "rgb_disagrees_old_mmw": bool(rgb_known and old_known and rgb_pose != old_pose),
                "potential_transition_or_review_frame": bool(
                    near_boundary or rgb_pose in {"UNKNOWN", "UNCERTAIN"} or as_float(row.get("rgb_teacher_confidence"), 0.0) < 0.35
                ),
                "note": "analysis_only_not_ground_truth",
            }
            rows.append(out_row)

    labels = pd.DataFrame(rows)
    disagreements = labels[
        (labels["rgb_disagrees_segment_label"])
        | (labels["rgb_disagrees_old_mmw"])
        | (labels["potential_transition_or_review_frame"])
    ]
    labels.to_csv(out / "rgb_assisted_frame_labels.csv", index=False)
    disagreements.to_csv(out / "rgb_mmw_disagreement_cases.csv", index=False)
    write_report(out, labels, disagreements)
    print(f"RGB-assisted labels written to {out}")
    print(f"Rows: {len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
