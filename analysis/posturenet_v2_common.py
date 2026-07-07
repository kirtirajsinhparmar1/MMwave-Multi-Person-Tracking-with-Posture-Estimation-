"""Shared helpers for the bounded RadarPostureNet-v2 analysis pass.

The scripts in this pass are intentionally conservative:

* protocol labels come from the user supplied registry, never from displayed UI
  posture;
* validation groups by sessions/segments instead of random frames;
* raw point-cloud tensors are used only if point rows with target association
  are actually present in logs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINED_ROOT = REPO_ROOT.parent
PARENT_LOGS = COMBINED_ROOT / "logs"
LOCAL_LOGS = REPO_ROOT / "logs"
DEFAULT_CFG = (
    r"C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples"
    r"\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking"
    r"\chirp_configs\ODS_6m_default.cfg"
)
STATIC_CFG = (
    r"C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples"
    r"\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking"
    r"\chirp_configs\ODS_6m_staticRetention.cfg"
)


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    recording_date: str
    people_count: int
    positions: str
    sequence_description: str
    distances_m: str
    poses_subposes: str
    trust_level: str
    notes: str
    cfg_hint: str = DEFAULT_CFG


SESSION_SPECS: list[SessionSpec] = [
    SessionSpec(
        "session_20260703_205540",
        "2026-07-03",
        1,
        "CENTER",
        "standing 1m, 2m, 3m, 4m; sitting lean-back 1m, 2m, 3m, 4m",
        "1;2;3;4",
        "STANDING/STANDING; SITTING/SITTING_LEAN_BACK",
        "HIGH",
        "Distance: 1m, 2m, 3m, 4m. People: 1. Position: center/front. Duration approximately 40-60 seconds per segment. Sitting was leaned back.",
    ),
    SessionSpec(
        "sitting_ab_default_cfg",
        "2026-07-04",
        1,
        "CENTER",
        "sitting lean-back 1m, 2m, 3m, 4m",
        "1;2;3;4",
        "SITTING/SITTING_LEAN_BACK",
        "HIGH",
        "Default cfg. This was just sitting leaned back. Corrected protocol includes 1m. Duration approximately 40-60 seconds per segment.",
    ),
    SessionSpec(
        "sitting_ab_static_retention_cfg",
        "2026-07-04",
        1,
        "CENTER",
        "sitting lean-back 1m, 2m, 3m, 4m",
        "1;2;3;4",
        "SITTING/SITTING_LEAN_BACK",
        "HIGH",
        "Static-retention cfg. This was just sitting leaned back. Corrected protocol includes 1m. Duration approximately 40-60 seconds per segment.",
        STATIC_CFG,
    ),
    SessionSpec(
        "sitting_relative_gate_refined_live_test",
        "2026-07-04",
        1,
        "CENTER",
        "standing 1m-5m; sitting lean-back 1m-5m; sitting upright 1m-5m; sitting lean-forward 1m-5m",
        "1;2;3;4;5",
        "STANDING/STANDING; SITTING/SITTING_LEAN_BACK; SITTING/SITTING_UPRIGHT; SITTING/SITTING_LEAN_FORWARD",
        "HIGH",
        "At least 40-45 seconds per segment. Occasional UI disappearance observed.",
    ),
    SessionSpec(
        "session_20260704_145249",
        "2026-07-04",
        2,
        "LEFT;RIGHT",
        "standing left/right at 1m-5m; sitting lean-back left/right at 1m-5m",
        "1;2;3;4;5",
        "STANDING/STANDING; SITTING/SITTING_LEAN_BACK",
        "HIGH",
        "Two-person simultaneous posture recording. From each center distance mark, one person was placed 1m left and one person 1m right.",
    ),
    SessionSpec(
        "session_20260704_150636",
        "2026-07-04",
        2,
        "LEFT;RIGHT",
        "sitting upright left/right at 1m-5m; sitting lean-forward left/right at 1m-5m",
        "1;2;3;4;5",
        "SITTING/SITTING_UPRIGHT; SITTING/SITTING_LEAN_FORWARD",
        "HIGH",
        "Two-person simultaneous sitting subtype recording. From each center distance mark, one person was placed 1m left and one person 1m right.",
    ),
    SessionSpec(
        "session_20260704_152302",
        "2026-07-04",
        1,
        "CENTER",
        "standing 1m-5m; sitting lean-back 1m-5m; sitting upright 1m-5m; sitting lean-forward 1m-5m",
        "1;2;3;4;5",
        "STANDING/STANDING; SITTING/SITTING_LEAN_BACK; SITTING/SITTING_UPRIGHT; SITTING/SITTING_LEAN_FORWARD",
        "HIGH",
        "Single person straight/front to the sensor. Duration approximately 40-60 seconds per segment.",
    ),
    SessionSpec(
        "session_20260706_173741",
        "2026-07-06",
        1,
        "CENTER;RIGHT;LEFT",
        "standing center/front 1m-5m; standing right side 1m-5m; standing left side 1m-5m",
        "1;2;3;4;5",
        "STANDING/STANDING",
        "HIGH",
        "Single-person standing at center/front, right side, and left side. Duration approximately 40-60 seconds per segment.",
    ),
    SessionSpec(
        "session_20260706_175519",
        "2026-07-06",
        1,
        "CENTER;RIGHT;LEFT",
        "sitting lean-back center/front 1m-5m; sitting lean-back right side 1m-5m; sitting lean-back left side 1m-5m",
        "1;2;3;4;5",
        "SITTING/SITTING_LEAN_BACK",
        "HIGH",
        "Single-person sitting lean-back at center/front, right side, and left side. Duration approximately 40-60 seconds per segment.",
    ),
]


SEGMENT_FIELDS = [
    "session_id",
    "segment_id",
    "person_slot",
    "expected_pose",
    "expected_subpose",
    "expected_distance_m",
    "expected_position",
    "start_time_s",
    "end_time_s",
    "label_confidence",
    "assigned_tid",
    "assignment_confidence",
    "notes",
]


def protocol_segments(session_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        segment_id: str,
        person_slot: str,
        pose: str,
        subpose: str,
        distance: float,
        position: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "session_id": session_id,
                "segment_id": segment_id,
                "person_slot": person_slot,
                "expected_pose": pose,
                "expected_subpose": subpose,
                "expected_distance_m": f"{distance:.1f}",
                "expected_position": position,
                "start_time_s": "",
                "end_time_s": "",
                "label_confidence": "HIGH",
                "assigned_tid": "",
                "assignment_confidence": "",
                "notes": notes,
            }
        )

    if session_id == "session_20260703_205540":
        for d in [1, 2, 3, 4]:
            add(f"standing_{d}m", "center", "STANDING", "STANDING", d, "CENTER", "standing center/front")
        for d in [1, 2, 3, 4]:
            add(
                f"leanback_{d}m",
                "center",
                "SITTING",
                "SITTING_LEAN_BACK",
                d,
                "CENTER",
                "sitting was leaned back",
            )
    elif session_id in {"sitting_ab_default_cfg", "sitting_ab_static_retention_cfg"}:
        for d in [1, 2, 3, 4]:
            add(
                f"leanback_{d}m",
                "center",
                "SITTING",
                "SITTING_LEAN_BACK",
                d,
                "CENTER",
                "corrected protocol includes 1m",
            )
    elif session_id in {"sitting_relative_gate_refined_live_test", "session_20260704_152302"}:
        for d in [1, 2, 3, 4, 5]:
            add(f"standing_{d}m", "center", "STANDING", "STANDING", d, "CENTER", "single person center/front")
        for subpose, prefix in [
            ("SITTING_LEAN_BACK", "leanback"),
            ("SITTING_UPRIGHT", "upright"),
            ("SITTING_LEAN_FORWARD", "leanforward"),
        ]:
            for d in [1, 2, 3, 4, 5]:
                add(
                    f"{prefix}_{d}m",
                    "center",
                    "SITTING",
                    subpose,
                    d,
                    "CENTER",
                    "single person center/front sitting subtype",
                )
    elif session_id == "session_20260704_145249":
        for pose, subpose, prefix in [
            ("STANDING", "STANDING", "standing"),
            ("SITTING", "SITTING_LEAN_BACK", "leanback"),
        ]:
            for d in [1, 2, 3, 4, 5]:
                add(f"{prefix}_{d}m", "left_person", pose, subpose, d, "LEFT", "simultaneous two-person left slot")
                add(f"{prefix}_{d}m", "right_person", pose, subpose, d, "RIGHT", "simultaneous two-person right slot")
    elif session_id == "session_20260704_150636":
        for subpose, prefix in [
            ("SITTING_UPRIGHT", "upright"),
            ("SITTING_LEAN_FORWARD", "leanforward"),
        ]:
            for d in [1, 2, 3, 4, 5]:
                add(f"{prefix}_{d}m", "left_person", "SITTING", subpose, d, "LEFT", "simultaneous two-person left slot")
                add(f"{prefix}_{d}m", "right_person", "SITTING", subpose, d, "RIGHT", "simultaneous two-person right slot")
    elif session_id == "session_20260706_173741":
        for position, slot in [("CENTER", "center"), ("RIGHT", "right"), ("LEFT", "left")]:
            for d in [1, 2, 3, 4, 5]:
                add(f"standing_{slot}_{d}m", slot, "STANDING", "STANDING", d, position, f"standing {slot} side")
    elif session_id == "session_20260706_175519":
        for position, slot in [("CENTER", "center"), ("RIGHT", "right"), ("LEFT", "left")]:
            for d in [1, 2, 3, 4, 5]:
                add(
                    f"leanback_{slot}_{d}m",
                    slot,
                    "SITTING",
                    "SITTING_LEAN_BACK",
                    d,
                    position,
                    f"sitting lean-back {slot} side",
                )
    else:
        raise KeyError(f"No protocol registered for {session_id}")

    return rows


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_value(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return result.stderr.strip() or f"git exited {result.returncode}"
    except Exception as exc:  # pragma: no cover - defensive report helper
        return f"unavailable: {exc}"


def search_roots() -> list[Path]:
    roots = [PARENT_LOGS, REPO_ROOT.parent / "logs", LOCAL_LOGS]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def discover_session_path(session_id: str) -> Path | None:
    candidates = []
    for root in search_roots():
        path = root / session_id
        if path.exists():
            score = 0
            if (path / "mmwave_pose.csv").exists():
                score += 10
            if (path / "mmwave_tracks.csv").exists():
                score += 10
            if (path / "pose_predictions_ui.csv").exists():
                score += 3
            if (path / "session_metadata.json").exists():
                score += 2
            candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def inventory_session(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            "exists": False,
            "csv_files": "",
            "has_rgb_video": False,
            "has_metadata": False,
            "cfg_path": "",
            "recording_date": "",
            "notes": "folder missing",
        }
    csv_files = sorted(p.name for p in path.glob("*.csv"))
    metadata_path = path / "session_metadata.json"
    metadata = read_json(metadata_path)
    videos_dir = path / "videos"
    has_video = any(
        p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} for p in videos_dir.glob("*")
    ) if videos_dir.exists() else False
    cfg_path = str(metadata.get("mmwave_cfg_path") or "")
    recording_date = str(
        metadata.get("created_wall_time_iso")
        or metadata.get("recording_date")
        or ""
    )
    notes = []
    required = ["mmwave_frames.csv", "mmwave_tracks.csv", "mmwave_pose.csv"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        notes.append("missing " + ";".join(missing))
    if not has_video:
        notes.append("no RGB video detected")
    if not metadata_path.exists():
        notes.append("no session_metadata.json")
    if metadata and metadata.get("mmwave_log_points") is False:
        notes.append("metadata says mmwave_log_points=false")
    return {
        "exists": True,
        "csv_files": ";".join(csv_files),
        "has_rgb_video": has_video,
        "has_metadata": metadata_path.exists(),
        "cfg_path": cfg_path,
        "recording_date": recording_date,
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def first_existing(path: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: object, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def normalized_label(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"SIT", "SITTING"}:
        return "SITTING"
    if text in {"STAND", "STANDING"}:
        return "STANDING"
    if text in {"WALKING", "MOVING"}:
        return "MOVING"
    if text in {"LYING", "LYING_DOWN"}:
        return "LYING"
    if text in {"FALLING", "FALL"}:
        return "FALLING"
    if text in {"WARMUP", "UNKNOWN", "NA", "NAN", "NONE", ""}:
        return "UNKNOWN"
    return text


def cfg_family(cfg_path: str) -> str:
    lower = (cfg_path or "").lower()
    if "staticretention" in lower or "static_retention" in lower:
        return "static_retention"
    if "default" in lower:
        return "default"
    return "unknown"


def position_code(position: str) -> int:
    return {"LEFT": -1, "CENTER": 0, "RIGHT": 1}.get(str(position).upper(), 0)


def confidence_rank(value: str) -> int:
    text = str(value or "").upper()
    if text == "HIGH":
        return 3
    if text == "MEDIUM":
        return 2
    if text == "LOW":
        return 1
    try:
        number = float(text)
        if number >= 0.8:
            return 3
        if number >= 0.5:
            return 2
        return 1
    except Exception:
        return 0


def mean(values: Iterable[float]) -> float:
    nums = [v for v in values if not math.isnan(v)]
    if not nums:
        return math.nan
    return sum(nums) / len(nums)


def std(values: Iterable[float]) -> float:
    nums = [v for v in values if not math.isnan(v)]
    if len(nums) <= 1:
        return 0.0 if nums else math.nan
    m = sum(nums) / len(nums)
    return math.sqrt(sum((v - m) ** 2 for v in nums) / len(nums))


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return numerator / denominator


def bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
