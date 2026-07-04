#!/usr/bin/env python
"""Create posture session registry and fill protocol segment times from logs.

This is an offline analysis utility. It does not touch runtime logic, cfg files,
RGB code, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ABS_LOG_ROOT = Path(r"C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs")


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    session_path: Path
    sequence_description: str
    segment_file: Path
    notes: str
    distances: list[float]
    blocks: list[tuple[str, str, str]]


SESSION_SPECS = [
    SessionSpec(
        "session_20260703_205540",
        ABS_LOG_ROOT / "session_20260703_205540",
        "standing 1-4m then sitting lean-back 1-4m, approx 40-60 sec per segment",
        Path("analysis_inputs/session_20260703_205540_segments.csv"),
        "user-provided protocol; sitting was leaned back",
        [1.0, 2.0, 3.0, 4.0],
        [("standing", "STANDING", "STANDING"), ("leanback", "SITTING", "SITTING_LEAN_BACK")],
    ),
    SessionSpec(
        "sitting_ab_default_cfg",
        ABS_LOG_ROOT / "sitting_ab_default_cfg",
        "sitting lean-back 1-4m, approx 40-60 sec per segment",
        Path("analysis_inputs/sitting_ab_default_segments_corrected_1to4.csv"),
        "default cfg; corrected protocol includes 1m",
        [1.0, 2.0, 3.0, 4.0],
        [("leanback", "SITTING", "SITTING_LEAN_BACK")],
    ),
    SessionSpec(
        "sitting_ab_static_retention_cfg",
        ABS_LOG_ROOT / "sitting_ab_static_retention_cfg",
        "sitting lean-back 1-4m, approx 40-60 sec per segment",
        Path("analysis_inputs/sitting_ab_static_retention_segments_corrected_1to4.csv"),
        "static-retention cfg; corrected protocol includes 1m",
        [1.0, 2.0, 3.0, 4.0],
        [("leanback", "SITTING", "SITTING_LEAN_BACK")],
    ),
    SessionSpec(
        "sitting_relative_gate_refined_live_test",
        ABS_LOG_ROOT / "sitting_relative_gate_refined_live_test",
        "standing 1-5m, sitting lean-back 1-5m, sitting upright 1-5m, sitting lean-forward 1-5m; at least 40-45 sec per segment",
        Path("analysis_inputs/sitting_relative_gate_live_segments.csv"),
        "live refined-gate validation; user noticed occasional UI disappearance",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [
            ("standing", "STANDING", "STANDING"),
            ("leanback", "SITTING", "SITTING_LEAN_BACK"),
            ("upright", "SITTING", "SITTING_UPRIGHT"),
            ("leanforward", "SITTING", "SITTING_LEAN_FORWARD"),
        ],
    ),
]


SEGMENT_COLUMNS = [
    "session_id",
    "segment_id",
    "expected_pose",
    "expected_subpose",
    "expected_distance_m",
    "start_time_s",
    "end_time_s",
    "label_confidence",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="analysis_inputs/posture_session_registry.csv")
    parser.add_argument("--out-dir", default="analysis_inputs")
    return parser.parse_args()


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, dest)
    return dest


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metadata_cfg(meta: dict[str, Any]) -> str:
    for key in ("cfg_path", "config_path", "mmwave_cfg_path", "radar_cfg_path"):
        value = meta.get(key)
        if value:
            return str(value)
    mmwave = meta.get("mmwave") if isinstance(meta.get("mmwave"), dict) else {}
    for key in ("cfg_path", "config_path"):
        value = mmwave.get(key)
        if value:
            return str(value)
    return ""


def metadata_date(meta: dict[str, Any]) -> str:
    for key in ("created_wall_time_iso", "start_wall_time_iso", "recording_date", "started_at"):
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def has_rgb_video(path: Path, meta: dict[str, Any]) -> str:
    present = any((path / name).exists() for name in ("rgb_annotated.mp4", "videos/rgb_annotated.mp4"))
    if not present:
        present = bool(list(path.glob("**/rgb_annotated.mp4")))
    if present:
        return "true"
    if bool(meta.get("rgb_record_video")):
        return "missing_file_but_metadata_requested_video"
    return "false"


def segment_rows(spec: SessionSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, pose, subpose in spec.blocks:
        for distance in spec.distances:
            note = ""
            if spec.session_id == "session_20260703_205540":
                note = "standing first block" if pose == "STANDING" else "sitting was leaned back"
            elif spec.session_id == "sitting_ab_default_cfg":
                note = "corrected protocol includes 1m" if distance == 1.0 else "sitting leaned back"
            elif spec.session_id == "sitting_ab_static_retention_cfg":
                note = "corrected protocol includes 1m" if distance == 1.0 else "sitting leaned back"
            elif spec.session_id == "sitting_relative_gate_refined_live_test":
                note = "live refined-gate validation protocol"
            rows.append(
                {
                    "session_id": spec.session_id,
                    "segment_id": f"{prefix}_{int(distance)}m",
                    "expected_pose": pose,
                    "expected_subpose": subpose,
                    "expected_distance_m": distance,
                    "start_time_s": "",
                    "end_time_s": "",
                    "label_confidence": "",
                    "notes": note,
                }
            )
    return rows


def load_existing_times(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        existing = pd.read_csv(path)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in existing.iterrows():
        sid = str(row.get("segment_id", ""))
        if not sid:
            continue
        out[sid] = {
            "start_time_s": row.get("start_time_s", ""),
            "end_time_s": row.get("end_time_s", ""),
            "label_confidence": row.get("label_confidence", row.get("confidence", "")),
            "notes": row.get("notes", ""),
        }
    return out


def numeric_time(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    for col in ("time", "time_s", "timestamp_s", "elapsed_s"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                if values.max() > 1e6:
                    values = values - values.min()
                return values.astype(float)
    for col in ("host_monotonic_ns", "monotonic_ns"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return ((values - values.min()) / 1e9).astype(float)
    for col in ("host_wall_time_iso", "timestamp", "wall_time"):
        if col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce", utc=True)
            if values.notna().any():
                return (values - values.min()).dt.total_seconds().astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def read_range_series(session: Path) -> pd.DataFrame:
    candidates = [session / "mmwave_tracks.csv", session / "pose_predictions_ui.csv", Path("logs") / session.name / "pose_predictions_ui.csv"]
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        out = pd.DataFrame(index=df.index)
        out["timestamp_s"] = numeric_time(df)
        for target, names in {
            "range_m": ["range_m", "range", "distance_m"],
            "x_m": ["x_m", "x"],
            "y_m": ["y_m", "y"],
            "tid": ["tid", "track_id", "target_id"],
        }.items():
            col = next((c for c in names if c in df.columns), None)
            out[target] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
        if out["range_m"].isna().all() and out["x_m"].notna().any() and out["y_m"].notna().any():
            out["range_m"] = np.sqrt(out["x_m"] ** 2 + out["y_m"] ** 2)
        out = out.dropna(subset=["timestamp_s", "range_m"])
        if not out.empty:
            return out.sort_values("timestamp_s")
    return pd.DataFrame(columns=["timestamp_s", "range_m", "tid"])


def contiguous_runs(times: pd.Series, max_gap_s: float = 3.5) -> list[tuple[float, float, int]]:
    vals = sorted(float(v) for v in times.dropna().unique())
    if not vals:
        return []
    runs: list[tuple[float, float, int]] = []
    start = prev = vals[0]
    count = 1
    for value in vals[1:]:
        if value - prev <= max_gap_s:
            prev = value
            count += 1
        else:
            runs.append((start, prev, count))
            start = prev = value
            count = 1
    runs.append((start, prev, count))
    return runs


def fill_times(rows: list[dict[str, Any]], session: Path) -> tuple[list[dict[str, Any]], list[str]]:
    report: list[str] = []
    data = read_range_series(session)
    if data.empty:
        report.append(f"- {session.name}: no range-bearing log rows found; used time-order fallback.")
    cursor = float(data["timestamp_s"].min()) if not data.empty else 0.0
    session_end = float(data["timestamp_s"].max()) if not data.empty else len(rows) * 50.0
    for idx, row in enumerate(rows):
        if str(row.get("start_time_s", "")).strip() and str(row.get("end_time_s", "")).strip():
            report.append(f"- {row['segment_id']}: kept existing {row['start_time_s']}-{row['end_time_s']}s.")
            continue
        dist = float(row["expected_distance_m"])
        chosen: tuple[float, float, int, float] | None = None
        if not data.empty:
            for tol in (0.35, 0.55, 0.80, 1.10, 1.50):
                cand = data[(data["timestamp_s"] >= cursor) & ((data["range_m"] - dist).abs() <= tol)]
                runs = contiguous_runs(cand["timestamp_s"], max_gap_s=3.5)
                runs = [r for r in runs if r[1] - r[0] >= 15.0]
                if runs:
                    raw_start, raw_end, count = runs[0]
                    chosen = (raw_start, raw_end, count, tol)
                    break
        if chosen:
            raw_start, raw_end, count, tol = chosen
            duration = raw_end - raw_start
            trim = 4.0 if duration >= 28.0 else max(0.0, min(2.0, duration * 0.10))
            start = raw_start + trim
            end = raw_end - trim
            if end <= start:
                start, end = raw_start, raw_end
            conf = 0.95 if end - start >= 40.0 and tol <= 0.80 else min(0.90, max(0.35, (end - start) / 45.0) * (0.80 / tol))
            row["start_time_s"] = round(start, 3)
            row["end_time_s"] = round(end, 3)
            row["label_confidence"] = round(conf, 3)
            row["notes"] = f"{row.get('notes', '')}; auto range plateau tol={tol:.2f}m samples={count}".strip("; ")
            cursor = raw_end + 2.0
            warning = " weak/short" if end - start < 35.0 else ""
            report.append(
                f"- {row['segment_id']}: {start:.2f}-{end:.2f}s from range plateau at {dist:g}m "
                f"(tol={tol:.2f}m, confidence={conf:.2f}).{warning}"
            )
        else:
            fallback_start = cursor + 4.0
            fallback_end = min(fallback_start + 45.0, session_end)
            if fallback_end <= fallback_start:
                fallback_start = idx * 50.0 + 4.0
                fallback_end = fallback_start + 42.0
            row["start_time_s"] = round(fallback_start, 3)
            row["end_time_s"] = round(fallback_end, 3)
            row["label_confidence"] = 0.25
            row["notes"] = f"{row.get('notes', '')}; time-order fallback, inspect manually".strip("; ")
            cursor = fallback_end + 2.0
            report.append(f"- {row['segment_id']}: fallback {fallback_start:.2f}-{fallback_end:.2f}s (confidence=0.25).")
    return rows, report


def write_registry(registry_path: Path) -> pd.DataFrame:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    backup(registry_path)
    rows = []
    for spec in SESSION_SPECS:
        meta = read_json(spec.session_path / "session_metadata.json")
        rows.append(
            {
                "session_id": spec.session_id,
                "session_path": str(spec.session_path),
                "cfg_path": metadata_cfg(meta),
                "recording_date": metadata_date(meta),
                "sequence_description": spec.sequence_description,
                "has_rgb_video": has_rgb_video(spec.session_path, meta),
                "segment_file": str(spec.segment_file),
                "trust_level": "HIGH",
                "notes": spec.notes,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(registry_path, index=False)
    return df


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Posture Session Registry Segment Filling Report",
        "",
        "Generated from user-provided protocols and range/time evidence in the logs.",
        "",
    ]
    write_registry(registry_path)
    for spec in SESSION_SPECS:
        path = spec.segment_file
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_times(path)
        rows = segment_rows(spec)
        for row in rows:
            old = existing.get(row["segment_id"])
            if not old:
                continue
            if str(old.get("start_time_s", "")).strip() and not (isinstance(old.get("start_time_s"), float) and math.isnan(old["start_time_s"])):
                row["start_time_s"] = old.get("start_time_s", "")
                row["end_time_s"] = old.get("end_time_s", "")
                row["label_confidence"] = old.get("label_confidence", "")
                row["notes"] = old.get("notes", row["notes"]) or row["notes"]
        rows, lines = fill_times(rows, spec.session_path)
        backup(path)
        pd.DataFrame(rows, columns=SEGMENT_COLUMNS).to_csv(path, index=False)
        report_lines.extend([f"## {spec.session_id}", "", *lines, ""])
    out_report = Path("analysis_outputs/posture_session_registry_segment_filling_report.md")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote registry: {registry_path}")
    print(f"Wrote segment filling report: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
