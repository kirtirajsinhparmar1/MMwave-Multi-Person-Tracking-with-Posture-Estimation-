"""Optional per-frame associated point-cloud logger for posture data collection."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any


ASSOCIATED_POINT_COLUMNS = [
    "session_id",
    "frame",
    "timestamp_s",
    "tid",
    "track_index",
    "point_index",
    "association_source",
    "association_confidence",
    "point_x_m",
    "point_y_m",
    "point_z_m",
    "point_range_m",
    "point_azimuth_deg",
    "point_elevation_deg",
    "point_doppler_mps",
    "point_snr",
    "point_noise",
    "point_quality",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_range_m",
    "target_azimuth_deg",
    "target_elevation_deg",
    "target_vx_mps",
    "target_vy_mps",
    "target_vz_mps",
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_range_m",
    "relative_radial_m",
    "relative_lateral_m",
    "height_above_ground_m",
    "is_valid_point",
    "geom_pts_for_tid",
    "points_total_frame",
    "quality_label_for_tid",
    "old_display_pose",
    "old_model_stand_prob",
    "old_model_sit_prob",
    "old_model_move_prob",
    "old_model_lie_prob",
    "old_model_fall_prob",
]


class AssociatedPointCloudLogger:
    """Buffered CSV logger for points associated to tracked target IDs."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        session_id: str | None = None,
        max_points_per_tid: int = 64,
        log_format: str = "csv",
        ground_z: float = 0.0,
        flush_every_rows: int = 2048,
        flush_every_frames: int = 30,
        write_empty_summary: bool = True,
        debug: bool = False,
    ) -> None:
        if str(log_format).lower() != "csv":
            raise ValueError("Associated point logging currently supports only csv format")
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id or self.out_dir.name)
        self.max_points_per_tid = max(1, int(max_points_per_tid))
        self.ground_z = float(ground_z)
        self.flush_every_rows = max(1, int(flush_every_rows))
        self.flush_every_frames = max(1, int(flush_every_frames))
        self.write_empty_summary = bool(write_empty_summary)
        self.debug = bool(debug)
        self.path = self.out_dir / "mmwave_associated_points.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=ASSOCIATED_POINT_COLUMNS)
        self._writer.writeheader()
        self._buffer: list[dict[str, Any]] = []
        self._frames_since_flush = 0
        self._closed = False
        self._error_count = 0

    def log_frame(
        self,
        *,
        frame_num: int,
        timestamp_s: float | None,
        points: list[list[float]] | None,
        results: dict[int, dict[str, Any]] | None,
        associations: dict[int, dict[str, Any]] | None,
        ground_z: float | None = None,
    ) -> None:
        if self._closed:
            return
        try:
            frame = int(frame_num)
            ts = float(timestamp_s if timestamp_s is not None else time.time())
            points_total = len(points or [])
            frame_ground_z = self.ground_z if ground_z is None else float(ground_z)
            for tid in sorted(results or {}):
                pose = (results or {}).get(tid, {}) or {}
                assoc = (associations or {}).get(tid, {}) or {}
                source = str(assoc.get("final_assoc", "unknown") or "unknown")
                associated_points = list(assoc.get("points", []) or [])
                geom_pts = len(associated_points)
                selected_points = self._select_points(associated_points)
                if not selected_points and self.write_empty_summary:
                    self._buffer.append(
                        self._build_row(
                            frame=frame,
                            timestamp_s=ts,
                            tid=tid,
                            pose=pose,
                            point=None,
                            source=source,
                            geom_pts=0,
                            points_total=points_total,
                            ground_z=frame_ground_z,
                        )
                    )
                    continue
                for point in selected_points:
                    self._buffer.append(
                        self._build_row(
                            frame=frame,
                            timestamp_s=ts,
                            tid=tid,
                            pose=pose,
                            point=point,
                            source=source,
                            geom_pts=geom_pts,
                            points_total=points_total,
                            ground_z=frame_ground_z,
                        )
                    )
            self._frames_since_flush += 1
            if (
                len(self._buffer) >= self.flush_every_rows
                or self._frames_since_flush >= self.flush_every_frames
            ):
                self.flush()
        except Exception as exc:
            self._record_error(exc)

    def flush(self) -> None:
        if self._closed or not self._buffer:
            return
        try:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()
            self._frames_since_flush = 0
        except Exception as exc:
            self._buffer.clear()
            self._record_error(exc)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._file.close()
            self._closed = True

    def _select_points(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(points) <= self.max_points_per_tid:
            return list(points)

        selected: dict[int, dict[str, Any]] = {}

        def add(point: dict[str, Any]) -> None:
            if len(selected) < self.max_points_per_tid:
                selected[int(point.get("index", len(selected)) or 0)] = point

        for point in sorted(points, key=lambda item: _finite_or(item.get("snr"), -math.inf), reverse=True):
            add(point)
            if len(selected) >= max(1, self.max_points_per_tid // 4):
                break

        for point in sorted(points, key=lambda item: _finite_or(item.get("z"), -math.inf), reverse=True):
            add(point)
            if len(selected) >= max(1, self.max_points_per_tid // 2):
                break

        for point in sorted(points, key=lambda item: _finite_or(item.get("z"), math.inf)):
            add(point)
            if len(selected) >= max(1, (self.max_points_per_tid * 3) // 4):
                break

        remaining = [point for point in points if int(point.get("index", -1) or -1) not in selected]
        if remaining:
            step = max(1, len(remaining) // max(1, self.max_points_per_tid - len(selected)))
            for point in remaining[::step]:
                add(point)
                if len(selected) >= self.max_points_per_tid:
                    break

        if len(selected) < self.max_points_per_tid:
            for point in points:
                add(point)
                if len(selected) >= self.max_points_per_tid:
                    break

        return sorted(selected.values(), key=lambda item: int(item.get("index", 0) or 0))

    def _build_row(
        self,
        *,
        frame: int,
        timestamp_s: float,
        tid: int,
        pose: dict[str, Any],
        point: dict[str, Any] | None,
        source: str,
        geom_pts: int,
        points_total: int,
        ground_z: float,
    ) -> dict[str, Any]:
        tx = _float_or_blank(pose.get("x", pose.get("target_x_m", "")))
        ty = _float_or_blank(pose.get("y", pose.get("target_y_m", "")))
        tz = _float_or_blank(pose.get("z", pose.get("target_z", "")))
        vx = _float_or_blank(pose.get("vx", ""))
        vy = _float_or_blank(pose.get("vy", ""))
        vz = _float_or_blank(pose.get("vz", ""))
        target_range, target_azimuth, target_elevation = _range_angles(tx, ty, tz)

        is_valid = point is not None
        px = _float_or_blank(point.get("x", "") if point else "")
        py = _float_or_blank(point.get("y", "") if point else "")
        pz = _float_or_blank(point.get("z", "") if point else "")
        point_range, point_azimuth, point_elevation = _range_angles(px, py, pz)

        rel_x = _difference(px, tx)
        rel_y = _difference(py, ty)
        rel_z = _difference(pz, tz)
        rel_range, _, _ = _range_angles(rel_x, rel_y, rel_z)
        height_above_ground = _difference(pz, ground_z)
        probs = pose.get("probabilities", {}) or {}

        return {
            "session_id": self.session_id,
            "frame": frame,
            "timestamp_s": timestamp_s,
            "tid": int(tid),
            "track_index": _blank_if_none((point or {}).get("track_index", pose.get("track_index", ""))),
            "point_index": _blank_if_none((point or {}).get("index", "")),
            "association_source": source or "unknown",
            "association_confidence": _association_confidence(source),
            "point_x_m": px,
            "point_y_m": py,
            "point_z_m": pz,
            "point_range_m": point_range,
            "point_azimuth_deg": point_azimuth,
            "point_elevation_deg": point_elevation,
            "point_doppler_mps": _float_or_blank((point or {}).get("doppler", "")),
            "point_snr": _float_or_blank((point or {}).get("snr", "")),
            "point_noise": _float_or_blank((point or {}).get("noise", "")),
            "point_quality": _point_quality(point),
            "target_x_m": tx,
            "target_y_m": ty,
            "target_z_m": tz,
            "target_range_m": target_range,
            "target_azimuth_deg": target_azimuth,
            "target_elevation_deg": target_elevation,
            "target_vx_mps": vx,
            "target_vy_mps": vy,
            "target_vz_mps": vz,
            "relative_x_m": rel_x,
            "relative_y_m": rel_y,
            "relative_z_m": rel_z,
            "relative_range_m": rel_range,
            "relative_radial_m": rel_y,
            "relative_lateral_m": rel_x,
            "height_above_ground_m": height_above_ground,
            "is_valid_point": 1 if is_valid else 0,
            "geom_pts_for_tid": int(geom_pts),
            "points_total_frame": int(points_total),
            "quality_label_for_tid": str(pose.get("quality", "NO_POINTS" if not is_valid else "")),
            "old_display_pose": str(pose.get("displayed_label", pose.get("final_label", ""))),
            "old_model_stand_prob": _probability(probs, "STANDING", pose.get("stand_prob", "")),
            "old_model_sit_prob": _probability(probs, "SITTING", pose.get("sit_prob", "")),
            "old_model_move_prob": _probability(probs, "MOVING", ""),
            "old_model_lie_prob": _probability(probs, "LYING", ""),
            "old_model_fall_prob": _probability(probs, "FALLING", ""),
        }

    def _record_error(self, exc: Exception) -> None:
        self._error_count += 1
        if self.debug and self._error_count <= 5:
            print(f"[associated-point-log] write failed: {exc}", flush=True)


def _range_angles(x: Any, y: Any, z: Any) -> tuple[Any, Any, Any]:
    if not all(_is_number(value) for value in (x, y, z)):
        return "", "", ""
    xf = float(x)
    yf = float(y)
    zf = float(z)
    horizontal = math.sqrt(xf * xf + yf * yf)
    total = math.sqrt(horizontal * horizontal + zf * zf)
    azimuth = math.degrees(math.atan2(xf, yf))
    elevation = math.degrees(math.atan2(zf, horizontal))
    return total, azimuth, elevation


def _difference(left: Any, right: Any) -> Any:
    if not (_is_number(left) and _is_number(right)):
        return ""
    return float(left) - float(right)


def _probability(probabilities: dict[str, Any], key: str, fallback: Any) -> Any:
    value = probabilities.get(key, fallback)
    return _float_or_blank(value)


def _association_confidence(source: str) -> float:
    normalized = str(source or "").lower()
    if normalized in {"target_index", "hybrid_target_index", "index"}:
        return 1.0
    if normalized in {"nearest", "hybrid_nearest"}:
        return 0.5
    if normalized in {"unassociated", "auto_none", "unknown"}:
        return 0.0
    return 0.0


def _point_quality(point: dict[str, Any] | None) -> str:
    if point is None:
        return "NO_POINTS"
    snr = _float_or_blank(point.get("snr", ""))
    if _is_number(snr) and float(snr) <= 0.0:
        return "LOW_SNR"
    return "OK"


def _float_or_blank(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return number


def _finite_or(value: Any, default: float) -> float:
    number = _float_or_blank(value)
    if number == "":
        return default
    return float(number)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value
