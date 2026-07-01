"""Structured logging for combined mmWave/RGB experiment sessions."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def wall_time_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def monotonic_ns() -> int:
    return time.monotonic_ns()


class CombinedSessionLogger:
    def __init__(
        self,
        log_root,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        log_rgb_keypoints: bool = False,
        log_mmwave_points: bool = False,
    ) -> None:
        self.log_root = Path(log_root).expanduser().resolve()
        self.session_id = str(session_id)
        self.session_dir = self.log_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_rgb_keypoints = bool(log_rgb_keypoints)
        self.log_mmwave_points = bool(log_mmwave_points)
        self._files = []
        self._file_by_writer: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._closed = False
        self._last_flush_ns = monotonic_ns()
        self._flush_interval_ns = 1_000_000_000
        self.latest_mmwave_frame_num = None
        self.latest_mmwave_monotonic_ns = None
        self.latest_rgb_frame_num = None
        self.latest_rgb_monotonic_ns = None

        self._events_file = (self.session_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        )
        self._files.append(self._events_file)
        self._init_csv_writers()
        self._write_metadata(metadata or {})
        self.log_event("combined_session_started", {"session_dir": str(self.session_dir)})

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        record = self._base_record("event")
        record.update({"event_type": event_type, "payload": payload or {}})
        try:
            self._events_file.write(json.dumps(record, default=str) + "\n")
            self._events_file.flush()
        except Exception:
            pass

    def log_mmwave_frame(self, frame_record: dict[str, Any]) -> None:
        record = self._with_base(frame_record, "mmwave")
        frame_num = record.get("mmwave_frame_num")
        self._write_row("mmwave_frames", record)
        self.latest_mmwave_frame_num = frame_num
        self.latest_mmwave_monotonic_ns = record.get("host_monotonic_ns")
        self._write_sync("mmwave", frame_num, record.get("host_monotonic_ns"))

    def log_mmwave_track(self, track_record: dict[str, Any]) -> None:
        self._write_row("mmwave_tracks", self._with_base(track_record, "mmwave"))

    def log_mmwave_pose(self, pose_record: dict[str, Any]) -> None:
        self._write_row("mmwave_pose", self._with_base(pose_record, "mmwave"))

    def log_mmwave_point(self, point_record: dict[str, Any]) -> None:
        if self.log_mmwave_points:
            self._write_row("mmwave_points", self._with_base(point_record, "mmwave"))

    def log_rgb_frame(self, frame_record: dict[str, Any]) -> None:
        record = self._with_base(frame_record, "rgb")
        frame_num = record.get("rgb_frame_num")
        self._write_row("rgb_frames", record)
        self.latest_rgb_frame_num = frame_num
        self.latest_rgb_monotonic_ns = record.get("host_monotonic_ns")
        self._write_sync("rgb", frame_num, record.get("host_monotonic_ns"))

    def log_rgb_track(self, track_record: dict[str, Any]) -> None:
        self._write_row("rgb_tracks", self._with_base(track_record, "rgb"))

    def log_rgb_keypoint(self, keypoint_record: dict[str, Any]) -> None:
        if self.log_rgb_keypoints:
            self._write_row("rgb_keypoints", self._with_base(keypoint_record, "rgb"))

    def log_rgb_action(self, action_record: dict[str, Any]) -> None:
        self._write_row("rgb_actions", self._with_base(action_record, "rgb"))

    def close(self) -> None:
        if self._closed:
            return
        self.log_event("combined_session_stopped", {"session_dir": str(self.session_dir)})
        self._closed = True
        for handle in self._files:
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    def _init_csv_writers(self) -> None:
        self._writer(
            "mmwave_frames",
            "mmwave_frames.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "mmwave_frame_num",
                "num_tracks",
                "num_points",
                "parse_ok",
                "error_count",
            ],
        )
        self._writer(
            "mmwave_tracks",
            "mmwave_tracks.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "mmwave_frame_num",
                "tid",
                "x_m",
                "y_m",
                "z_m",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "confidence",
                "g",
                "num_associated_points",
                "height_min_z_m",
                "height_max_z_m",
                "height_m",
            ],
        )
        self._writer(
            "mmwave_pose",
            "mmwave_pose.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "mmwave_frame_num",
                "tid",
                "window_ready",
                "ml_label",
                "ml_confidence",
                "final_label",
                "motion_label",
                "speed_mps",
                "height_drop_flag",
                "quality_flag",
                "num_points",
                "prob_standing",
                "prob_sitting",
                "prob_lying",
                "prob_falling",
            ],
        )
        if self.log_mmwave_points:
            self._writer(
                "mmwave_points",
                "mmwave_points.csv",
                [
                    "session_id",
                    "host_wall_time_iso",
                    "host_monotonic_ns",
                    "source",
                    "mmwave_frame_num",
                    "point_index",
                    "track_index",
                    "x_m",
                    "y_m",
                    "z_m",
                    "doppler",
                    "snr",
                    "noise",
                ],
            )
        self._writer(
            "rgb_frames",
            "rgb_frames.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "rgb_frame_num",
                "width",
                "height",
                "fps_estimate",
                "frame_read_ok",
                "num_detections",
                "num_tracks",
                "num_actions",
                "error_count",
            ],
        )
        self._writer(
            "rgb_tracks",
            "rgb_tracks.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "rgb_frame_num",
                "rgb_track_id",
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
                "bbox_confidence",
                "pose_confidence",
                "tracker_state",
                "track_age",
                "time_since_update",
                "action_window_ready",
                "action_label",
                "action_confidence",
            ],
        )
        if self.log_rgb_keypoints:
            self._writer(
                "rgb_keypoints",
                "rgb_keypoints.csv",
                [
                    "session_id",
                    "host_wall_time_iso",
                    "host_monotonic_ns",
                    "source",
                    "rgb_frame_num",
                    "rgb_track_id",
                    "joint_index",
                    "x_px",
                    "y_px",
                    "score",
                    "x_norm",
                    "y_norm",
                ],
            )
        self._writer(
            "rgb_actions",
            "rgb_actions.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "rgb_frame_num",
                "rgb_track_id",
                "action_window_ready",
                "action_label",
                "action_confidence",
                "prob_standing",
                "prob_walking",
                "prob_sitting",
                "prob_lying_down",
                "prob_stand_up",
                "prob_sit_down",
                "prob_fall_down",
            ],
        )
        self._writer(
            "sync_index",
            "sync_index.csv",
            [
                "session_id",
                "host_wall_time_iso",
                "host_monotonic_ns",
                "source",
                "source_type",
                "source_frame_num",
                "latest_mmwave_frame_num",
                "latest_mmwave_monotonic_ns",
                "latest_rgb_frame_num",
                "latest_rgb_monotonic_ns",
                "delta_ms_if_both_available",
            ],
        )

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("created_wall_time_iso", wall_time_iso())
        payload.setdefault("created_monotonic_ns", monotonic_ns())
        with (self.session_dir / "session_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")

    def _writer(self, key: str, filename: str, fieldnames: list[str]) -> None:
        handle = (self.session_dir / filename).open("w", newline="", encoding="utf-8")
        self._files.append(handle)
        self._file_by_writer[key] = handle
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        self._writers[key] = writer

    def _write_row(self, writer_key: str, record: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            writer = self._writers[writer_key]
            writer.writerow({key: self._clean(value) for key, value in record.items()})
            self._flush_writer(writer_key, periodic=True)
        except Exception as exc:
            self.log_event(
                "logger_error",
                {"writer": writer_key, "error": str(exc), "record": record},
            )

    def _write_sync(self, source_type: str, source_frame_num, source_monotonic_ns) -> None:
        delta_ms = ""
        if (
            self.latest_mmwave_monotonic_ns is not None
            and self.latest_rgb_monotonic_ns is not None
        ):
            delta_ms = (
                int(self.latest_rgb_monotonic_ns) - int(self.latest_mmwave_monotonic_ns)
            ) / 1_000_000.0
        self._write_row(
            "sync_index",
            {
                "session_id": self.session_id,
                "host_wall_time_iso": wall_time_iso(),
                "host_monotonic_ns": source_monotonic_ns or monotonic_ns(),
                "source": source_type,
                "source_type": source_type,
                "source_frame_num": source_frame_num,
                "latest_mmwave_frame_num": self.latest_mmwave_frame_num,
                "latest_mmwave_monotonic_ns": self.latest_mmwave_monotonic_ns,
                "latest_rgb_frame_num": self.latest_rgb_frame_num,
                "latest_rgb_monotonic_ns": self.latest_rgb_monotonic_ns,
                "delta_ms_if_both_available": delta_ms,
            },
        )

    def _flush_writer(self, writer_key: str, periodic: bool = False) -> None:
        now_ns = monotonic_ns()
        if periodic and now_ns - self._last_flush_ns < self._flush_interval_ns:
            return
        handle = self._file_by_writer.get(writer_key)
        if handle is not None:
            try:
                handle.flush()
                self._last_flush_ns = now_ns
            except Exception:
                pass
            return
        for file_handle in self._files:
            try:
                file_handle.flush()
            except Exception:
                pass
        self._last_flush_ns = now_ns

    def _base_record(self, source: str) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "host_wall_time_iso": wall_time_iso(),
            "host_monotonic_ns": monotonic_ns(),
            "source": source,
        }

    def _with_base(self, record: dict[str, Any], source: str) -> dict[str, Any]:
        combined = self._base_record(source)
        combined.update(record or {})
        combined.setdefault("session_id", self.session_id)
        combined.setdefault("source", source)
        return combined

    @staticmethod
    def _clean(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, default=str)
        return value
