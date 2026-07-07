"""Live pose integration for the vendored TI-style visualizer.

This module does not open serial ports. It receives parsed TI Visualizer
``outputDict`` frames, builds one independent 8-frame Pose/Fall feature window
per tracker TID, runs ONNX inference, and optionally logs per-TID predictions.

The ONNX model was trained on TI IWRL6432 Pose/Fall data. Live IWR6843ISK-ODS
performance must be validated before treating labels as reliable.
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import pose_feature_extractor as features
from pose_model_runtime import CLASS_NAMES as DEFAULT_CLASS_NAMES, PoseModelRuntime, PoseSmoother


UNASSOCIATED_TRACK_INDEXES = {253, 254, 255}
STALE_TRACK_FRAMES = 30
POSE_MIN_CONFIDENCE_DEFAULTS = {
    "STANDING": 0.70,
    "SITTING": 0.45,
    "LYING": 0.60,
    "FALLING": 0.70,
    "MOVING": 0.35,
    "UNKNOWN": 0.00,
}
POSE_STABILITY_FRAME_DEFAULTS = {
    "STANDING": 12,
    "SITTING": 8,
    "LYING": 14,
    "FALLING": 4,
    "MOVING": 4,
    "UNKNOWN": 6,
}
STAND_SIT_MARGIN_DEFAULTS = {
    "near": 0.06,
    "mid": 0.10,
    "far": 0.15,
}
STAND_TO_SIT_FRAME_DEFAULTS = {
    "near": 6,
    "mid": 8,
    "far": 12,
}
SIT_TO_STAND_FRAME_DEFAULTS = {
    "near": 8,
    "mid": 10,
    "far": 14,
}
MOVING_OVERRIDE_FRAME_DEFAULTS = {
    "near": 3,
    "mid": 4,
    "far": 5,
}
STRONG_STAND_SIT_MARGIN_DEFAULTS = {
    "near": 0.12,
    "mid": 0.18,
    "far": 0.25,
}
SITTING_DROP_DEFAULTS = {
    "near": 0.20,
    "mid": 0.25,
    "far": 0.35,
}
STAND_TO_SIT_GATE_MIN_CONFIDENCE_DEFAULT = 0.65
STAND_TO_SIT_GATE_MARGIN_DEFAULT = 0.15
STAND_TO_SIT_GATE_FRAMES_DEFAULT = 12
SITTING_RELATIVE_GATE_ENABLED_DEFAULT = True
SITTING_RELATIVE_RANGE_MIN_M_DEFAULT = 3.0
SITTING_RELATIVE_MIN_PROB_DEFAULT = 0.55
SITTING_RELATIVE_MARGIN_DEFAULT = 0.12
SITTING_RELATIVE_FRAMES_DEFAULT = 16
SITTING_RELATIVE_STANDING_VETO_PROB_DEFAULT = 0.50
SITTING_RELATIVE_STANDING_VETO_MARGIN_DEFAULT = 0.05
MOVING_OVERRIDE_REQUIRE_BODY_TRANSLATION_FOR_SITTING_DEFAULT = True
SIT_TO_STAND_RECOVERY_MARGIN_DEFAULT = 0.10
SIT_TO_STAND_RECOVERY_FRAMES_DEFAULT = 6
ASSOC_METHODS = {"auto", "target_index", "nearest", "hybrid"}
HUMAN_MODEL_STATE_PROVISIONAL = "PROVISIONAL"
HUMAN_MODEL_STATE_CONFIRMED = "CONFIRMED"
HUMAN_MODEL_STATE_SUSPECT = "SUSPECT_GHOST"
HUMAN_MODEL_STATE_LOST = "LOST"
HUMAN_MODEL_BAD_QUALITIES = {"NO_POINTS", "TARGET_ONLY"}
HUMAN_MODEL_WEAK_QUALITIES = {"NO_POINTS", "TARGET_ONLY", "LOW_POINTS"}


class TiStylePoseManager:
    def __init__(
        self,
        model_path,
        smoothing_window: int = 7,
        min_confidence: float = 0.55,
        unknown_confidence: float = 0.45,
        moving_speed_threshold: float = 0.18,
        moving_confirm_frames: int = 4,
        fall_height_drop_threshold: float = 0.35,
        fall_vertical_speed_threshold: float = 0.35,
        fall_high_confidence: float = 0.85,
        fall_min_height_drop_with_high_confidence: float = 0.20,
        fall_stability_frames: int | None = None,
        display_stability_frames: int = 16,
        display_min_confidence: float = 0.55,
        display_hysteresis: bool = True,
        display_stability_ratio: float = 0.70,
        falling_fast_update: bool = True,
        falling_stability_frames: int = POSE_STABILITY_FRAME_DEFAULTS["FALLING"],
        sitting_stability_frames: int = 8,
        sitting_stability_ratio: float = 0.50,
        sitting_min_confidence: float = POSE_MIN_CONFIDENCE_DEFAULTS["SITTING"],
        sitting_max_speed: float = 0.25,
        standing_min_confidence: float = POSE_MIN_CONFIDENCE_DEFAULTS["STANDING"],
        lying_min_confidence: float = POSE_MIN_CONFIDENCE_DEFAULTS["LYING"],
        falling_min_confidence: float = POSE_MIN_CONFIDENCE_DEFAULTS["FALLING"],
        moving_min_confidence: float = POSE_MIN_CONFIDENCE_DEFAULTS["MOVING"],
        standing_stability_frames: int = POSE_STABILITY_FRAME_DEFAULTS["STANDING"],
        lying_stability_frames: int = POSE_STABILITY_FRAME_DEFAULTS["LYING"],
        moving_stability_frames: int = POSE_STABILITY_FRAME_DEFAULTS["MOVING"],
        unknown_stability_frames: int = POSE_STABILITY_FRAME_DEFAULTS["UNKNOWN"],
        range_near_max: float = 2.0,
        range_mid_max: float = 4.0,
        stand_sit_near_margin: float = STAND_SIT_MARGIN_DEFAULTS["near"],
        stand_sit_mid_margin: float = STAND_SIT_MARGIN_DEFAULTS["mid"],
        stand_sit_far_margin: float = STAND_SIT_MARGIN_DEFAULTS["far"],
        stand_to_sit_near_frames: int = STAND_TO_SIT_FRAME_DEFAULTS["near"],
        stand_to_sit_mid_frames: int = STAND_TO_SIT_FRAME_DEFAULTS["mid"],
        stand_to_sit_far_frames: int = STAND_TO_SIT_FRAME_DEFAULTS["far"],
        sit_to_stand_near_frames: int = SIT_TO_STAND_FRAME_DEFAULTS["near"],
        sit_to_stand_mid_frames: int = SIT_TO_STAND_FRAME_DEFAULTS["mid"],
        sit_to_stand_far_frames: int = SIT_TO_STAND_FRAME_DEFAULTS["far"],
        moving_override_near_frames: int = MOVING_OVERRIDE_FRAME_DEFAULTS["near"],
        moving_override_mid_frames: int = MOVING_OVERRIDE_FRAME_DEFAULTS["mid"],
        moving_override_far_frames: int = MOVING_OVERRIDE_FRAME_DEFAULTS["far"],
        strong_stand_sit_near_margin: float = STRONG_STAND_SIT_MARGIN_DEFAULTS["near"],
        strong_stand_sit_mid_margin: float = STRONG_STAND_SIT_MARGIN_DEFAULTS["mid"],
        strong_stand_sit_far_margin: float = STRONG_STAND_SIT_MARGIN_DEFAULTS["far"],
        moving_require_translation: bool = True,
        moving_translation_window: int = 8,
        moving_translation_min_m: float = 0.25,
        sensor_height_m: float = 1.25,
        sensor_pitch_deg: float = 0.0,
        sensor_roll_deg: float = 0.0,
        sensor_yaw_deg: float = 0.0,
        use_sensor_calibration: bool = False,
        floor_z_m: float = 0.0,
        assoc_debug: bool = False,
        assoc_method: str = "auto",
        assoc_nearest_radius_m: float = 0.75,
        assoc_nearest_z_min: float = -0.5,
        assoc_nearest_z_max: float = 2.5,
        assoc_min_points_good: int = 3,
        use_standing_baseline: bool = False,
        standing_baseline_min_frames: int = 20,
        sitting_drop_near_m: float = SITTING_DROP_DEFAULTS["near"],
        sitting_drop_mid_m: float = SITTING_DROP_DEFAULTS["mid"],
        sitting_drop_far_m: float = SITTING_DROP_DEFAULTS["far"],
        sitting_drop_min_sit_prob: float = 0.30,
        sitting_drop_centroid_m: float = 0.25,
        sitting_drop_top_m: float = 0.25,
        sitting_drop_target_z_m: float = 0.20,
        stand_to_sit_min_confidence: float = STAND_TO_SIT_GATE_MIN_CONFIDENCE_DEFAULT,
        stand_to_sit_margin: float = STAND_TO_SIT_GATE_MARGIN_DEFAULT,
        stand_to_sit_frames: int = STAND_TO_SIT_GATE_FRAMES_DEFAULT,
        stand_to_sit_allow_target_only: bool = False,
        sitting_relative_gate: bool = SITTING_RELATIVE_GATE_ENABLED_DEFAULT,
        sitting_relative_range_min_m: float = SITTING_RELATIVE_RANGE_MIN_M_DEFAULT,
        sitting_relative_min_prob: float = SITTING_RELATIVE_MIN_PROB_DEFAULT,
        sitting_relative_margin: float = SITTING_RELATIVE_MARGIN_DEFAULT,
        sitting_relative_frames: int = SITTING_RELATIVE_FRAMES_DEFAULT,
        sitting_relative_standing_veto_prob: float = (
            SITTING_RELATIVE_STANDING_VETO_PROB_DEFAULT
        ),
        sitting_relative_standing_veto_margin: float = (
            SITTING_RELATIVE_STANDING_VETO_MARGIN_DEFAULT
        ),
        moving_override_require_body_translation_for_sitting: bool = (
            MOVING_OVERRIDE_REQUIRE_BODY_TRANSLATION_FOR_SITTING_DEFAULT
        ),
        sit_to_stand_recovery_margin: float = SIT_TO_STAND_RECOVERY_MARGIN_DEFAULT,
        sit_to_stand_recovery_frames: int = SIT_TO_STAND_RECOVERY_FRAMES_DEFAULT,
        min_associated_points_for_inference: int = 1,
        allow_target_only: bool = False,
        enable_3d_labels: bool = False,
        label_format: str = "{tid} | {final_label} {confidence_percent}%",
        label_z_offset: float = 0.35,
        label_min_confidence: float = 0.45,
        label_max_distance: float | None = None,
        label_debug: bool = False,
        enable_human_models: bool = False,
        human_model_debug: bool = False,
        human_model_stale_frames: int = 10,
        human_model_ghost_distance_m: float = 0.75,
        human_model_confirm_frames: int = 5,
        human_model_confirm_min_geom_pts: int = 3,
        human_model_confirm_min_quality_frames: int = 3,
        human_model_confirmed_grace_frames: int = 30,
        human_model_bad_evidence_demote_frames: int = 60,
        human_model_ghost_min_bad_frames: int = 8,
        human_model_ghost_no_points_frames: int = 8,
        human_model_show_provisional: bool = False,
        human_model_show_suspect: bool = False,
        ground_z: float = 0.0,
        human_model_target_height: float = 1.70,
        human_model_target_sitting_height: float = 1.20,
        human_model_target_lying_length: float = 1.70,
        debug: bool = False,
        log_dir=None,
        cfg_path=None,
        cli_port: str | None = None,
        data_port: str | None = None,
        allow_missing_scaler: bool = False,
    ):
        self.model = PoseModelRuntime(
            model_path,
            allow_missing_scaler=allow_missing_scaler,
            debug=debug,
        )
        self.class_names = list(getattr(self.model, "class_names", DEFAULT_CLASS_NAMES))
        self.smoother = PoseSmoother(smoothing_window, class_names=self.class_names)
        self.smoothing_window = max(1, int(smoothing_window))
        self.min_confidence = float(min_confidence)
        self.unknown_confidence = float(unknown_confidence)
        self.moving_speed_threshold = float(moving_speed_threshold)
        self.moving_confirm_frames = max(1, int(moving_confirm_frames))
        self.fall_height_drop_threshold = float(fall_height_drop_threshold)
        self.fall_vertical_speed_threshold = float(fall_vertical_speed_threshold)
        self.fall_high_confidence = float(fall_high_confidence)
        self.fall_min_height_drop_with_high_confidence = float(
            fall_min_height_drop_with_high_confidence
        )
        self.display_stability_frames = max(1, int(display_stability_frames))
        self.display_min_confidence = float(display_min_confidence)
        self.display_hysteresis = bool(display_hysteresis)
        self.display_stability_ratio = max(0.0, min(1.0, float(display_stability_ratio)))
        self.falling_fast_update = bool(falling_fast_update)
        if fall_stability_frames is None:
            fall_stability_frames = falling_stability_frames
        self.fall_stability_frames = max(1, int(fall_stability_frames))
        self.falling_stability_frames = self.fall_stability_frames
        self.sitting_stability_frames = max(1, int(sitting_stability_frames))
        self.sitting_stability_ratio = max(0.0, min(1.0, float(sitting_stability_ratio)))
        self.sitting_min_confidence = float(sitting_min_confidence)
        self.sitting_max_speed = float(sitting_max_speed)
        self.standing_min_confidence = float(standing_min_confidence)
        self.lying_min_confidence = float(lying_min_confidence)
        self.falling_min_confidence = float(falling_min_confidence)
        self.moving_min_confidence = float(moving_min_confidence)
        self.unknown_min_confidence = POSE_MIN_CONFIDENCE_DEFAULTS["UNKNOWN"]
        self.standing_stability_frames = max(1, int(standing_stability_frames))
        self.lying_stability_frames = max(1, int(lying_stability_frames))
        self.moving_stability_frames = max(1, int(moving_stability_frames))
        self.unknown_stability_frames = max(1, int(unknown_stability_frames))
        self.pose_min_confidence_by_pose = {
            "STANDING": self.standing_min_confidence,
            "SITTING": self.sitting_min_confidence,
            "LYING": self.lying_min_confidence,
            "FALLING": self.falling_min_confidence,
            "MOVING": self.moving_min_confidence,
            "UNKNOWN": self.unknown_min_confidence,
        }
        self.pose_stability_frames_by_pose = {
            "STANDING": self.standing_stability_frames,
            "SITTING": self.sitting_stability_frames,
            "LYING": self.lying_stability_frames,
            "FALLING": self.fall_stability_frames,
            "MOVING": self.moving_stability_frames,
            "UNKNOWN": self.unknown_stability_frames,
        }
        self.range_near_max = float(range_near_max)
        self.range_mid_max = max(self.range_near_max, float(range_mid_max))
        self.stand_sit_margin_by_zone = {
            "near": float(stand_sit_near_margin),
            "mid": float(stand_sit_mid_margin),
            "far": float(stand_sit_far_margin),
        }
        self.stand_to_sit_frames_by_zone = {
            "near": max(1, int(stand_to_sit_near_frames)),
            "mid": max(1, int(stand_to_sit_mid_frames)),
            "far": max(1, int(stand_to_sit_far_frames)),
        }
        self.sit_to_stand_frames_by_zone = {
            "near": max(1, int(sit_to_stand_near_frames)),
            "mid": max(1, int(sit_to_stand_mid_frames)),
            "far": max(1, int(sit_to_stand_far_frames)),
        }
        self.unknown_to_stand_sit_frames_by_zone = dict(
            self.stand_to_sit_frames_by_zone
        )
        self.moving_override_frames_by_zone = {
            "near": max(1, int(moving_override_near_frames)),
            "mid": max(1, int(moving_override_mid_frames)),
            "far": max(1, int(moving_override_far_frames)),
        }
        self.strong_stand_sit_margin_by_zone = {
            "near": float(strong_stand_sit_near_margin),
            "mid": float(strong_stand_sit_mid_margin),
            "far": float(strong_stand_sit_far_margin),
        }
        self.moving_require_translation = bool(moving_require_translation)
        self.moving_translation_window = max(2, int(moving_translation_window))
        self.moving_translation_min_m = float(moving_translation_min_m)
        self.sensor_height_m = float(sensor_height_m)
        self.sensor_pitch_deg = float(sensor_pitch_deg)
        self.sensor_roll_deg = float(sensor_roll_deg)
        self.sensor_yaw_deg = float(sensor_yaw_deg)
        self.use_sensor_calibration = bool(use_sensor_calibration)
        self.floor_z_m = float(floor_z_m)
        self.assoc_debug = bool(assoc_debug)
        assoc_method = str(assoc_method or "auto").lower()
        self.assoc_method = assoc_method if assoc_method in ASSOC_METHODS else "auto"
        self.assoc_nearest_radius_m = max(0.0, float(assoc_nearest_radius_m))
        self.assoc_nearest_z_min = float(assoc_nearest_z_min)
        self.assoc_nearest_z_max = float(assoc_nearest_z_max)
        self.assoc_min_points_good = max(1, int(assoc_min_points_good))
        self.use_standing_baseline = bool(use_standing_baseline)
        self.standing_baseline_min_frames = max(1, int(standing_baseline_min_frames))
        self.sitting_drop_by_zone = {
            "near": float(sitting_drop_near_m),
            "mid": float(sitting_drop_mid_m),
            "far": float(sitting_drop_far_m),
        }
        self.sitting_drop_min_sit_prob = float(sitting_drop_min_sit_prob)
        self.sitting_drop_centroid_m = float(sitting_drop_centroid_m)
        self.sitting_drop_top_m = float(sitting_drop_top_m)
        self.sitting_drop_target_z_m = float(sitting_drop_target_z_m)
        self.stand_to_sit_min_confidence = float(stand_to_sit_min_confidence)
        self.stand_to_sit_margin = float(stand_to_sit_margin)
        self.stand_to_sit_frames = max(1, int(stand_to_sit_frames))
        self.stand_to_sit_allow_target_only = bool(stand_to_sit_allow_target_only)
        self.sitting_relative_gate = bool(sitting_relative_gate)
        self.sitting_relative_range_min_m = max(0.0, float(sitting_relative_range_min_m))
        self.sitting_relative_min_prob = float(sitting_relative_min_prob)
        self.sitting_relative_margin = float(sitting_relative_margin)
        self.sitting_relative_frames = max(1, int(sitting_relative_frames))
        self.sitting_relative_standing_veto_prob = float(
            sitting_relative_standing_veto_prob
        )
        self.sitting_relative_standing_veto_margin = float(
            sitting_relative_standing_veto_margin
        )
        self.moving_override_require_body_translation_for_sitting = bool(
            moving_override_require_body_translation_for_sitting
        )
        self.sit_to_stand_recovery_margin = float(sit_to_stand_recovery_margin)
        self.sit_to_stand_recovery_frames = max(1, int(sit_to_stand_recovery_frames))
        self.min_associated_points_for_inference = max(
            0, int(min_associated_points_for_inference)
        )
        self.allow_target_only = bool(allow_target_only)
        self.enable_3d_labels = bool(enable_3d_labels)
        self.label_format = str(label_format)
        self.label_z_offset = float(label_z_offset)
        self.label_min_confidence = float(label_min_confidence)
        self.label_max_distance = (
            None if label_max_distance is None else float(label_max_distance)
        )
        self.label_debug = bool(label_debug)
        self.enable_human_models = bool(enable_human_models)
        self.human_model_debug = bool(human_model_debug)
        self.human_model_stale_frames = max(0, int(human_model_stale_frames))
        self.human_model_ghost_distance_m = max(0.0, float(human_model_ghost_distance_m))
        self.human_model_confirm_frames = max(1, int(human_model_confirm_frames))
        self.human_model_confirm_min_geom_pts = max(
            0, int(human_model_confirm_min_geom_pts)
        )
        self.human_model_confirm_min_quality_frames = max(
            1, int(human_model_confirm_min_quality_frames)
        )
        self.human_model_confirmed_grace_frames = max(
            0, int(human_model_confirmed_grace_frames)
        )
        self.human_model_bad_evidence_demote_frames = max(
            0, int(human_model_bad_evidence_demote_frames)
        )
        self.human_model_ghost_min_bad_frames = max(1, int(human_model_ghost_min_bad_frames))
        self.human_model_ghost_no_points_frames = max(
            1, int(human_model_ghost_no_points_frames)
        )
        self.human_model_show_provisional = bool(human_model_show_provisional)
        self.human_model_show_suspect = bool(human_model_show_suspect)
        self.ground_z = float(ground_z)
        self.human_model_target_height = float(human_model_target_height)
        self.human_model_target_sitting_height = float(human_model_target_sitting_height)
        self.human_model_target_lying_length = float(human_model_target_lying_length)
        self.debug = bool(debug)
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.last_seen_frame: dict[int, int] = {}
        self.latest_results: dict[int, dict] = {}
        self.speed_history = defaultdict(lambda: deque(maxlen=self.moving_confirm_frames))
        self.height_history = defaultdict(lambda: deque(maxlen=8))
        display_history_len = max(
            self.display_stability_frames,
            *self.pose_stability_frames_by_pose.values(),
            *self.stand_to_sit_frames_by_zone.values(),
            *self.sit_to_stand_frames_by_zone.values(),
            *self.moving_override_frames_by_zone.values(),
            self.stand_to_sit_frames,
            self.sitting_relative_frames,
            self.sit_to_stand_recovery_frames,
        )
        self.display_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.moving_override_history = defaultdict(
            lambda: deque(maxlen=max(self.moving_override_frames_by_zone.values()))
        )
        self.stand_to_sit_gate_history = defaultdict(
            lambda: deque(maxlen=self.stand_to_sit_frames)
        )
        self.sitting_relative_gate_history = defaultdict(
            lambda: deque(maxlen=self.sitting_relative_frames)
        )
        self.sit_to_stand_recovery_history = defaultdict(
            lambda: deque(maxlen=self.sit_to_stand_recovery_frames)
        )
        self.display_state: dict[int, dict[str, Any]] = {}
        self.human_model_validation: dict[int, dict[str, Any]] = {}
        self._last_human_ui_summary: dict[str, Any] = {
            "frame": 0,
            "active": 0,
            "confirmed": 0,
            "provisional": 0,
            "suspect": 0,
            "rendered": 0,
            "stale": 0,
        }
        self.raw_label_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.confidence_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.probability_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.position_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.velocity_history = defaultdict(lambda: deque(maxlen=display_history_len))
        self.translation_history = defaultdict(
            lambda: deque(maxlen=self.moving_translation_window)
        )
        self.standing_baseline: dict[int, dict[str, Any]] = {}
        self._last_assoc_debug_summary_frame = -999999
        self._log_file = None
        self._log_writer = None
        self._log_path: Path | None = None

        features.reset_all()
        if log_dir is not None:
            self._init_logging(log_dir, cfg_path, cli_port, data_port)

    def process_output_dict(self, output_dict: dict[str, Any] | None) -> dict[int, dict]:
        if not isinstance(output_dict, dict):
            return {}

        frame_num = _int_value(output_dict.get("frameNum"), 0)
        tracks = _rows(output_dict.get("trackData"))
        points = _rows(output_dict.get("pointCloud"))
        track_indexes = _flat_values(output_dict.get("trackIndexes"))
        target_heights = _height_by_tid(output_dict.get("heightData"))
        track_index_to_tid = {
            index: int(_float_at(track, 0))
            for index, track in enumerate(tracks)
            if len(track) > 0
        }

        results: dict[int, dict] = {}
        seen_tids: set[int] = set()
        for track in tracks:
            if len(track) < 4:
                continue
            target = self._track_to_target(track)
            tid = int(target["tid"])
            seen_tids.add(tid)
            self.last_seen_frame[tid] = frame_num

            vx = float(target["vel_x"])
            vy = float(target["vel_y"])
            vz = float(target["vel_z"])
            horizontal_speed = math.sqrt(vx * vx + vy * vy)
            vertical_speed = abs(vz)
            range_m = math.sqrt(
                float(target["pos_x"]) * float(target["pos_x"])
                + float(target["pos_y"]) * float(target["pos_y"])
            )
            range_zone = self._range_zone(range_m)
            motion_state = self._update_motion_state(tid, horizontal_speed)
            height_drop = self._update_height_drop(tid, float(target["pos_z"]))

            cal_target = self._calibrated_point(
                float(target["pos_x"]),
                float(target["pos_y"]),
                float(target["pos_z"]),
            )
            assoc = self._associate_points(
                tid=tid,
                target=target,
                points=points,
                track_indexes=track_indexes,
                track_index_to_tid=track_index_to_tid,
                frame_num=frame_num,
                tracks_total=len(tracks),
            )
            associated_points = assoc["points"]
            geometry = self._point_geometry(
                associated_points=associated_points,
                target=target,
                cal_target=cal_target,
            )
            build_result = features.build_22_feature_vector(target, associated_points)
            num_points = int(build_result.num_points)
            can_use_frame = self._can_use_frame_for_inference(num_points)
            if can_use_frame:
                features.update_8_frame_window(
                    tid, build_result.feature22, build_result.quality
                )
            window_age = features.get_window_age(tid)
            window_ready = features.is_window_ready(tid)

            raw_label = "WARMUP"
            raw_confidence = 0.0
            smoothed_label = "WARMUP"
            smoothed_confidence = 0.0
            probabilities = {name: 0.0 for name in self.class_names}
            prediction_exists = False

            if window_ready and can_use_frame:
                vector176 = features.build_176_feature_vector(tid)
                raw = self.model.predict(vector176)
                raw_label = raw["predicted_label"]
                raw_confidence = float(raw["confidence"])
                smoothed = self.smoother.update(tid, raw["probabilities"])
                smoothed_label = smoothed["smoothed_label"]
                smoothed_confidence = float(smoothed["smoothed_confidence"])
                probabilities = smoothed["smoothed_probabilities"]
                prediction_exists = True

            quality = self._quality_label(
                window_ready=window_ready,
                prediction_exists=prediction_exists,
                can_use_frame=can_use_frame,
                num_points=num_points,
                smoothed_confidence=smoothed_confidence,
            )
            low_quality = quality in {"LOW_POINTS", "NO_POINTS", "LOW_QUALITY"}
            self._update_tuning_histories(
                tid=tid,
                raw_label=raw_label,
                confidence=smoothed_confidence,
                probabilities=probabilities,
                position=(
                    float(target["pos_x"]),
                    float(target["pos_y"]),
                    float(target["pos_z"]),
                ),
                velocity=(vx, vy, vz),
            )
            fall_gate_passed, fall_gate_reason = self._evaluate_fall_gate(
                tid=tid,
                smoothed_label=smoothed_label,
                smoothed_confidence=smoothed_confidence,
                probabilities=probabilities,
                height_drop=height_drop,
                vertical_speed=vertical_speed,
            )
            sitting_gate_passed, sitting_gate_reason = self._evaluate_sitting_gate(
                smoothed_label=smoothed_label,
                smoothed_confidence=smoothed_confidence,
                probabilities=probabilities,
                horizontal_speed=horizontal_speed,
            )
            stand_sit = self._resolve_stand_sit(
                tid=tid,
                prediction_exists=prediction_exists,
                smoothed_label=smoothed_label,
                smoothed_confidence=smoothed_confidence,
                probabilities=probabilities,
                range_m=range_m,
                range_zone=range_zone,
                geometry=geometry,
                horizontal_speed=horizontal_speed,
                motion_state=motion_state,
            )
            candidate_label = self._final_label(
                window_ready=window_ready,
                prediction_exists=prediction_exists,
                smoothed_label=smoothed_label,
                smoothed_confidence=smoothed_confidence,
                motion_state=motion_state,
                height_drop=height_drop,
                horizontal_speed=horizontal_speed,
                vertical_speed=vertical_speed,
                probabilities=probabilities,
                fall_gate_passed=fall_gate_passed,
                sitting_gate_passed=sitting_gate_passed,
                stand_sit=stand_sit,
            )
            candidate_confidence = (
                float(stand_sit.get("candidate_confidence", smoothed_confidence))
                if prediction_exists and bool(stand_sit.get("active"))
                else smoothed_confidence
                if prediction_exists
                else 0.0
            )
            display = self._update_display_state(
                tid=tid,
                candidate_label=candidate_label,
                candidate_confidence=candidate_confidence,
                window_ready=window_ready,
                prediction_exists=prediction_exists,
                horizontal_speed=horizontal_speed,
                motion_state=motion_state,
                raw_label=raw_label,
                smoothed_label=smoothed_label,
                target_position=(
                    float(target["pos_x"]),
                    float(target["pos_y"]),
                    float(target["pos_z"]),
                ),
                range_m=range_m,
                range_zone=range_zone,
                quality=quality,
                geom_quality=str(geometry.get("geom_quality", "")),
                stand_sit=stand_sit,
                fall_gate_passed=fall_gate_passed,
                fall_gate_reason=fall_gate_reason,
                sitting_gate_passed=sitting_gate_passed,
                sitting_gate_reason=sitting_gate_reason,
            )
            displayed_label = display["displayed_label"]
            displayed_confidence = display["displayed_confidence"]
            self._update_standing_baseline(
                tid=tid,
                displayed_label=displayed_label,
                stand_sit=stand_sit,
                geometry=geometry,
                target=target,
                range_m=range_m,
                horizontal_speed=horizontal_speed,
                motion_state=motion_state,
            )

            result = {
                "tid": tid,
                "window_ready": window_ready,
                "window_count": window_age,
                "window_age": window_age,
                "num_points": num_points,
                "selected_num_points": num_points,
                "low_quality": low_quality,
                "quality": quality,
                "reason": quality if quality != "OK" else "",
                "raw_label": raw_label,
                "raw_confidence": raw_confidence,
                "ml_top_label": smoothed_label,
                "ml_top_confidence": smoothed_confidence,
                "smoothed_label": smoothed_label,
                "smoothed_confidence": smoothed_confidence,
                "candidate_confidence": candidate_confidence,
                "final_confidence": displayed_confidence,
                "probabilities": probabilities,
                "below_min_confidence": (
                    prediction_exists and smoothed_confidence < self.min_confidence
                ),
                "prediction_exists": prediction_exists,
                "motion_state": motion_state,
                "candidate_label": candidate_label,
                "pre_display_final_label": candidate_label,
                "final_label": displayed_label,
                "displayed_label": displayed_label,
                "displayed_confidence": displayed_confidence,
                "display_stability_count": display["display_stability_count"],
                "display_stability_required": display["display_stability_required"],
                "display_stability_ratio": display["display_stability_ratio"],
                "candidate_stable_count": display["candidate_stable_count"],
                "pose_min_confidence": display["pose_min_confidence"],
                "pose_required_frames": display["pose_required_frames"],
                "range_m": range_m,
                "range_zone": range_zone,
                "stand_prob": stand_sit["standing_prob"],
                "sit_prob": stand_sit["sitting_prob"],
                "sit_minus_stand_margin": display.get(
                    "sit_minus_stand_margin",
                    stand_sit["sitting_prob"] - stand_sit["standing_prob"],
                ),
                "stand_sit_margin": stand_sit["margin"],
                "stand_sit_zone": stand_sit["range_zone"],
                "stand_sit_decision": stand_sit["decision"],
                "stand_sit_reason": stand_sit.get("reason", ""),
                "stand_sit_strong": display.get("strong_stand_sit", False),
                "stand_sit_strong_margin": display.get("strong_stand_sit_margin", 0.0),
                "stand_sit_required_frames": display["stand_sit_required_frames"],
                "stand_sit_stable_count": display["stand_sit_stable_count"],
                "moving_override_stable_count": display["moving_override_stable_count"],
                "moving_override_required": display["moving_override_required"],
                "moving_override_state": display.get("moving_override_state", "NONE"),
                "moving_override_reason": display.get("moving_override_reason", ""),
                "moving_override_blocked_by_body_still": display.get(
                    "moving_override_blocked_by_body_still", False
                ),
                "moving_translation_displacement_m": display.get(
                    "moving_translation_displacement_m", 0.0
                ),
                "moving_translation_confirmed": display.get(
                    "moving_translation_confirmed", False
                ),
                "stand_to_sit_gate": display.get("stand_to_sit_gate", "NA"),
                "stand_to_sit_conf": display.get("stand_to_sit_conf", 0.0),
                "stand_to_sit_margin": display.get("stand_to_sit_margin", 0.0),
                "stand_to_sit_stable_count": display.get(
                    "stand_to_sit_stable_count", 0
                ),
                "stand_to_sit_required": display.get("stand_to_sit_required", 0),
                "stand_to_sit_quality_ok": display.get(
                    "stand_to_sit_quality_ok", False
                ),
                "sitting_relative_gate_state": display.get(
                    "sitting_relative_gate_state", "NA"
                ),
                "sitting_relative_gate_stable_count": display.get(
                    "sitting_relative_gate_stable_count", 0
                ),
                "sitting_relative_gate_required_frames": display.get(
                    "sitting_relative_gate_required_frames", self.sitting_relative_frames
                ),
                "sitting_relative_gate_passed": display.get(
                    "sitting_relative_gate_passed", False
                ),
                "sitting_relative_gate_min_prob": display.get(
                    "sitting_relative_gate_min_prob", self.sitting_relative_min_prob
                ),
                "sitting_relative_gate_margin": display.get(
                    "sitting_relative_gate_margin", self.sitting_relative_margin
                ),
                "sitting_relative_gate_range_min_m": display.get(
                    "sitting_relative_gate_range_min_m", self.sitting_relative_range_min_m
                ),
                "sitting_relative_gate_range_ok": display.get(
                    "sitting_relative_gate_range_ok", False
                ),
                "sitting_relative_standing_veto_prob": display.get(
                    "sitting_relative_standing_veto_prob",
                    self.sitting_relative_standing_veto_prob,
                ),
                "sitting_relative_standing_veto_margin": display.get(
                    "sitting_relative_standing_veto_margin",
                    self.sitting_relative_standing_veto_margin,
                ),
                "sitting_relative_standing_veto_ok": display.get(
                    "sitting_relative_standing_veto_ok", False
                ),
                "sit_to_stand_recovery_count": display.get(
                    "sit_to_stand_recovery_count", 0
                ),
                "sit_to_stand_recovery_required": display.get(
                    "sit_to_stand_recovery_required", self.sit_to_stand_recovery_frames
                ),
                "display_status": display["display_status"],
                "transition_reason": display["transition_reason"],
                "final_display_pose": display.get("final_display_pose", displayed_label),
                "final_reason": display["transition_reason"],
                "fall_gate_passed": fall_gate_passed,
                "fall_gate_reason": fall_gate_reason,
                "sitting_gate_passed": sitting_gate_passed,
                "sitting_gate_reason": sitting_gate_reason,
                "stability_count": display["display_stability_count"],
                "stability_required": display["display_stability_required"],
                "stability_ratio": display["display_stability_ratio"],
                "assoc_mode": assoc["final_assoc"],
                "assoc_method": assoc["final_assoc"],
                "track_index": tid,
                "points_total": assoc["points_total"],
                "tracks_total": assoc["tracks_total"],
                "has_target_index": assoc["has_target_index"],
                "has_tid": assoc["has_tid"],
                "points_by_target_index": assoc["points_by_target_index"],
                "points_by_nearest": assoc["points_by_nearest"],
                "horizontal_speed": horizontal_speed,
                "vertical_speed": vertical_speed,
                "height_drop": height_drop,
                "x": float(target["pos_x"]),
                "y": float(target["pos_y"]),
                "z": float(target["pos_z"]),
                "raw_x": float(target["pos_x"]),
                "raw_y": float(target["pos_y"]),
                "raw_z": float(target["pos_z"]),
                "cal_x": cal_target["x"],
                "cal_y": cal_target["y"],
                "cal_z": cal_target["z"],
                "target_z": float(target["pos_z"]),
                "cal_target_z": cal_target["z"],
                "floor_z": self.floor_z_m,
                "floor_relative_z": cal_target["z"] - self.floor_z_m,
                "sensor_height_m": self.sensor_height_m,
                "sensor_pitch_deg": self.sensor_pitch_deg,
                "target_height": float(target_heights.get(tid, 0.0)),
                **geometry,
                "baseline_ready": stand_sit.get("baseline_ready", False),
                "baseline_frames": stand_sit.get("baseline_frames", 0),
                "baseline_top_z": stand_sit.get("baseline_top_z"),
                "baseline_centroid_z": stand_sit.get("baseline_centroid_z"),
                "height_drop_from_baseline": stand_sit.get("height_drop_from_baseline"),
                "centroid_drop": stand_sit.get("centroid_drop"),
                "target_z_drop": stand_sit.get("target_z_drop"),
                "geometry_decision": stand_sit.get("geometry_decision", "NA"),
                "geometry_reason": stand_sit.get("geometry_reason", "NA"),
                "geometry_range_threshold": stand_sit.get(
                    "geometry_range_threshold", 0.0
                ),
                "model_asset_used": self._model_asset_for_label(displayed_label),
                "model_scale": self._model_scale_for_label(displayed_label),
                "ground_z": self.ground_z,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "frame": frame_num,
            }
            results[tid] = result

        self._reset_stale_tracks(frame_num, seen_tids)
        self._update_human_model_validation(frame_num, results, seen_tids)
        self.latest_results = results
        self._write_log_rows(results)
        self._debug_print(frame_num, len(tracks), results)
        return results

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def reset_tid(self, tid: int) -> None:
        tid_int = int(tid)
        features.reset_tid(tid_int)
        self.smoother.reset_tid(tid_int)
        self.last_seen_frame.pop(tid_int, None)
        self.latest_results.pop(tid_int, None)
        self.speed_history.pop(tid_int, None)
        self.height_history.pop(tid_int, None)
        self.display_history.pop(tid_int, None)
        self.display_state.pop(tid_int, None)
        self.moving_override_history.pop(tid_int, None)
        self.stand_to_sit_gate_history.pop(tid_int, None)
        self.sitting_relative_gate_history.pop(tid_int, None)
        self.sit_to_stand_recovery_history.pop(tid_int, None)
        self.raw_label_history.pop(tid_int, None)
        self.confidence_history.pop(tid_int, None)
        self.probability_history.pop(tid_int, None)
        self.position_history.pop(tid_int, None)
        self.velocity_history.pop(tid_int, None)
        self.translation_history.pop(tid_int, None)
        self.standing_baseline.pop(tid_int, None)
        self.human_model_validation.pop(tid_int, None)

    def reset_all(self) -> None:
        features.reset_all()
        self.smoother.reset_all()
        self.last_seen_frame.clear()
        self.latest_results.clear()
        self.speed_history.clear()
        self.height_history.clear()
        self.display_history.clear()
        self.display_state.clear()
        self.moving_override_history.clear()
        self.stand_to_sit_gate_history.clear()
        self.sitting_relative_gate_history.clear()
        self.sit_to_stand_recovery_history.clear()
        self.raw_label_history.clear()
        self.confidence_history.clear()
        self.probability_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.translation_history.clear()
        self.standing_baseline.clear()
        self.human_model_validation.clear()
        self._last_human_ui_summary = {
            "frame": 0,
            "active": 0,
            "confirmed": 0,
            "provisional": 0,
            "suspect": 0,
            "rendered": 0,
            "stale": 0,
        }

    def get_human_ui_debug_summary(self) -> dict[str, Any]:
        return dict(self._last_human_ui_summary)

    def _update_human_model_validation(
        self,
        frame_num: int,
        results: dict[int, dict],
        active_tids: set[int],
    ) -> None:
        if not self.enable_human_models:
            return

        frame_num = int(frame_num)
        active_tids = {int(tid) for tid in active_tids}
        for tid in sorted(active_tids):
            pose = results.get(tid, {})
            state = self.human_model_validation.get(tid)
            if state is None:
                state = self._new_human_model_validation_state(tid, frame_num)
                self.human_model_validation[tid] = state
                self._print_human_ui_transition(
                    tid, "NEW_TO_PROVISIONAL", "new_tid"
                )
            elif state.get("state") == HUMAN_MODEL_STATE_LOST:
                state["state"] = HUMAN_MODEL_STATE_PROVISIONAL
                state["first_frame"] = frame_num
                state["good_history"].clear()
                state["bad_frames"] = 0
                state["no_points_frames"] = 0
                self._print_human_ui_transition(
                    tid, "LOST_TO_PROVISIONAL", "tid_reappeared"
                )

            evidence = self._human_model_evidence(pose)
            good = bool(evidence["good"])
            bad = bool(evidence["bad"])
            no_points = bool(evidence["no_points"])
            state["last_frame"] = frame_num
            state["stale_age"] = 0
            state["good_history"].append(good)
            if bad:
                state["bad_frames"] = int(state.get("bad_frames", 0) or 0) + 1
            else:
                state["bad_frames"] = 0
            if no_points:
                state["no_points_frames"] = int(state.get("no_points_frames", 0) or 0) + 1
            else:
                state["no_points_frames"] = 0

            previous_state = str(state.get("state", HUMAN_MODEL_STATE_PROVISIONAL))
            persisted_frames = max(
                1, frame_num - int(state.get("first_frame", frame_num)) + 1
            )
            good_frames = sum(1 for item in state["good_history"] if item)
            reason = str(evidence["reason"])

            can_confirm = (
                persisted_frames >= self.human_model_confirm_frames
                and good_frames >= self.human_model_confirm_min_quality_frames
            )
            if previous_state in {
                HUMAN_MODEL_STATE_PROVISIONAL,
                HUMAN_MODEL_STATE_SUSPECT,
            } and can_confirm:
                state["state"] = HUMAN_MODEL_STATE_CONFIRMED
                state["bad_frames"] = 0
                state["no_points_frames"] = 0
                transition = (
                    "PROVISIONAL_TO_CONFIRMED"
                    if previous_state == HUMAN_MODEL_STATE_PROVISIONAL
                    else "SUSPECT_GHOST_TO_CONFIRMED"
                )
                self._print_human_ui_transition(tid, transition, "good_evidence")
                reason = "good_evidence"
            elif (
                previous_state == HUMAN_MODEL_STATE_PROVISIONAL
                and (
                    int(state.get("no_points_frames", 0) or 0)
                    >= self.human_model_ghost_no_points_frames
                    or int(state.get("bad_frames", 0) or 0)
                    >= self.human_model_ghost_min_bad_frames
                )
            ):
                state["state"] = HUMAN_MODEL_STATE_SUSPECT
                if (
                    int(state.get("no_points_frames", 0) or 0)
                    >= self.human_model_ghost_no_points_frames
                ):
                    reason = "provisional_no_points_ghost"
                else:
                    reason = "provisional_bad_evidence_ghost"
                self._print_human_ui_transition(
                    tid, "PROVISIONAL_TO_SUSPECT_GHOST", reason
                )
            elif (
                str(state.get("state")) == HUMAN_MODEL_STATE_CONFIRMED
                and (bad or no_points)
            ):
                reason = "confirmed_retained_despite_low_evidence"

            state["good_frames"] = int(good_frames)
            state["reason"] = reason
            state["rendered"] = self._human_model_should_render_state(
                str(state.get("state"))
            )
            for key, value in evidence.items():
                if key == "reason":
                    continue
                state[key] = value
            if pose:
                pose.update(self._human_model_validation_fields(state))

        remove_validation_tids: list[int] = []
        for tid, state in list(self.human_model_validation.items()):
            if tid in active_tids:
                continue
            if state.get("state") != HUMAN_MODEL_STATE_LOST:
                state["state"] = HUMAN_MODEL_STATE_LOST
                state["rendered"] = False
                state["reason"] = "not_active"
            last_frame = int(state.get("last_frame", frame_num) or frame_num)
            state["stale_age"] = max(0, frame_num - last_frame)
            if state["stale_age"] >= self.human_model_stale_frames:
                remove_validation_tids.append(tid)

        self._update_human_ui_summary(frame_num)
        self._print_human_ui_debug(frame_num)
        for tid in remove_validation_tids:
            self.human_model_validation.pop(tid, None)

    def _new_human_model_validation_state(self, tid: int, frame_num: int) -> dict[str, Any]:
        history_len = max(
            self.human_model_confirm_frames,
            self.human_model_confirm_min_quality_frames,
            1,
        )
        return {
            "tid": int(tid),
            "state": HUMAN_MODEL_STATE_PROVISIONAL,
            "first_frame": int(frame_num),
            "last_frame": int(frame_num),
            "good_history": deque(maxlen=history_len),
            "good_frames": 0,
            "bad_frames": 0,
            "no_points_frames": 0,
            "stale_age": 0,
            "rendered": False,
            "reason": "new_tid",
        }

    def _human_model_evidence(self, pose: dict[str, Any]) -> dict[str, Any]:
        geom_pts = _int_value(pose.get("geom_pts", pose.get("num_points", 0)), 0)
        quality = str(pose.get("quality", "UNKNOWN") or "UNKNOWN").upper()
        geom_quality = str(pose.get("geom_quality", "UNKNOWN") or "UNKNOWN").upper()
        assoc = str(
            pose.get("assoc_mode", pose.get("assoc_method", "UNKNOWN")) or "UNKNOWN"
        ).lower()
        position_valid = self._human_model_position_valid(pose)
        height_valid = self._human_model_height_valid(pose)
        assoc_has_points = not (
            assoc in {"auto_none", "unknown", ""}
            or (assoc in {"target_index", "hybrid_target_index"} and geom_pts <= 0)
        )
        good = (
            geom_pts >= self.human_model_confirm_min_geom_pts
            and assoc_has_points
            and quality not in HUMAN_MODEL_BAD_QUALITIES
            and geom_quality != "TARGET_ONLY"
            and position_valid
            and height_valid
        )
        no_points = geom_pts <= 0 or quality == "NO_POINTS"
        bad_reasons: list[str] = []
        if geom_pts <= 0:
            bad_reasons.append("geom_pts_0")
        if quality in HUMAN_MODEL_WEAK_QUALITIES:
            bad_reasons.append(f"quality_{quality.lower()}")
        if geom_quality == "TARGET_ONLY":
            bad_reasons.append("geom_target_only")
        if assoc in {"auto_none", "unknown", ""}:
            bad_reasons.append("assoc_auto_none")
        if assoc in {"target_index", "hybrid_target_index"} and geom_pts <= 0:
            bad_reasons.append("assoc_index_no_points")
        if not position_valid:
            bad_reasons.append("invalid_position")
        if not height_valid:
            bad_reasons.append("invalid_body_geometry")
        return {
            "good": good,
            "bad": bool(bad_reasons),
            "no_points": no_points,
            "geom_pts": geom_pts,
            "quality": quality,
            "geom_quality": geom_quality,
            "assoc": assoc,
            "position_valid": position_valid,
            "height_valid": height_valid,
            "reason": "good_evidence" if good else ",".join(bad_reasons) or "weak_evidence",
        }

    def _human_model_position_valid(self, pose: dict[str, Any]) -> bool:
        values = [
            _optional_float(pose.get("x")),
            _optional_float(pose.get("y")),
            _optional_float(pose.get("z")),
        ]
        return all(value is not None and math.isfinite(value) for value in values)

    def _human_model_height_valid(self, pose: dict[str, Any]) -> bool:
        target_height = _optional_float(pose.get("target_height"))
        geom_height = _optional_float(pose.get("geom_height"))
        candidates = [
            value for value in (target_height, geom_height) if value is not None and value > 0
        ]
        if not candidates:
            return True
        return max(candidates) >= 0.20

    def _human_model_should_render_state(self, state: str) -> bool:
        if state == HUMAN_MODEL_STATE_CONFIRMED:
            return True
        if state == HUMAN_MODEL_STATE_PROVISIONAL:
            return self.human_model_show_provisional
        if state == HUMAN_MODEL_STATE_SUSPECT:
            return self.human_model_show_suspect
        return False

    def _human_model_validation_fields(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "human_model_validation_state": str(
                state.get("state", HUMAN_MODEL_STATE_PROVISIONAL)
            ),
            "human_model_good_frames": int(state.get("good_frames", 0) or 0),
            "human_model_bad_frames": int(state.get("bad_frames", 0) or 0),
            "human_model_no_points_frames": int(
                state.get("no_points_frames", 0) or 0
            ),
            "human_model_stale_age": int(state.get("stale_age", 0) or 0),
            "human_model_rendered": bool(state.get("rendered", False)),
            "human_model_reason": str(state.get("reason", "")),
        }

    def _update_human_ui_summary(self, frame_num: int, rendered: int | None = None) -> None:
        states = [str(item.get("state", "")) for item in self.human_model_validation.values()]
        active = sum(1 for state in states if state != HUMAN_MODEL_STATE_LOST)
        confirmed = states.count(HUMAN_MODEL_STATE_CONFIRMED)
        provisional = states.count(HUMAN_MODEL_STATE_PROVISIONAL)
        suspect = states.count(HUMAN_MODEL_STATE_SUSPECT)
        stale = states.count(HUMAN_MODEL_STATE_LOST)
        if rendered is None:
            rendered = sum(
                1
                for item in self.human_model_validation.values()
                if item.get("state") != HUMAN_MODEL_STATE_LOST
                and self._human_model_should_render_state(str(item.get("state")))
            )
        self._last_human_ui_summary = {
            "frame": int(frame_num),
            "active": int(active),
            "active_tracks": int(active),
            "confirmed": int(confirmed),
            "provisional": int(provisional),
            "suspect": int(suspect),
            "rendered": int(rendered),
            "stale": int(stale),
        }

    def _print_human_ui_transition(self, tid: int, transition: str, reason: str) -> None:
        if not self.human_model_debug:
            return
        print(
            f"[HUMAN_UI] tid={int(tid)} transition={transition} reason={reason}",
            flush=True,
        )

    def _print_human_ui_debug(self, frame_num: int) -> None:
        if not self.human_model_debug:
            return
        summary = self._last_human_ui_summary
        print(
            "[HUMAN_UI] frame={} active_tracks={} confirmed={} provisional={} suspect={} rendered={} stale={}".format(
                frame_num,
                summary.get("active", 0),
                summary.get("confirmed", 0),
                summary.get("provisional", 0),
                summary.get("suspect", 0),
                summary.get("rendered", 0),
                summary.get("stale", 0),
            ),
            flush=True,
        )
        for tid in sorted(self.human_model_validation):
            state = self.human_model_validation[tid]
            print(
                "[HUMAN_UI] tid={} state={} geom_pts={} quality={} geom_quality={} assoc={} good_frames={} bad_frames={} no_points_frames={} stale_age={} rendered={} reason={}".format(
                    tid,
                    state.get("state", HUMAN_MODEL_STATE_PROVISIONAL),
                    state.get("geom_pts", 0),
                    state.get("quality", "UNKNOWN"),
                    state.get("geom_quality", "UNKNOWN"),
                    state.get("assoc", "UNKNOWN"),
                    state.get("good_frames", 0),
                    state.get("bad_frames", 0),
                    state.get("no_points_frames", 0),
                    state.get("stale_age", 0),
                    str(bool(state.get("rendered", False))).lower(),
                    state.get("reason", ""),
                ),
                flush=True,
            )

    def get_3d_label_records(self, track_data=None, height_data=None) -> list[dict]:
        if not self.enable_3d_labels:
            return []

        track_positions = _track_position_by_tid(track_data)
        target_heights = _height_by_tid(height_data)
        records: list[dict] = []

        for tid in sorted(self.latest_results):
            pose = self.latest_results[tid]
            window_ready = bool(pose.get("window_ready", False))
            final_label = str(pose.get("displayed_label", pose.get("final_label", "")))
            confidence = float(
                pose.get("displayed_confidence", pose.get("final_confidence", 0.0))
            )
            validation_state = str(
                pose.get("human_model_validation_state", HUMAN_MODEL_STATE_CONFIRMED)
            )
            if validation_state != HUMAN_MODEL_STATE_CONFIRMED:
                final_label = validation_state or "UNKNOWN"
                confidence = 0.0

            x, y, z = track_positions.get(
                tid,
                (
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                ),
            )
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            if self.label_max_distance is not None:
                distance = math.sqrt(x * x + y * y + z * z)
                if distance > self.label_max_distance:
                    continue

            target_height = float(
                target_heights.get(tid, pose.get("target_height", 0.0)) or 0.0
            )
            z_label = self._label_z_for_pose(final_label, target_height)
            quality = str(pose.get("quality", "OK"))
            if validation_state != HUMAN_MODEL_STATE_CONFIRMED:
                text = f"{tid} | {final_label}"
            else:
                text = self._format_3d_label(tid, pose, quality)

            records.append(
                {
                    "tid": int(tid),
                    "text": text,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z_label),
                    "final_label": final_label,
                    "posture_ml": str(pose.get("smoothed_label", "")),
                    "motion_state": str(pose.get("motion_state", "")),
                    "confidence": confidence,
                    "quality": quality,
                    "window_ready": window_ready,
                    "human_model_validation_state": validation_state,
                }
            )
        return records

    def get_3d_model_records(self, track_data=None, height_data=None) -> list[dict]:
        if not self.enable_human_models:
            return []

        track_positions = _track_position_by_tid(track_data)
        target_heights = _height_by_tid(height_data)
        records: list[dict] = []

        for tid in sorted(self.latest_results):
            pose = self.latest_results[tid]
            validation_state = str(
                pose.get("human_model_validation_state", HUMAN_MODEL_STATE_PROVISIONAL)
            )
            if not self._human_model_should_render_state(validation_state):
                continue
            x, y, z = track_positions.get(
                tid,
                (
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                ),
            )
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue

            target_height = float(
                target_heights.get(tid, pose.get("target_height", 0.0)) or 0.0
            )
            displayed_label = str(pose.get("displayed_label", pose.get("final_label", "")))
            render_label = displayed_label
            if validation_state != HUMAN_MODEL_STATE_CONFIRMED:
                render_label = "UNKNOWN"

            records.append(
                {
                    "tid": int(tid),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "bottom_z": float(self.ground_z),
                    "ground_z": float(self.ground_z),
                    "height": float(target_height),
                    "target_height": float(target_height),
                    "final_label": render_label,
                    "displayed_label": (
                        displayed_label
                        if validation_state == HUMAN_MODEL_STATE_CONFIRMED
                        else validation_state
                    ),
                    "candidate_label": str(pose.get("candidate_label", "")),
                    "final_confidence": float(pose.get("displayed_confidence", 0.0)),
                    "displayed_confidence": float(pose.get("displayed_confidence", 0.0)),
                    "confidence": float(pose.get("displayed_confidence", 0.0)),
                    "posture_ml": str(pose.get("smoothed_label", "")),
                    "motion_state": str(pose.get("motion_state", "")),
                    "quality": str(pose.get("quality", "OK")),
                    "window_ready": bool(pose.get("window_ready", False)),
                    "num_points": int(pose.get("num_points", 0) or 0),
                    "model_asset_used": self._model_asset_for_label(render_label),
                    "model_scale": self._model_scale_for_label(render_label),
                    "frame": int(pose.get("frame", 0) or 0),
                    "human_model_validation_state": validation_state,
                    "human_model_good_frames": int(
                        pose.get("human_model_good_frames", 0) or 0
                    ),
                    "human_model_bad_frames": int(
                        pose.get("human_model_bad_frames", 0) or 0
                    ),
                    "human_model_no_points_frames": int(
                        pose.get("human_model_no_points_frames", 0) or 0
                    ),
                    "human_model_stale_age": int(
                        pose.get("human_model_stale_age", 0) or 0
                    ),
                    "human_model_reason": str(pose.get("human_model_reason", "")),
                }
            )
        frame = 0
        if self.latest_results:
            try:
                frame = max(int(item.get("frame", 0) or 0) for item in self.latest_results.values())
            except Exception:
                frame = int(self._last_human_ui_summary.get("frame", 0) or 0)
        self._update_human_ui_summary(frame, rendered=len(records))
        return records

    def _format_3d_label(self, tid: int, pose: dict, quality: str) -> str:
        final_label = str(pose.get("displayed_label", pose.get("final_label", "")))
        candidate_label = str(pose.get("candidate_label", final_label))
        confidence = float(
            pose.get("displayed_confidence", pose.get("final_confidence", 0.0))
        )
        confidence_percent = _percent_text(confidence)
        window_count = int(pose.get("window_count", pose.get("window_age", 0)) or 0)
        if final_label == "WARMUP" or not pose.get("window_ready", False):
            return f"{tid} | WARMUP {window_count}/8"
        if final_label == "NO_POSE" or not pose.get("prediction_exists", False):
            return f"{tid} | NO POSE"
        display_status = str(pose.get("display_status", ""))
        stability_count = int(pose.get("display_stability_count", 0) or 0)
        stability_required = int(
            pose.get("display_stability_required", self.display_stability_frames) or 1
        )
        if self.label_debug:
            fall_ok = str(bool(pose.get("fall_gate_passed", False))).lower()
            return (
                f"{tid} | {final_label} {confidence_percent}%\n"
                f"Cand:{candidate_label} Stable:{stability_count}/{stability_required} "
                f"FallOK:{fall_ok} Q:{quality}"
            )
        if display_status == "PENDING" and candidate_label != final_label:
            text = (
                f"{tid} | {final_label} -> {candidate_label} "
                f"{confidence_percent}%"
            )
            if quality != "OK":
                text = f"{text} *"
            return text

        values = {
            "tid": tid,
            "final_label": final_label,
            "displayed_label": final_label,
            "candidate_label": candidate_label,
            "posture_ml": str(pose.get("smoothed_label", "")),
            "smoothed_label": str(pose.get("smoothed_label", "")),
            "motion_state": str(pose.get("motion_state", "")),
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "quality": quality,
            "window_ready": bool(pose.get("window_ready", False)),
            "window_count": window_count,
            "num_points": int(pose.get("num_points", 0) or 0),
        }
        try:
            text = self.label_format.format(**values)
        except Exception:
            text = f"{tid} | {final_label} {confidence_percent}%"
        if quality != "OK":
            text = f"{text} *"
        return text

    def _track_to_target(self, track: list[float]) -> dict[str, float]:
        return {
            "tid": int(track[0]),
            "pos_x": _float_at(track, 1),
            "pos_y": _float_at(track, 2),
            "pos_z": _float_at(track, 3),
            "vel_x": _float_at(track, 4),
            "vel_y": _float_at(track, 5),
            "vel_z": _float_at(track, 6),
            "acc_x": _float_at(track, 7),
            "acc_y": _float_at(track, 8),
            "acc_z": _float_at(track, 9),
            "confidence": _float_at(track, 11),
        }

    def _associated_points(self, tid: int, points: list[list[float]]) -> list[dict[str, float]]:
        associated: list[dict[str, float]] = []
        for index, point in enumerate(points):
            track_index = int(_float_at(point, 6, 255.0))
            if track_index in UNASSOCIATED_TRACK_INDEXES:
                continue
            if track_index != int(tid):
                continue
            associated.append(
                {
                    "index": index,
                    "x": _float_at(point, 0),
                    "y": _float_at(point, 1),
                    "z": _float_at(point, 2),
                    "doppler": _float_at(point, 3),
                    "snr": _float_at(point, 4),
                    "track_index": track_index,
                }
            )

        # TODO: Future improvement: align delayed IWR6843 track index pointCloud
        # with previous frame target positions.
        return associated

    def _associate_points(
        self,
        *,
        tid: int,
        target: dict[str, float],
        points: list[list[float]],
        track_indexes: list[float],
        track_index_to_tid: dict[int, int],
        frame_num: int,
        tracks_total: int,
    ) -> dict[str, Any]:
        by_index = self._points_by_target_index(
            tid=tid,
            points=points,
            track_indexes=track_indexes,
            track_index_to_tid=track_index_to_tid,
        )
        by_nearest = self._points_by_nearest(target=target, points=points)
        method = self.assoc_method
        if method == "target_index":
            final_points = by_index
            final_assoc = "target_index"
        elif method == "nearest":
            final_points = by_nearest
            final_assoc = "nearest"
        else:
            final_points = by_index if by_index else by_nearest
            final_assoc = "target_index" if by_index else "nearest"
            if method == "auto" and not by_index and not by_nearest:
                final_assoc = "auto_none"
            elif method == "hybrid" and not by_index and by_nearest:
                final_assoc = "hybrid_nearest"
            elif method == "hybrid" and by_index:
                final_assoc = "hybrid_target_index"

        has_point_track = any(
            int(_float_at(point, 6, 255.0)) not in UNASSOCIATED_TRACK_INDEXES
            for point in points
            if len(point) > 6
        )
        assoc = {
            "points": final_points,
            "points_total": len(points),
            "tracks_total": int(tracks_total),
            "has_target_index": len(track_indexes) >= len(points) and len(points) > 0,
            "has_tid": has_point_track,
            "points_by_target_index": len(by_index),
            "points_by_nearest": len(by_nearest),
            "final_assoc": final_assoc,
        }
        self._assoc_debug_print(frame_num, tid, assoc)
        return assoc

    def _points_by_target_index(
        self,
        *,
        tid: int,
        points: list[list[float]],
        track_indexes: list[float],
        track_index_to_tid: dict[int, int],
    ) -> list[dict[str, float]]:
        associated: list[dict[str, float]] = []
        for index, point in enumerate(points):
            candidates: list[int] = []
            if index < len(track_indexes):
                candidates.append(int(track_indexes[index]))
            if len(point) > 6:
                candidates.append(int(_float_at(point, 6, 255.0)))
            matched_value: int | None = None
            for value in candidates:
                if value in UNASSOCIATED_TRACK_INDEXES:
                    continue
                if value == int(tid) or track_index_to_tid.get(value) == int(tid):
                    matched_value = value
                    break
            if matched_value is None:
                continue
            associated.append(self._point_dict(index, point, matched_value))
        return associated

    def _points_by_nearest(
        self,
        *,
        target: dict[str, float],
        points: list[list[float]],
    ) -> list[dict[str, float]]:
        associated: list[dict[str, float]] = []
        tx = float(target.get("pos_x", 0.0) or 0.0)
        ty = float(target.get("pos_y", 0.0) or 0.0)
        radius = self.assoc_nearest_radius_m
        z_min = min(self.assoc_nearest_z_min, self.assoc_nearest_z_max)
        z_max = max(self.assoc_nearest_z_min, self.assoc_nearest_z_max)
        for index, point in enumerate(points):
            px = _float_at(point, 0)
            py = _float_at(point, 1)
            pz = _float_at(point, 2)
            if not (z_min <= pz <= z_max):
                continue
            distance_xy = math.sqrt((px - tx) * (px - tx) + (py - ty) * (py - ty))
            if distance_xy <= radius:
                associated.append(self._point_dict(index, point, -1))
        return associated

    def _point_dict(
        self, index: int, point: list[float], track_index: int
    ) -> dict[str, float]:
        return {
            "index": index,
            "x": _float_at(point, 0),
            "y": _float_at(point, 1),
            "z": _float_at(point, 2),
            "doppler": _float_at(point, 3),
            "snr": _float_at(point, 4),
            "track_index": int(track_index),
        }

    def _assoc_debug_print(self, frame_num: int, tid: int, assoc: dict[str, Any]) -> None:
        if not self.assoc_debug:
            return
        final_points = int(assoc.get("points_by_target_index", 0) or 0)
        final_points = int(len(assoc.get("points", []) or []))
        should_print = frame_num % 30 == 0 or final_points == 0
        if not should_print:
            return
        if frame_num != self._last_assoc_debug_summary_frame:
            self._last_assoc_debug_summary_frame = frame_num
            print(
                "[POINT_ASSOC] "
                f"frame={frame_num} points_total={assoc.get('points_total', 0)} "
                f"tracks_total={assoc.get('tracks_total', 0)} "
                f"has_target_index={str(bool(assoc.get('has_target_index'))).lower()} "
                f"has_tid={str(bool(assoc.get('has_tid'))).lower()}",
                flush=True,
            )
        print(
            "[POINT_ASSOC] "
            f"tid={tid} track_index={tid} "
            f"points_by_target_index={assoc.get('points_by_target_index', 0)} "
            f"points_by_nearest={assoc.get('points_by_nearest', 0)} "
            f"final_assoc={assoc.get('final_assoc', '')} "
            f"final_points={final_points}",
            flush=True,
        )

    def _calibrated_point(self, x: float, y: float, z: float) -> dict[str, float]:
        raw_x = float(x)
        raw_y = float(y)
        raw_z = float(z)
        if not self.use_sensor_calibration:
            return {"x": raw_x, "y": raw_y, "z": raw_z}

        yaw = math.radians(self.sensor_yaw_deg)
        pitch = math.radians(self.sensor_pitch_deg)
        roll = math.radians(self.sensor_roll_deg)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x1 = raw_x * cos_yaw - raw_y * sin_yaw
        y1 = raw_x * sin_yaw + raw_y * cos_yaw
        z1 = raw_z

        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        x2 = x1

        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        x3 = x2 * cos_roll + z2 * sin_roll
        z3 = -x2 * sin_roll + z2 * cos_roll
        return {"x": x3, "y": y2, "z": z3 + self.sensor_height_m}

    def _point_geometry(
        self,
        *,
        associated_points: list[dict[str, float]],
        target: dict[str, float],
        cal_target: dict[str, float],
    ) -> dict[str, Any]:
        target_range_m = math.sqrt(
            float(target["pos_x"]) * float(target["pos_x"])
            + float(target["pos_y"]) * float(target["pos_y"])
        )
        base = {
            "associated_point_count": len(associated_points),
            "geom_pts": len(associated_points),
            "target_x": float(target["pos_x"]),
            "target_y": float(target["pos_y"]),
            "target_z": float(target["pos_z"]),
            "target_range_m": target_range_m,
            "target_speed": math.sqrt(
                float(target["vel_x"]) * float(target["vel_x"])
                + float(target["vel_y"]) * float(target["vel_y"])
            ),
            "geom_centroid_z": None,
            "geom_top_z": None,
            "geom_bottom_z": None,
            "geom_height": None,
            "geom_floor_centroid_z": None,
            "geom_floor_top_z": None,
            "geom_floor_bottom_z": None,
            "point_centroid_x": None,
            "point_centroid_y": None,
            "point_centroid_z": None,
            "point_top_z": None,
            "point_bottom_z": None,
            "point_height_extent": None,
            "point_vertical_spread": None,
            "point_range_min": None,
            "point_range_max": None,
            "floor_relative_top_z": None,
            "floor_relative_centroid_z": None,
            "floor_relative_bottom_z": None,
            "geom_quality": "TARGET_ONLY",
        }
        if not associated_points:
            return base

        xs = [float(point["x"]) for point in associated_points]
        ys = [float(point["y"]) for point in associated_points]
        zs = [float(point["z"]) for point in associated_points]
        cal_points = [
            self._calibrated_point(point["x"], point["y"], point["z"])
            for point in associated_points
        ]
        cal_zs = [float(point["z"]) for point in cal_points]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        centroid_z = sum(zs) / len(zs)
        top_z = max(zs)
        bottom_z = min(zs)
        height_extent = top_z - bottom_z
        ranges = [math.sqrt(x * x + y * y) for x, y in zip(xs, ys)]
        cal_centroid_z = sum(cal_zs) / len(cal_zs)
        cal_top_z = max(cal_zs)
        cal_bottom_z = min(cal_zs)
        spread = math.sqrt(sum((z - centroid_z) ** 2 for z in zs) / len(zs))
        base.update(
            {
                "geom_centroid_z": centroid_z,
                "geom_top_z": top_z,
                "geom_bottom_z": bottom_z,
                "geom_height": height_extent,
                "geom_floor_centroid_z": cal_centroid_z - self.floor_z_m,
                "geom_floor_top_z": cal_top_z - self.floor_z_m,
                "geom_floor_bottom_z": cal_bottom_z - self.floor_z_m,
                "point_centroid_x": centroid_x,
                "point_centroid_y": centroid_y,
                "point_centroid_z": centroid_z,
                "point_top_z": top_z,
                "point_bottom_z": bottom_z,
                "point_height_extent": height_extent,
                "point_vertical_spread": spread,
                "point_range_min": min(ranges),
                "point_range_max": max(ranges),
                "floor_relative_top_z": cal_top_z - self.floor_z_m,
                "floor_relative_centroid_z": cal_centroid_z - self.floor_z_m,
                "floor_relative_bottom_z": cal_bottom_z - self.floor_z_m,
                "geom_quality": "POINT_GEOMETRY",
            }
        )
        return base

    def _update_motion_state(self, tid: int, horizontal_speed: float) -> str:
        history = self.speed_history[int(tid)]
        history.append(float(horizontal_speed) > self.moving_speed_threshold)
        if len(history) >= self.moving_confirm_frames and all(history):
            return "MOVING"
        return "STATIC"

    def _update_height_drop(self, tid: int, current_z: float) -> float:
        history = self.height_history[int(tid)]
        drop = max(0.0, max(history) - current_z) if history else 0.0
        history.append(float(current_z))
        return float(drop)

    def _can_use_frame_for_inference(self, num_points: int) -> bool:
        num_points = int(num_points)
        if num_points <= 0:
            return self.allow_target_only
        return num_points >= self.min_associated_points_for_inference

    def _quality_label(
        self,
        *,
        window_ready: bool,
        prediction_exists: bool,
        can_use_frame: bool,
        num_points: int,
        smoothed_confidence: float,
    ) -> str:
        if not window_ready:
            return "WARMUP"
        if int(num_points) <= 0:
            return "NO_POINTS"
        if not can_use_frame or int(num_points) < 5:
            return "LOW_POINTS"
        if prediction_exists and smoothed_confidence < self.min_confidence:
            return "LOW_CONF"
        return "OK"

    def _update_tuning_histories(
        self,
        *,
        tid: int,
        raw_label: str,
        confidence: float,
        probabilities: dict[str, float],
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> None:
        tid_int = int(tid)
        self.raw_label_history[tid_int].append(str(raw_label).upper())
        self.confidence_history[tid_int].append(float(confidence or 0.0))
        self.probability_history[tid_int].append(dict(probabilities or {}))
        self.position_history[tid_int].append(tuple(float(value) for value in position))
        self.velocity_history[tid_int].append(tuple(float(value) for value in velocity))

    def _probability(self, probabilities: dict[str, float], label: str) -> float:
        try:
            return float((probabilities or {}).get(label, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _recent_candidate_ratio(self, tid: int, label: str, frames: int) -> float:
        history = list(self.display_history.get(int(tid), []))[-max(1, int(frames)) :]
        if not history:
            return 0.0
        target = str(label).upper()
        count = sum(1 for item_label, _confidence in history if item_label == target)
        return float(count) / float(len(history))

    def _evaluate_sitting_gate(
        self,
        *,
        smoothed_label: str,
        smoothed_confidence: float,
        probabilities: dict[str, float],
        horizontal_speed: float,
    ) -> tuple[bool, str]:
        label = str(smoothed_label).upper()
        sitting_prob = self._probability(probabilities, "SITTING")
        standing_prob = self._probability(probabilities, "STANDING")
        confidence = max(float(smoothed_confidence or 0.0), sitting_prob)
        if horizontal_speed > self.sitting_max_speed:
            return False, "speed_high"
        if label == "SITTING" and confidence >= self.sitting_min_confidence:
            return True, "sitting_top"
        if (
            sitting_prob >= self.sitting_min_confidence
            and sitting_prob >= standing_prob - 0.12
        ):
            return True, "sitting_close_to_standing"
        return False, "insufficient_sitting_probability"

    def _evaluate_fall_gate(
        self,
        *,
        tid: int,
        smoothed_label: str,
        smoothed_confidence: float,
        probabilities: dict[str, float],
        height_drop: float,
        vertical_speed: float,
    ) -> tuple[bool, str]:
        label = str(smoothed_label).upper()
        if label != "FALLING":
            return False, "ml_not_falling"

        confidence = float(smoothed_confidence or 0.0)
        strong_drop = height_drop >= self.fall_height_drop_threshold
        fast_vertical = vertical_speed >= self.fall_vertical_speed_threshold
        high_conf_with_drop = (
            confidence >= self.fall_high_confidence
            and height_drop >= self.fall_min_height_drop_with_high_confidence
        )

        sitting_prob = self._probability(probabilities, "SITTING")
        recent_sitting = self._recent_candidate_ratio(
            tid, "SITTING", self.display_stability_frames
        )
        previous = self.display_state.get(int(tid), {})
        previous_label = str(previous.get("label", "")).upper()
        if (
            not strong_drop
            and (sitting_prob >= self.sitting_min_confidence or recent_sitting >= 0.25)
        ):
            return False, "slow_sit_guard"
        if previous_label == "SITTING" and not strong_drop:
            return False, "stable_sitting_guard"

        if strong_drop:
            return True, "height_drop"
        if fast_vertical:
            return True, "vertical_speed"
        if high_conf_with_drop:
            return True, "high_confidence_with_mild_drop"
        return False, "no_physical_fall_evidence"

    def _range_zone(self, range_m: float | None) -> str:
        try:
            value = float(range_m)
        except Exception:
            return "unknown"
        if not math.isfinite(value):
            return "unknown"
        if value <= self.range_near_max:
            return "near"
        if value <= self.range_mid_max:
            return "mid"
        return "far"

    def _zone_value(self, values: dict[str, Any], zone: str, default: Any = None) -> Any:
        if zone in values:
            return values[zone]
        if "mid" in values:
            return values["mid"]
        return default

    def _stand_sit_transition_frames(
        self, previous_label: str, candidate_label: str, range_zone: str
    ) -> int:
        previous = str(previous_label or "").upper()
        candidate = str(candidate_label or "").upper()
        if previous == "STANDING" and candidate == "SITTING":
            return int(self._zone_value(self.stand_to_sit_frames_by_zone, range_zone, 8))
        if previous == "SITTING" and candidate == "STANDING":
            return int(self._zone_value(self.sit_to_stand_frames_by_zone, range_zone, 10))
        if previous in {"MOVING", "UNKNOWN", "WARMUP", "NO_POSE", ""} and candidate in {
            "STANDING",
            "SITTING",
        }:
            return int(
                self._zone_value(
                    self.unknown_to_stand_sit_frames_by_zone, range_zone, 8
                )
            )
        return int(self._zone_value(self.stand_to_sit_frames_by_zone, range_zone, 8))

    def _empty_stand_sit_info(
        self,
        *,
        probabilities: dict[str, float],
        range_m: float | None,
        range_zone: str,
    ) -> dict[str, Any]:
        standing_prob = self._probability(probabilities, "STANDING")
        sitting_prob = self._probability(probabilities, "SITTING")
        return {
            "active": False,
            "standing_prob": standing_prob,
            "sitting_prob": sitting_prob,
            "margin": standing_prob - sitting_prob,
            "range_m": range_m,
            "range_zone": range_zone,
            "decision": "NA",
            "resolved_label": "",
            "candidate_confidence": 0.0,
            "reason": "NA",
            "baseline_ready": False,
            "baseline_frames": 0,
            "baseline_top_z": None,
            "baseline_centroid_z": None,
            "baseline_target_z": None,
            "height_drop_from_baseline": None,
            "centroid_drop": None,
            "target_z_drop": None,
            "geometry_decision": "NA",
            "geometry_reason": "NA",
            "geometry_range_threshold": 0.0,
        }

    def _resolve_stand_sit(
        self,
        *,
        tid: int,
        prediction_exists: bool,
        smoothed_label: str,
        smoothed_confidence: float,
        probabilities: dict[str, float],
        range_m: float | None,
        range_zone: str,
        geometry: dict[str, Any],
        horizontal_speed: float,
        motion_state: str,
    ) -> dict[str, Any]:
        info = self._empty_stand_sit_info(
            probabilities=probabilities,
            range_m=range_m,
            range_zone=range_zone,
        )
        if not prediction_exists:
            return info

        label = str(smoothed_label or "").upper()
        if label not in {"STANDING", "SITTING"}:
            return info

        standing_prob = info["standing_prob"]
        sitting_prob = info["sitting_prob"]
        margin = float(standing_prob) - float(sitting_prob)
        required_margin = float(
            self._zone_value(self.stand_sit_margin_by_zone, range_zone, 0.10)
        )
        previous = self.display_state.get(int(tid), {})
        previous_label = str(previous.get("label", "")).upper()
        self._apply_geometry_stand_sit_evidence(
            tid=tid,
            info=info,
            geometry=geometry,
            range_zone=range_zone,
            horizontal_speed=horizontal_speed,
            motion_state=motion_state,
            previous_label=previous_label,
        )

        info["active"] = True
        if margin >= required_margin:
            info.update(
                {
                    "decision": "STANDING",
                    "resolved_label": "STANDING",
                    "candidate_confidence": standing_prob,
                    "reason": "stand_sit_margin_standing",
                }
            )
        elif -margin >= required_margin:
            info.update(
                {
                    "decision": "SITTING",
                    "resolved_label": "SITTING",
                    "candidate_confidence": sitting_prob,
                    "reason": "stand_sit_margin_sitting",
                }
            )
        elif previous_label in {"STANDING", "SITTING"}:
            info.update(
                {
                    "decision": "HOLD",
                    "resolved_label": previous_label,
                    "candidate_confidence": (
                        standing_prob if previous_label == "STANDING" else sitting_prob
                    ),
                    "reason": "stand_sit_hold_previous_ambiguous",
                }
            )
        else:
            info.update(
                {
                    "decision": "HOLD",
                    "resolved_label": "UNKNOWN",
                    "candidate_confidence": float(smoothed_confidence or 0.0),
                    "reason": "stand_sit_hold_previous_ambiguous",
                }
            )

        geometry_decision = str(info.get("geometry_decision") or "NA").upper()
        if geometry_decision in {"STANDING", "SITTING"}:
            if geometry_decision == "SITTING":
                confidence = max(sitting_prob, float(smoothed_confidence or 0.0) * 0.5)
            else:
                confidence = max(standing_prob, float(smoothed_confidence or 0.0) * 0.5)
            info.update(
                {
                    "decision": geometry_decision,
                    "resolved_label": geometry_decision,
                    "candidate_confidence": confidence,
                    "reason": str(info.get("geometry_reason") or "geometry_sitting_drop"),
                }
            )
        return info

    def _apply_geometry_stand_sit_evidence(
        self,
        *,
        tid: int,
        info: dict[str, Any],
        geometry: dict[str, Any],
        range_zone: str,
        horizontal_speed: float,
        motion_state: str,
        previous_label: str,
    ) -> None:
        threshold = float(
            self._zone_value(self.sitting_drop_by_zone, range_zone, SITTING_DROP_DEFAULTS["mid"])
        )
        baseline = self.standing_baseline.get(int(tid), {})
        baseline_frames = int(baseline.get("frames", 0) or 0)
        baseline_ready = (
            self.use_standing_baseline
            and baseline_frames >= self.standing_baseline_min_frames
        )
        info.update(
            {
                "baseline_ready": baseline_ready,
                "baseline_frames": baseline_frames,
                "baseline_top_z": baseline.get("top_z"),
                "baseline_centroid_z": baseline.get("centroid_z"),
                "baseline_target_z": baseline.get("target_z"),
                "geometry_range_threshold": threshold,
            }
        )
        if not baseline_ready:
            return

        top_z = _optional_float(geometry.get("geom_top_z"))
        centroid_z = _optional_float(geometry.get("geom_centroid_z"))
        target_z = _optional_float(geometry.get("target_z"))
        height_extent = _optional_float(geometry.get("geom_height"))
        baseline_top = _optional_float(baseline.get("top_z"))
        baseline_centroid = _optional_float(baseline.get("centroid_z"))
        baseline_target = _optional_float(baseline.get("target_z"))
        baseline_height = _optional_float(baseline.get("height_extent"))

        top_drop = (
            max(0.0, baseline_top - top_z)
            if baseline_top is not None and top_z is not None
            else None
        )
        centroid_drop = (
            max(0.0, baseline_centroid - centroid_z)
            if baseline_centroid is not None and centroid_z is not None
            else None
        )
        target_z_drop = (
            max(0.0, baseline_target - target_z)
            if baseline_target is not None and target_z is not None
            else None
        )
        height_drop_from_baseline = (
            max(0.0, baseline_height - height_extent)
            if baseline_height is not None and height_extent is not None
            else None
        )
        info.update(
            {
                "height_drop_from_baseline": height_drop_from_baseline,
                "centroid_drop": centroid_drop,
                "target_z_drop": target_z_drop,
            }
        )

        sitting_prob = float(info.get("sitting_prob", 0.0) or 0.0)
        standing_prob = float(info.get("standing_prob", 0.0) or 0.0)
        margin = standing_prob - sitting_prob
        stationary = (
            float(horizontal_speed or 0.0) <= self.sitting_max_speed
            and str(motion_state).upper() != "MOVING"
        )
        model_not_strong_against_sit = margin < float(
            self._zone_value(self.strong_stand_sit_margin_by_zone, range_zone, 0.18)
        )
        point_geometry = str(geometry.get("geom_quality", "")) == "POINT_GEOMETRY"
        point_drop = (
            (centroid_drop is not None and centroid_drop >= threshold)
            or (top_drop is not None and top_drop >= threshold)
            or (
                height_drop_from_baseline is not None
                and height_drop_from_baseline >= threshold
            )
        )
        target_drop = (
            target_z_drop is not None and target_z_drop >= self.sitting_drop_target_z_m
        )
        sitting_drop = point_drop or (not point_geometry and target_drop)

        if (
            stationary
            and sitting_prob >= self.sitting_drop_min_sit_prob
            and model_not_strong_against_sit
            and sitting_drop
            and previous_label in {"STANDING", "SITTING", ""}
        ):
            reason = (
                "upright_sitting_geometry_supported"
                if point_geometry
                else "geometry_sitting_drop_target_only"
            )
            info.update(
                {
                    "geometry_decision": "SITTING",
                    "geometry_reason": reason,
                }
            )
        elif stationary and margin > 0.0 and not sitting_drop:
            info.update(
                {
                    "geometry_decision": "STANDING",
                    "geometry_reason": "geometry_no_sitting_drop",
                }
            )

    def _final_label(
        self,
        *,
        window_ready: bool,
        prediction_exists: bool,
        smoothed_label: str,
        smoothed_confidence: float,
        motion_state: str,
        height_drop: float,
        horizontal_speed: float,
        vertical_speed: float,
        probabilities: dict[str, float],
        fall_gate_passed: bool,
        sitting_gate_passed: bool,
        stand_sit: dict[str, Any],
    ) -> str:
        if not window_ready:
            return "WARMUP"
        if not prediction_exists:
            return "NO_POSE"
        if smoothed_confidence < self.unknown_confidence:
            return "UNKNOWN"

        label = str(smoothed_label).upper()
        if bool(stand_sit.get("active")):
            resolved = str(stand_sit.get("resolved_label") or "").upper()
            if resolved in {"STANDING", "SITTING", "UNKNOWN"}:
                return resolved
        if label == "FALLING":
            if fall_gate_passed:
                return "FALLING"
            if sitting_gate_passed:
                return "SITTING"
            return "UNKNOWN"
        if label == "LYING":
            return "LYING"
        if label == "SITTING":
            if sitting_gate_passed:
                return "SITTING"
            if motion_state == "MOVING":
                return "MOVING"
            return "SITTING"
        if label == "WALKING":
            return "MOVING"
        if label == "STANDING" and motion_state == "MOVING":
            return "MOVING"
        if label == "STANDING":
            return "STANDING"
        return label if label in self.class_names else "UNKNOWN"

    def _display_requirements(self, candidate_label: str) -> tuple[int, float, float]:
        label = str(candidate_label).upper()
        if label == "WALKING":
            label = "MOVING"
        if label == "FALLING":
            required = (
                self.pose_stability_frames_by_pose["FALLING"]
                if self.falling_fast_update
                else self.display_stability_frames
            )
        else:
            required = self.pose_stability_frames_by_pose.get(
                label, self.display_stability_frames
            )
        min_confidence = self.pose_min_confidence_by_pose.get(
            label, self.display_min_confidence
        )
        if label == "SITTING":
            required_ratio = self.sitting_stability_ratio
        else:
            required_ratio = self.display_stability_ratio
        return required, required_ratio, min_confidence

    def _update_display_state(
        self,
        *,
        tid: int,
        candidate_label: str,
        candidate_confidence: float,
        window_ready: bool,
        prediction_exists: bool,
        horizontal_speed: float,
        motion_state: str,
        raw_label: str,
        smoothed_label: str,
        target_position: tuple[float, float, float],
        range_m: float | None,
        range_zone: str,
        quality: str,
        geom_quality: str,
        stand_sit: dict[str, Any],
        fall_gate_passed: bool,
        fall_gate_reason: str,
        sitting_gate_passed: bool,
        sitting_gate_reason: str,
    ) -> dict[str, Any]:
        candidate = str(candidate_label or "UNKNOWN").upper()
        confidence = float(candidate_confidence or 0.0)
        moving_override_required = int(
            self._zone_value(self.moving_override_frames_by_zone, range_zone, 4)
        )

        if not window_ready:
            self.display_history.pop(int(tid), None)
            self.display_state.pop(int(tid), None)
            self.moving_override_history.pop(int(tid), None)
            self.stand_to_sit_gate_history.pop(int(tid), None)
            self.sitting_relative_gate_history.pop(int(tid), None)
            self.sit_to_stand_recovery_history.pop(int(tid), None)
            return {
                "displayed_label": "WARMUP",
                "displayed_confidence": 0.0,
                "display_stability_count": 0,
                "display_stability_required": self.display_stability_frames,
                "display_stability_ratio": 0.0,
                "candidate_stable_count": 0,
                "pose_min_confidence": 0.0,
                "pose_required_frames": self.display_stability_frames,
                "stand_sit_required_frames": 0,
                "stand_sit_stable_count": 0,
                "moving_override_stable_count": 0,
                "moving_override_required": moving_override_required,
                "display_status": "WARMUP",
                "transition_reason": "warming_up",
            }

        if not prediction_exists:
            return {
                "displayed_label": "NO_POSE",
                "displayed_confidence": 0.0,
                "display_stability_count": 0,
                "display_stability_required": self.display_stability_frames,
                "display_stability_ratio": 0.0,
                "candidate_stable_count": 0,
                "pose_min_confidence": 0.0,
                "pose_required_frames": self.display_stability_frames,
                "stand_sit_required_frames": 0,
                "stand_sit_stable_count": 0,
                "moving_override_stable_count": 0,
                "moving_override_required": moving_override_required,
                "display_status": "NO_POSE",
                "transition_reason": "no_prediction",
            }

        if not self.display_hysteresis:
            self.display_state[int(tid)] = {
                "label": candidate,
                "confidence": confidence,
            }
            return {
                "displayed_label": candidate,
                "displayed_confidence": confidence,
                "display_stability_count": 1,
                "display_stability_required": 1,
                "display_stability_ratio": 1.0,
                "candidate_stable_count": 1,
                "pose_min_confidence": 0.0,
                "pose_required_frames": 1,
                "stand_sit_required_frames": 1 if candidate in {"STANDING", "SITTING"} else 0,
                "stand_sit_stable_count": 1 if candidate in {"STANDING", "SITTING"} else 0,
                "moving_override_stable_count": 0,
                "moving_override_required": moving_override_required,
                "display_status": "STABLE",
                "transition_reason": "hysteresis_disabled",
            }

        previous = self.display_state.get(int(tid))
        previous_label = str((previous or {}).get("label", "")).upper()
        stand_sit_active = bool(stand_sit.get("active"))
        moving_override_count = 0
        moving_override_reason = ""
        stand_to_sit_gate = "NA"
        stand_to_sit_conf = 0.0
        stand_to_sit_margin = 0.0
        stand_to_sit_count = 0
        stand_to_sit_required = self.stand_to_sit_frames
        stand_to_sit_quality_ok = False
        sitting_relative_gate_state = "DISABLED" if not self.sitting_relative_gate else "NA"
        sitting_relative_count = 0
        sitting_relative_required = self.sitting_relative_frames
        sitting_relative_passed = False
        sitting_relative_evidence = False
        sitting_relative_block_reason = ""
        sitting_relative_range_ok = False
        sitting_relative_standing_veto_ok = False
        sit_to_stand_recovery_count = 0
        sit_to_stand_recovery_required = self.sit_to_stand_recovery_frames
        sit_to_stand_recovery_forced = False
        moving_override_state = "NONE"
        moving_override_blocked_by_body_still = False
        translation = self._translation_evidence(
            tid=tid,
            target_position=target_position,
            horizontal_speed=horizontal_speed,
            motion_state=motion_state,
            raw_label=raw_label,
            smoothed_label=smoothed_label,
        )
        strong_margin = float(
            self._zone_value(self.strong_stand_sit_margin_by_zone, range_zone, 0.18)
        )
        stand_sit_decision = str(stand_sit.get("decision", "")).upper()
        stand_sit_margin = abs(float(stand_sit.get("margin", 0.0) or 0.0))
        strong_stand_sit = (
            stand_sit_decision in {"STANDING", "SITTING"}
            and stand_sit_margin >= strong_margin
        )
        standing_prob = self._probability(
            stand_sit.get("probabilities", {}), "STANDING"
        )
        sitting_prob = self._probability(stand_sit.get("probabilities", {}), "SITTING")
        falling_prob = self._probability(
            stand_sit.get("probabilities", {}), "FALLING"
        )
        lying_prob = self._probability(stand_sit.get("probabilities", {}), "LYING")
        if not standing_prob and not sitting_prob:
            standing_prob = float(stand_sit.get("standing_prob", 0.0) or 0.0)
            sitting_prob = float(stand_sit.get("sitting_prob", 0.0) or 0.0)
        sit_minus_stand_margin = float(sitting_prob) - float(standing_prob)
        falling_lying_dominant = max(falling_prob, lying_prob) > max(
            sitting_prob,
            standing_prob,
        )
        try:
            sitting_relative_range_ok = (
                range_m is not None
                and not math.isnan(float(range_m))
                and float(range_m) >= self.sitting_relative_range_min_m
            )
        except (TypeError, ValueError):
            sitting_relative_range_ok = False
        sitting_relative_standing_veto_ok = bool(
            standing_prob < self.sitting_relative_standing_veto_prob
            and (standing_prob - sitting_prob)
            < self.sitting_relative_standing_veto_margin
        )
        if self.sitting_relative_gate and prediction_exists and window_ready:
            if translation["confirmed"]:
                sitting_relative_block_reason = "sitting_relative_gate_blocked_by_body_motion"
            elif falling_lying_dominant:
                sitting_relative_block_reason = "sitting_relative_gate_blocked_fall_lying"
            elif not sitting_relative_range_ok:
                sitting_relative_block_reason = "sitting_relative_gate_blocked_range"
            elif not sitting_relative_standing_veto_ok:
                sitting_relative_block_reason = "sitting_relative_gate_blocked_standing_veto"
            sitting_relative_evidence = bool(
                sitting_relative_range_ok
                and sitting_prob >= self.sitting_relative_min_prob
                and sit_minus_stand_margin >= self.sitting_relative_margin
                and sitting_relative_standing_veto_ok
                and not translation["confirmed"]
                and not falling_lying_dominant
            )
            relative_history = self.sitting_relative_gate_history[int(tid)]
            relative_history.append(sitting_relative_evidence)
            for value in reversed(relative_history):
                if not value:
                    break
                sitting_relative_count += 1
            if sitting_relative_evidence:
                sitting_relative_gate_state = (
                    "PASS"
                    if sitting_relative_count >= sitting_relative_required
                    else "WAIT"
                )
                sitting_relative_passed = sitting_relative_gate_state == "PASS"
            elif sitting_relative_block_reason:
                sitting_relative_gate_state = "BLOCK"
            else:
                sitting_relative_gate_state = "NA"
        else:
            self.sitting_relative_gate_history.pop(int(tid), None)
        in_stand_sit_context = (
            previous_label in {"STANDING", "SITTING"}
            or candidate in {"STANDING", "SITTING"}
            or (stand_sit_active and str(stand_sit.get("resolved_label", "")).upper() in {"STANDING", "SITTING"})
        )
        if in_stand_sit_context:
            motion_evidence = (
                float(horizontal_speed or 0.0) > self.moving_speed_threshold
                or str(motion_state).upper() == "MOVING"
            )
            motion_history = self.moving_override_history[int(tid)]
            motion_history.append(bool(motion_evidence))
            for value in reversed(motion_history):
                if not value:
                    break
                moving_override_count += 1
            if motion_evidence and candidate == "MOVING" and previous_label in {"STANDING", "SITTING"}:
                if (
                    self.moving_override_require_body_translation_for_sitting
                    and sitting_relative_passed
                    and not translation["confirmed"]
                ):
                    candidate = "SITTING"
                    confidence = max(
                        confidence,
                        sitting_prob,
                        float(previous.get("confidence", confidence) or 0.0),
                    )
                    moving_override_reason = "moving_override_blocked_body_still_sitting"
                    moving_override_state = "BLOCKED_BODY_STILL_SITTING"
                    moving_override_blocked_by_body_still = True
                elif (
                    strong_stand_sit
                    and self.moving_require_translation
                    and not translation["confirmed"]
                ):
                    candidate = str(stand_sit.get("resolved_label") or previous_label)
                    confidence = max(
                        confidence,
                        float(previous.get("confidence", confidence) or 0.0),
                    )
                    moving_override_reason = "moving_override_speed_only_rejected"
                    moving_override_state = "BLOCKED_STRONG_STAND_SIT"
                elif moving_override_count >= moving_override_required:
                    moving_override_reason = (
                        "moving_override_translation_confirmed"
                        if translation["confirmed"]
                        else "moving_override_sustained"
                    )
                    moving_override_state = (
                        "TRANSLATION_CONFIRMED"
                        if translation["confirmed"]
                        else "SUSTAINED"
                    )
                else:
                    candidate = previous_label
                    confidence = float(previous.get("confidence", confidence) or 0.0)
                    moving_override_reason = "moving_override_waiting"
                    moving_override_state = "WAITING"
            elif motion_evidence and candidate in {"STANDING", "SITTING"}:
                if (
                    self.moving_override_require_body_translation_for_sitting
                    and candidate == "SITTING"
                    and sitting_relative_passed
                    and not translation["confirmed"]
                ):
                    moving_override_reason = "moving_override_blocked_body_still_sitting"
                    moving_override_state = "BLOCKED_BODY_STILL_SITTING"
                    moving_override_blocked_by_body_still = True
                elif (
                    strong_stand_sit
                    and self.moving_require_translation
                    and not translation["confirmed"]
                ):
                    moving_override_reason = "moving_override_blocked_by_strong_stand_sit"
                    moving_override_state = "BLOCKED_STRONG_STAND_SIT"
                elif moving_override_count >= moving_override_required:
                    candidate = "MOVING"
                    confidence = max(confidence, self.moving_min_confidence)
                    moving_override_reason = (
                        "moving_override_translation_confirmed"
                        if translation["confirmed"]
                        else "moving_override_sustained"
                    )
                    moving_override_state = (
                        "TRANSLATION_CONFIRMED"
                        if translation["confirmed"]
                        else "SUSTAINED"
                    )
                else:
                    moving_override_reason = "moving_override_waiting"
                    moving_override_state = "WAITING"
        else:
            self.moving_override_history.pop(int(tid), None)

        if previous_label == "SITTING":
            recovery_margin = standing_prob - sitting_prob
            recovery_evidence = recovery_margin >= self.sit_to_stand_recovery_margin
            if not recovery_evidence and standing_prob <= 0.0 and sitting_prob <= 0.0:
                recovery_evidence = (
                    str(smoothed_label).upper() == "STANDING"
                    and confidence >= self.display_min_confidence
                ) or (
                    str(raw_label).upper() == "STANDING"
                    and confidence >= self.display_min_confidence
                )
            recovery_history = self.sit_to_stand_recovery_history[int(tid)]
            recovery_history.append(bool(recovery_evidence))
            for value in reversed(recovery_history):
                if not value:
                    break
                sit_to_stand_recovery_count += 1
            if sit_to_stand_recovery_count >= sit_to_stand_recovery_required:
                candidate = "STANDING"
                confidence = max(
                    confidence,
                    standing_prob,
                    float(previous.get("confidence", 0.0) or 0.0),
                )
                sit_to_stand_recovery_forced = True
        else:
            self.sit_to_stand_recovery_history.pop(int(tid), None)

        if (
            previous_label == "STANDING"
            and candidate == "STANDING"
            and sitting_relative_passed
        ):
            candidate = "SITTING"
            confidence = max(confidence, sitting_prob)
            moving_override_reason = "sitting_relative_gate"

        if previous_label == "STANDING" and candidate == "SITTING":
            blocked_quality_values = {"NO_POINTS", "TARGET_ONLY", "NO_ASSOC_POINTS"}
            observed_quality = {
                str(quality or "").upper(),
                str(geom_quality or "").upper(),
            }
            stand_to_sit_quality_ok = (
                self.stand_to_sit_allow_target_only
                or not bool(observed_quality & blocked_quality_values)
            )
            stand_to_sit_conf = max(confidence, sitting_prob)
            stand_to_sit_margin = sitting_prob - standing_prob
            conf_ok = stand_to_sit_conf >= self.stand_to_sit_min_confidence
            margin_ok = stand_to_sit_margin >= self.stand_to_sit_margin
            gate_evidence = bool(stand_to_sit_quality_ok and conf_ok and margin_ok)
            gate_history = self.stand_to_sit_gate_history[int(tid)]
            gate_history.append(gate_evidence)
            for value in reversed(gate_history):
                if not value:
                    break
                stand_to_sit_count += 1
            relative_override = bool(
                self.sitting_relative_gate and sitting_relative_passed
            )
            relative_waiting = bool(
                self.sitting_relative_gate
                and sitting_relative_evidence
                and not sitting_relative_passed
            )
            relative_blocked = bool(
                self.sitting_relative_gate
                and sitting_relative_gate_state == "BLOCK"
                and sitting_relative_block_reason
            )
            if relative_override:
                stand_to_sit_gate = "RELATIVE_PASS"
                moving_override_reason = "sitting_relative_gate"
                confidence = max(confidence, sitting_prob)
            elif relative_waiting:
                stand_to_sit_gate = "WAIT"
                moving_override_reason = "sitting_relative_gate_waiting"
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            elif relative_blocked:
                stand_to_sit_gate = "BLOCK"
                moving_override_reason = sitting_relative_block_reason
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            elif not stand_to_sit_quality_ok:
                stand_to_sit_gate = "BLOCK"
                moving_override_reason = "stand_to_sit_blocked_target_only"
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            elif not conf_ok:
                stand_to_sit_gate = "BLOCK"
                moving_override_reason = "stand_to_sit_blocked_confidence"
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            elif not margin_ok:
                stand_to_sit_gate = "BLOCK"
                moving_override_reason = "stand_to_sit_blocked_margin"
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            elif stand_to_sit_count < stand_to_sit_required:
                stand_to_sit_gate = "WAIT"
                moving_override_reason = "stand_to_sit_waiting_gate"
                candidate = previous_label
                confidence = float(previous.get("confidence", confidence) or 0.0)
            else:
                stand_to_sit_gate = "PASS"
                moving_override_reason = "stand_to_sit_gate_passed"
        else:
            self.stand_to_sit_gate_history.pop(int(tid), None)

        history = self.display_history[int(tid)]
        history.append((candidate, confidence))

        required, _required_ratio, min_confidence = self._display_requirements(candidate)
        stand_sit_required = 0
        if stand_sit_active and candidate in {"STANDING", "SITTING"}:
            stand_sit_required = self._stand_sit_transition_frames(
                previous_label, candidate, range_zone
            )
            required = stand_sit_required
            min_confidence = 0.0
        required = max(1, int(required))
        window_len = max(required, self.display_stability_frames)
        recent = list(history)[-window_len:]
        labels = [label for label, _conf in recent]
        window_count = labels.count(candidate)
        count = 0
        for label in reversed(labels):
            if label != candidate:
                break
            count += 1
        enough_samples = len(recent) >= required
        if moving_override_reason in {
            "moving_override_sustained",
            "moving_override_translation_confirmed",
        } and candidate == "MOVING":
            required = moving_override_required
            count = moving_override_count
            enough_samples = moving_override_count >= required
        if stand_to_sit_gate == "PASS" and candidate == "SITTING":
            required = stand_to_sit_required
            count = stand_to_sit_count
            enough_samples = stand_to_sit_count >= required
            min_confidence = 0.0
        if sitting_relative_passed and candidate == "SITTING":
            required = sitting_relative_required
            count = sitting_relative_count
            enough_samples = sitting_relative_count >= required
            min_confidence = 0.0
        if sit_to_stand_recovery_forced and candidate == "STANDING":
            required = sit_to_stand_recovery_required
            count = sit_to_stand_recovery_count
            enough_samples = sit_to_stand_recovery_count >= required
            min_confidence = 0.0
        confidence_ok = confidence >= min_confidence
        gate_ok = True
        gate_reason = ""
        if candidate == "FALLING":
            gate_ok = bool(fall_gate_passed)
            gate_reason = fall_gate_reason
        elif candidate == "SITTING" and not stand_sit_active:
            gate_ok = bool(sitting_gate_passed)
            gate_reason = sitting_gate_reason
        elif candidate == "MOVING" and horizontal_speed <= self.moving_speed_threshold:
            gate_ok = False
            gate_reason = "speed_below_moving_threshold"

        candidate_stable = (
            gate_ok
            and confidence_ok
            and enough_samples
            and count >= required
        )

        if candidate_stable:
            self.display_state[int(tid)] = {
                "label": candidate,
                "confidence": confidence,
            }
            displayed_label = candidate
            displayed_confidence = confidence
            status = "STABLE"
            if moving_override_reason in {
                "moving_override_sustained",
                "moving_override_translation_confirmed",
            }:
                transition_reason = moving_override_reason
            elif stand_to_sit_gate == "PASS" and candidate == "SITTING":
                transition_reason = "stand_to_sit_gate_passed"
            elif sitting_relative_passed and candidate == "SITTING":
                transition_reason = "sitting_relative_gate"
            elif sit_to_stand_recovery_forced and candidate == "STANDING":
                transition_reason = "sit_to_stand_recovery"
            elif stand_to_sit_gate in {"BLOCK", "WAIT"}:
                transition_reason = moving_override_reason
            elif stand_sit_active and candidate in {"STANDING", "SITTING"}:
                if stand_sit.get("decision") == "HOLD":
                    transition_reason = "stand_sit_hold_previous_ambiguous"
                elif previous_label and previous_label != candidate:
                    transition_reason = "stand_sit_hysteresis_update"
                else:
                    transition_reason = str(stand_sit.get("reason") or "stand_sit_hysteresis_update")
            else:
                transition_reason = "pose_specific_stable_update"
        elif previous is None:
            displayed_label = candidate
            displayed_confidence = confidence
            status = "PENDING"
            if moving_override_reason in {
                "moving_override_waiting",
                "moving_override_blocked_by_strong_stand_sit",
                "moving_override_blocked_body_still_sitting",
                "moving_override_speed_only_rejected",
                "sitting_relative_gate_waiting",
                "sitting_relative_gate_blocked_by_body_motion",
                "sitting_relative_gate_blocked_fall_lying",
                "sitting_relative_gate_blocked_range",
                "sitting_relative_gate_blocked_standing_veto",
                "stand_to_sit_blocked_target_only",
                "stand_to_sit_blocked_margin",
                "stand_to_sit_blocked_confidence",
                "stand_to_sit_waiting_gate",
            }:
                transition_reason = moving_override_reason
            elif not gate_ok:
                transition_reason = f"gate_blocked:{gate_reason}"
            elif not confidence_ok:
                transition_reason = "confidence_below_pose_min"
            elif stand_sit_active and candidate in {"STANDING", "SITTING"}:
                transition_reason = "stand_sit_waiting_hysteresis"
            else:
                transition_reason = "pending_start"
        else:
            displayed_label = str(previous.get("label", candidate))
            displayed_confidence = float(previous.get("confidence", confidence) or 0.0)
            status = "PENDING" if candidate != displayed_label else "STABLE"
            if moving_override_reason in {
                "moving_override_waiting",
                "moving_override_blocked_by_strong_stand_sit",
                "moving_override_blocked_body_still_sitting",
                "moving_override_speed_only_rejected",
                "sitting_relative_gate_waiting",
                "sitting_relative_gate_blocked_by_body_motion",
                "sitting_relative_gate_blocked_fall_lying",
                "sitting_relative_gate_blocked_range",
                "sitting_relative_gate_blocked_standing_veto",
                "stand_to_sit_blocked_target_only",
                "stand_to_sit_blocked_margin",
                "stand_to_sit_blocked_confidence",
                "stand_to_sit_waiting_gate",
            }:
                transition_reason = moving_override_reason
            elif not gate_ok:
                transition_reason = f"gate_blocked:{gate_reason}"
            elif not confidence_ok:
                transition_reason = "confidence_below_pose_min"
            elif not enough_samples:
                transition_reason = (
                    "stand_sit_waiting_hysteresis"
                    if stand_sit_active and candidate in {"STANDING", "SITTING"}
                    else "waiting_for_samples"
                )
            elif count < required:
                transition_reason = (
                    "stand_sit_waiting_hysteresis"
                    if stand_sit_active and candidate in {"STANDING", "SITTING"}
                    else "waiting_for_pose_specific_stability"
                )
            else:
                transition_reason = "keep_previous"

        return {
            "displayed_label": displayed_label,
            "displayed_confidence": displayed_confidence,
            "display_stability_count": count,
            "display_stability_required": required,
            "display_stability_ratio": (
                float(window_count) / float(len(recent)) if recent else 0.0
            ),
            "candidate_stable_count": count,
            "pose_min_confidence": min_confidence,
            "pose_required_frames": required,
            "stand_sit_required_frames": stand_sit_required,
            "stand_sit_stable_count": count if stand_sit_active and candidate in {"STANDING", "SITTING"} else 0,
            "sit_minus_stand_margin": sit_minus_stand_margin,
            "moving_override_stable_count": moving_override_count,
            "moving_override_required": moving_override_required,
            "moving_override_state": moving_override_state,
            "moving_override_reason": moving_override_reason,
            "moving_override_blocked_by_body_still": moving_override_blocked_by_body_still,
            "moving_translation_displacement_m": translation["displacement_m"],
            "moving_translation_confirmed": translation["confirmed"],
            "strong_stand_sit": strong_stand_sit,
            "strong_stand_sit_margin": strong_margin,
            "stand_to_sit_gate": stand_to_sit_gate,
            "stand_to_sit_conf": stand_to_sit_conf,
            "stand_to_sit_margin": stand_to_sit_margin,
            "stand_to_sit_stable_count": stand_to_sit_count,
            "stand_to_sit_required": stand_to_sit_required,
            "stand_to_sit_quality_ok": stand_to_sit_quality_ok,
            "sitting_relative_gate_state": sitting_relative_gate_state,
            "sitting_relative_gate_stable_count": sitting_relative_count,
            "sitting_relative_gate_required_frames": sitting_relative_required,
            "sitting_relative_gate_passed": sitting_relative_passed,
            "sitting_relative_gate_min_prob": self.sitting_relative_min_prob,
            "sitting_relative_gate_margin": self.sitting_relative_margin,
            "sitting_relative_gate_range_min_m": self.sitting_relative_range_min_m,
            "sitting_relative_gate_range_ok": sitting_relative_range_ok,
            "sitting_relative_standing_veto_prob": self.sitting_relative_standing_veto_prob,
            "sitting_relative_standing_veto_margin": self.sitting_relative_standing_veto_margin,
            "sitting_relative_standing_veto_ok": sitting_relative_standing_veto_ok,
            "sit_to_stand_recovery_count": sit_to_stand_recovery_count,
            "sit_to_stand_recovery_required": sit_to_stand_recovery_required,
            "final_display_pose": displayed_label,
            "display_status": status,
            "transition_reason": transition_reason,
        }

    def _translation_evidence(
        self,
        *,
        tid: int,
        target_position: tuple[float, float, float],
        horizontal_speed: float,
        motion_state: str,
        raw_label: str,
        smoothed_label: str,
    ) -> dict[str, Any]:
        history = self.translation_history[int(tid)]
        position = tuple(float(value) for value in target_position)
        history.append(position)
        displacement = 0.0
        if len(history) >= 2:
            first = history[0]
            last = history[-1]
            displacement = math.sqrt(
                (last[0] - first[0]) ** 2
                + (last[1] - first[1]) ** 2
                + (last[2] - first[2]) ** 2
            )
        label_motion = str(raw_label).upper() in {"MOVING", "WALKING"} or str(
            smoothed_label
        ).upper() in {"MOVING", "WALKING"}
        velocity_confirmed = str(motion_state).upper() == "MOVING"
        displacement_confirmed = (
            len(history) >= self.moving_translation_window
            and displacement >= self.moving_translation_min_m
        )
        high_speed = (
            float(horizontal_speed or 0.0) >= self.moving_speed_threshold * 1.5
        )
        confirmed = bool(
            displacement_confirmed
            or (velocity_confirmed and (label_motion or high_speed))
        )
        return {
            "confirmed": confirmed,
            "displacement_m": float(displacement),
            "label_motion": label_motion,
            "velocity_confirmed": velocity_confirmed,
        }

    def _update_standing_baseline(
        self,
        *,
        tid: int,
        displayed_label: str,
        stand_sit: dict[str, Any],
        geometry: dict[str, Any],
        target: dict[str, float],
        range_m: float,
        horizontal_speed: float,
        motion_state: str,
    ) -> None:
        if not self.use_standing_baseline:
            return
        if str(displayed_label).upper() != "STANDING":
            return
        if str(stand_sit.get("decision", "")).upper() != "STANDING":
            return
        if float(horizontal_speed or 0.0) > self.sitting_max_speed:
            return
        if str(motion_state).upper() == "MOVING":
            return

        top_z = _optional_float(geometry.get("geom_top_z"))
        centroid_z = _optional_float(geometry.get("geom_centroid_z"))
        height_extent = _optional_float(geometry.get("geom_height"))
        target_z = _optional_float(target.get("pos_z"))
        if top_z is None and target_z is None:
            return

        state = self.standing_baseline.setdefault(
            int(tid),
            {
                "frames": 0,
                "top_z": top_z if top_z is not None else target_z,
                "centroid_z": centroid_z if centroid_z is not None else target_z,
                "height_extent": height_extent,
                "target_z": target_z,
                "range_m": float(range_m or 0.0),
                "confidence": float(stand_sit.get("candidate_confidence", 0.0) or 0.0),
            },
        )
        frames = int(state.get("frames", 0) or 0) + 1
        alpha = 0.15 if frames > 1 else 1.0
        for key, value in (
            ("top_z", top_z if top_z is not None else target_z),
            ("centroid_z", centroid_z if centroid_z is not None else target_z),
            ("height_extent", height_extent),
            ("target_z", target_z),
            ("range_m", float(range_m or 0.0)),
            ("confidence", float(stand_sit.get("candidate_confidence", 0.0) or 0.0)),
        ):
            if value is None:
                continue
            previous = _optional_float(state.get(key))
            state[key] = float(value) if previous is None else previous * (1.0 - alpha) + float(value) * alpha
        state["frames"] = frames

    def _label_z_for_pose(self, final_label: str, target_height: float) -> float:
        label = str(final_label).upper()
        if label in {"SITTING"}:
            top = self.human_model_target_sitting_height
        elif label in {"LYING", "FALLING"}:
            top = 0.50
        else:
            top = self.human_model_target_height
        if target_height > 0 and label not in {"LYING", "FALLING"}:
            top = max(top, float(target_height))
        return float(self.ground_z + top + self.label_z_offset)

    def _model_asset_for_label(self, final_label: str) -> str:
        label = str(final_label).upper()
        if label == "SITTING":
            return "human_sitting.obj"
        if label in {"LYING", "FALLING"}:
            return "human_lying.obj"
        return "human_standing.obj"

    def _model_scale_for_label(self, final_label: str) -> float:
        label = str(final_label).upper()
        if label == "SITTING":
            return self.human_model_target_sitting_height
        if label in {"LYING", "FALLING"}:
            return self.human_model_target_lying_length
        return self.human_model_target_height

    def _reset_stale_tracks(self, frame_num: int, seen_tids: set[int]) -> None:
        stale: list[int] = []
        for tid, last_seen in self.last_seen_frame.items():
            if tid in seen_tids:
                continue
            if frame_num - last_seen > STALE_TRACK_FRAMES:
                stale.append(tid)
        for tid in stale:
            self.reset_tid(tid)

    def _debug_print(self, frame_num: int, num_targets: int, results: dict[int, dict]) -> None:
        if not self.debug or frame_num % 30 != 0:
            return
        print(
            f"[pose-debug] frame={frame_num} targets={num_targets} active_pose={len(results)}",
            flush=True,
        )
        for tid in sorted(results):
            item = results[tid]
            range_m = item.get("range_m")
            range_text = f"{range_m:.2f}" if isinstance(range_m, (int, float)) else "NA"
            print(
                "[pose] "
                f"tid={tid} idx={item.get('track_index', '-')} "
                f"assoc={item.get('assoc_mode', '-')} "
                f"points_total={item.get('points_total', 0)} "
                f"geom_pts={item.get('geom_pts', item['num_points'])} "
                f"geom_quality={item.get('geom_quality', '-')} "
                f"pts={item['num_points']} window={item['window_count']}/8 "
                f"raw={item['raw_label']} {item['raw_confidence']:.2f} "
                f"smooth={item['smoothed_label']} {item['smoothed_confidence']:.2f} "
                f"cand={item.get('candidate_label', item['final_label'])} "
                f"display={item.get('displayed_label', item['final_label'])} "
                f"candidate_conf={item.get('candidate_confidence', 0.0):.2f} "
                f"candidate_stable={item.get('candidate_stable_count', item.get('display_stability_count', 0))}/"
                f"{item.get('pose_required_frames', item.get('display_stability_required', self.display_stability_frames))} "
                f"pose_min_conf={item.get('pose_min_confidence', 0.0):.2f} "
                f"pose_required_frames={item.get('pose_required_frames', item.get('display_stability_required', self.display_stability_frames))} "
                f"stand_prob={item.get('stand_prob', 0.0):.2f} "
                f"sit_prob={item.get('sit_prob', 0.0):.2f} "
                f"sit_minus_stand={item.get('sit_minus_stand_margin', 0.0):.2f} "
                f"stand_sit_margin={item.get('stand_sit_margin', 0.0):.2f} "
                f"stand_sit_zone={item.get('stand_sit_zone', item.get('range_zone', 'unknown'))} "
                f"stand_sit_decision={item.get('stand_sit_decision', 'NA')} "
                f"stand_sit_required_frames={item.get('stand_sit_required_frames', 0)} "
                f"stand_sit_stable={item.get('stand_sit_stable_count', 0)}/"
                f"{item.get('stand_sit_required_frames', 0)} "
                f"moving_override_stable={item.get('moving_override_stable_count', 0)}/"
                f"{item.get('moving_override_required', 0)} "
                f"moving_override_state={item.get('moving_override_state', 'NONE')} "
                f"moving_override_reason={item.get('moving_override_reason', '')} "
                f"moving_blocked_body_still={item.get('moving_override_blocked_by_body_still', False)} "
                f"translation_m={_fmt_float(item.get('moving_translation_displacement_m'))} "
                f"translation_confirmed={item.get('moving_translation_confirmed', False)} "
                f"stand_to_sit_gate={item.get('stand_to_sit_gate', 'NA')} "
                f"stand_to_sit_conf={item.get('stand_to_sit_conf', 0.0):.2f} "
                f"stand_to_sit_margin={item.get('stand_to_sit_margin', 0.0):.2f} "
                f"stand_to_sit_stable={item.get('stand_to_sit_stable_count', 0)}/"
                f"{item.get('stand_to_sit_required', 0)} "
                f"stand_to_sit_quality_ok={item.get('stand_to_sit_quality_ok', False)} "
                f"sitting_relative_gate={item.get('sitting_relative_gate_state', 'NA')} "
                f"sitting_relative_stable={item.get('sitting_relative_gate_stable_count', 0)}/"
                f"{item.get('sitting_relative_gate_required_frames', 0)} "
                f"sitting_relative_passed={item.get('sitting_relative_gate_passed', False)} "
                f"sitting_relative_range_ok={item.get('sitting_relative_gate_range_ok', False)} "
                f"sitting_relative_veto_ok={item.get('sitting_relative_standing_veto_ok', False)} "
                f"sit_to_stand_recovery={item.get('sit_to_stand_recovery_count', 0)}/"
                f"{item.get('sit_to_stand_recovery_required', 0)} "
                f"range_m={range_text} "
                f"range_zone={item.get('range_zone', 'unknown')} "
                f"geom_centroid_z={_fmt_float(item.get('geom_centroid_z'))} "
                f"geom_top_z={_fmt_float(item.get('geom_top_z'))} "
                f"geom_bottom_z={_fmt_float(item.get('geom_bottom_z'))} "
                f"geom_height={_fmt_float(item.get('geom_height'))} "
                f"target_z={_fmt_float(item.get('target_z'))} "
                f"floor_z={_fmt_float(item.get('floor_z'))} "
                f"cal_target_z={_fmt_float(item.get('cal_target_z'))} "
                f"baseline_ready={item.get('baseline_ready', False)} "
                f"baseline_frames={item.get('baseline_frames', 0)} "
                f"baseline_top_z={_fmt_float(item.get('baseline_top_z'))} "
                f"baseline_centroid_z={_fmt_float(item.get('baseline_centroid_z'))} "
                f"height_drop={_fmt_float(item.get('height_drop_from_baseline'))} "
                f"centroid_drop={_fmt_float(item.get('centroid_drop'))} "
                f"target_z_drop={_fmt_float(item.get('target_z_drop'))} "
                f"geometry_decision={item.get('geometry_decision', 'NA')} "
                f"geometry_reason={item.get('geometry_reason', 'NA')} "
                f"status={item.get('display_status', '')} "
                f"fall={item.get('fall_gate_passed', False)}:"
                f"{item.get('fall_gate_reason', '')} "
                f"sit={item.get('sitting_gate_passed', False)}:"
                f"{item.get('sitting_gate_reason', '')} "
                f"reason={item.get('transition_reason', '')} "
                f"final_reason={item.get('final_reason', item.get('transition_reason', ''))} "
                f"quality={item['quality']}",
                flush=True,
            )

    def _init_logging(self, log_dir, cfg_path, cli_port, data_port) -> None:
        log_root = Path(log_dir).expanduser().resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        self._log_path = log_root / "pose_predictions_ui.csv"
        metadata_path = log_root / "pose_ui_metadata.json"

        self._log_file = self._log_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "time",
            "frame",
            "tid",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "horizontal_speed",
            "vertical_speed",
            "height_drop",
            "range_m",
            "range_zone",
            "num_points",
            "track_index",
            "assoc_mode",
            "points_total",
            "points_by_target_index",
            "points_by_nearest",
            "geom_pts",
            "geom_quality",
            "geom_centroid_z",
            "geom_top_z",
            "geom_bottom_z",
            "geom_height",
            "geom_floor_centroid_z",
            "target_z",
            "cal_target_z",
            "floor_z",
            "selected_num_points",
            "quality",
            "low_quality",
            "window_ready",
            "window_count",
            "window_age",
            "raw_label",
            "raw_confidence",
            "smoothed_label",
            "smoothed_confidence",
            "ml_top_label",
            "ml_top_confidence",
            "candidate_label",
            "candidate_confidence",
            "final_label",
            "final_confidence",
            "displayed_label",
            "displayed_confidence",
            "display_stability_count",
            "display_stability_required",
            "display_stability_ratio",
            "candidate_stable_count",
            "pose_min_confidence",
            "pose_required_frames",
            "stand_prob",
            "sit_prob",
            "sit_minus_stand_margin",
            "stand_sit_margin",
            "stand_sit_zone",
            "stand_sit_decision",
            "stand_sit_reason",
            "stand_sit_strong",
            "stand_sit_strong_margin",
            "stand_sit_required_frames",
            "stand_sit_stable_count",
            "moving_override_stable_count",
            "moving_override_required",
            "moving_override_state",
            "moving_override_reason",
            "moving_override_blocked_by_body_still",
            "moving_translation_displacement_m",
            "moving_translation_confirmed",
            "stand_to_sit_gate",
            "stand_to_sit_conf",
            "stand_to_sit_margin",
            "stand_to_sit_stable_count",
            "stand_to_sit_required",
            "stand_to_sit_quality_ok",
            "sitting_relative_gate_state",
            "sitting_relative_gate_stable_count",
            "sitting_relative_gate_required_frames",
            "sitting_relative_gate_passed",
            "sitting_relative_gate_min_prob",
            "sitting_relative_gate_margin",
            "sitting_relative_gate_range_min_m",
            "sitting_relative_gate_range_ok",
            "sitting_relative_standing_veto_prob",
            "sitting_relative_standing_veto_margin",
            "sitting_relative_standing_veto_ok",
            "sit_to_stand_recovery_count",
            "sit_to_stand_recovery_required",
            "baseline_ready",
            "baseline_frames",
            "baseline_top_z",
            "baseline_centroid_z",
            "height_drop_from_baseline",
            "centroid_drop",
            "target_z_drop",
            "geometry_decision",
            "geometry_reason",
            "geometry_range_threshold",
            "display_status",
            "transition_reason",
            "final_display_pose",
            "fall_gate_passed",
            "fall_gate_reason",
            "sitting_gate_passed",
            "sitting_gate_reason",
            "stability_count",
            "stability_required",
            "stability_ratio",
            "motion_state",
            "model_asset_used",
            "model_scale",
            "ground_z",
        ] + [f"prob_{name}" for name in self.class_names]
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=fieldnames)
        self._log_writer.writeheader()

        metadata = {
            "model_path": self.model_path,
            "cfg_path": str(cfg_path) if cfg_path is not None else "",
            "cli_port": cli_port or "",
            "data_port": data_port or "",
            "class_names": self.class_names,
            "feature_order": features.FEATURE_NAMES_22,
            "flatten_order": "channel-major, 22 features by 8 frames",
            "smoothing_window": self.smoothing_window,
            "min_confidence": self.min_confidence,
            "unknown_confidence": self.unknown_confidence,
            "moving_speed_threshold": self.moving_speed_threshold,
            "moving_confirm_frames": self.moving_confirm_frames,
            "fall_height_drop_threshold": self.fall_height_drop_threshold,
            "fall_vertical_speed_threshold": self.fall_vertical_speed_threshold,
            "fall_high_confidence": self.fall_high_confidence,
            "fall_min_height_drop_with_high_confidence": (
                self.fall_min_height_drop_with_high_confidence
            ),
            "min_associated_points_for_inference": (
                self.min_associated_points_for_inference
            ),
            "allow_target_only": self.allow_target_only,
            "pose_3d_labels": self.enable_3d_labels,
            "pose_3d_label_format": self.label_format,
            "pose_3d_label_z_offset": self.label_z_offset,
            "pose_3d_label_min_confidence": self.label_min_confidence,
            "pose_3d_label_max_distance": self.label_max_distance,
            "pose_3d_label_debug": self.label_debug,
            "display_stability_frames": self.display_stability_frames,
            "display_min_confidence": self.display_min_confidence,
            "display_hysteresis": self.display_hysteresis,
            "display_stability_ratio": self.display_stability_ratio,
            "falling_fast_update": self.falling_fast_update,
            "falling_stability_frames": self.falling_stability_frames,
            "fall_stability_frames": self.fall_stability_frames,
            "sitting_stability_frames": self.sitting_stability_frames,
            "standing_stability_frames": self.standing_stability_frames,
            "lying_stability_frames": self.lying_stability_frames,
            "moving_stability_frames": self.moving_stability_frames,
            "unknown_stability_frames": self.unknown_stability_frames,
            "sitting_stability_ratio": self.sitting_stability_ratio,
            "sitting_min_confidence": self.sitting_min_confidence,
            "sitting_max_speed": self.sitting_max_speed,
            "standing_min_confidence": self.standing_min_confidence,
            "lying_min_confidence": self.lying_min_confidence,
            "falling_min_confidence": self.falling_min_confidence,
            "moving_min_confidence": self.moving_min_confidence,
            "pose_min_confidence_by_pose": self.pose_min_confidence_by_pose,
            "pose_stability_frames_by_pose": self.pose_stability_frames_by_pose,
            "range_near_max": self.range_near_max,
            "range_mid_max": self.range_mid_max,
            "stand_sit_margin_by_zone": self.stand_sit_margin_by_zone,
            "stand_to_sit_frames_by_zone": self.stand_to_sit_frames_by_zone,
            "sit_to_stand_frames_by_zone": self.sit_to_stand_frames_by_zone,
            "unknown_to_stand_sit_frames_by_zone": self.unknown_to_stand_sit_frames_by_zone,
            "moving_override_frames_by_zone": self.moving_override_frames_by_zone,
            "strong_stand_sit_margin_by_zone": self.strong_stand_sit_margin_by_zone,
            "moving_require_translation": self.moving_require_translation,
            "moving_translation_window": self.moving_translation_window,
            "moving_translation_min_m": self.moving_translation_min_m,
            "sensor_height_m": self.sensor_height_m,
            "sensor_pitch_deg": self.sensor_pitch_deg,
            "sensor_roll_deg": self.sensor_roll_deg,
            "sensor_yaw_deg": self.sensor_yaw_deg,
            "use_sensor_calibration": self.use_sensor_calibration,
            "floor_z_m": self.floor_z_m,
            "assoc_method": self.assoc_method,
            "assoc_nearest_radius_m": self.assoc_nearest_radius_m,
            "assoc_nearest_z_min": self.assoc_nearest_z_min,
            "assoc_nearest_z_max": self.assoc_nearest_z_max,
            "assoc_min_points_good": self.assoc_min_points_good,
            "use_standing_baseline": self.use_standing_baseline,
            "standing_baseline_min_frames": self.standing_baseline_min_frames,
            "sitting_drop_by_zone": self.sitting_drop_by_zone,
            "sitting_drop_min_sit_prob": self.sitting_drop_min_sit_prob,
            "sitting_drop_centroid_m": self.sitting_drop_centroid_m,
            "sitting_drop_top_m": self.sitting_drop_top_m,
            "sitting_drop_target_z_m": self.sitting_drop_target_z_m,
            "stand_to_sit_min_confidence": self.stand_to_sit_min_confidence,
            "stand_to_sit_margin": self.stand_to_sit_margin,
            "stand_to_sit_frames": self.stand_to_sit_frames,
            "stand_to_sit_allow_target_only": self.stand_to_sit_allow_target_only,
            "sitting_relative_gate": self.sitting_relative_gate,
            "sitting_relative_range_min_m": self.sitting_relative_range_min_m,
            "sitting_relative_min_prob": self.sitting_relative_min_prob,
            "sitting_relative_margin": self.sitting_relative_margin,
            "sitting_relative_frames": self.sitting_relative_frames,
            "sitting_relative_standing_veto_prob": self.sitting_relative_standing_veto_prob,
            "sitting_relative_standing_veto_margin": self.sitting_relative_standing_veto_margin,
            "moving_override_require_body_translation_for_sitting": (
                self.moving_override_require_body_translation_for_sitting
            ),
            "sit_to_stand_recovery_margin": self.sit_to_stand_recovery_margin,
            "sit_to_stand_recovery_frames": self.sit_to_stand_recovery_frames,
            "ground_z": self.ground_z,
            "human_model_target_height": self.human_model_target_height,
            "human_model_target_sitting_height": self.human_model_target_sitting_height,
            "human_model_target_lying_length": self.human_model_target_lying_length,
            "human_model_stale_frames": self.human_model_stale_frames,
            "human_model_ghost_distance_m": self.human_model_ghost_distance_m,
            "human_model_confirm_frames": self.human_model_confirm_frames,
            "human_model_confirm_min_geom_pts": self.human_model_confirm_min_geom_pts,
            "human_model_confirm_min_quality_frames": (
                self.human_model_confirm_min_quality_frames
            ),
            "human_model_confirmed_grace_frames": (
                self.human_model_confirmed_grace_frames
            ),
            "human_model_bad_evidence_demote_frames": (
                self.human_model_bad_evidence_demote_frames
            ),
            "human_model_ghost_min_bad_frames": self.human_model_ghost_min_bad_frames,
            "human_model_ghost_no_points_frames": (
                self.human_model_ghost_no_points_frames
            ),
            "human_model_show_provisional": self.human_model_show_provisional,
            "human_model_show_suspect": self.human_model_show_suspect,
            "normalization_enabled": bool(getattr(self.model, "normalization_enabled", False)),
            "scaler_path": str(getattr(self.model, "scaler_path", "") or ""),
            "date_time": datetime.now().isoformat(timespec="seconds"),
            "notes": (
                "Model was trained on TI IWRL6432 Pose/Fall data. "
                "Live IWR6843 accuracy must be validated. MOVING is derived "
                "from speed, not from the ML class output."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _write_log_rows(self, results: dict[int, dict]) -> None:
        if self._log_writer is None:
            return
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        for tid in sorted(results):
            item = results[tid]
            probs = item["probabilities"]
            self._log_writer.writerow(
                {
                    "time": timestamp,
                    "frame": item["frame"],
                    "tid": item["tid"],
                    "x": item["x"],
                    "y": item["y"],
                    "z": item["z"],
                    "vx": item["vx"],
                    "vy": item["vy"],
                    "vz": item["vz"],
                    "horizontal_speed": item["horizontal_speed"],
                    "vertical_speed": item["vertical_speed"],
                    "height_drop": item["height_drop"],
                    "range_m": item.get("range_m", ""),
                    "range_zone": item.get("range_zone", ""),
                    "num_points": item["num_points"],
                    "track_index": item.get("track_index", ""),
                    "assoc_mode": item.get("assoc_mode", ""),
                    "points_total": item.get("points_total", ""),
                    "points_by_target_index": item.get("points_by_target_index", ""),
                    "points_by_nearest": item.get("points_by_nearest", ""),
                    "geom_pts": item.get("geom_pts", ""),
                    "geom_quality": item.get("geom_quality", ""),
                    "geom_centroid_z": item.get("geom_centroid_z", ""),
                    "geom_top_z": item.get("geom_top_z", ""),
                    "geom_bottom_z": item.get("geom_bottom_z", ""),
                    "geom_height": item.get("geom_height", ""),
                    "geom_floor_centroid_z": item.get("geom_floor_centroid_z", ""),
                    "target_z": item.get("target_z", ""),
                    "cal_target_z": item.get("cal_target_z", ""),
                    "floor_z": item.get("floor_z", ""),
                    "selected_num_points": item.get("selected_num_points", item["num_points"]),
                    "quality": item.get("quality", ""),
                    "low_quality": item["low_quality"],
                    "window_ready": item["window_ready"],
                    "window_count": item.get("window_count", item["window_age"]),
                    "window_age": item["window_age"],
                    "raw_label": item["raw_label"],
                    "raw_confidence": item["raw_confidence"],
                    "smoothed_label": item["smoothed_label"],
                    "smoothed_confidence": item["smoothed_confidence"],
                    "ml_top_label": item.get("ml_top_label", item["smoothed_label"]),
                    "ml_top_confidence": item.get(
                        "ml_top_confidence", item["smoothed_confidence"]
                    ),
                    "candidate_label": item.get("candidate_label", ""),
                    "candidate_confidence": item.get("candidate_confidence", 0.0),
                    "final_label": item["final_label"],
                    "final_confidence": item.get("final_confidence", 0.0),
                    "displayed_label": item.get("displayed_label", item.get("final_label", "")),
                    "displayed_confidence": item.get("displayed_confidence", item.get("final_confidence", 0.0)),
                    "display_stability_count": item.get("display_stability_count", 0),
                    "display_stability_required": item.get("display_stability_required", self.display_stability_frames),
                    "display_stability_ratio": item.get("display_stability_ratio", 0.0),
                    "candidate_stable_count": item.get("candidate_stable_count", item.get("display_stability_count", 0)),
                    "pose_min_confidence": item.get("pose_min_confidence", 0.0),
                    "pose_required_frames": item.get("pose_required_frames", item.get("display_stability_required", self.display_stability_frames)),
                    "stand_prob": item.get("stand_prob", 0.0),
                    "sit_prob": item.get("sit_prob", 0.0),
                    "sit_minus_stand_margin": item.get("sit_minus_stand_margin", 0.0),
                    "stand_sit_margin": item.get("stand_sit_margin", 0.0),
                    "stand_sit_zone": item.get("stand_sit_zone", item.get("range_zone", "")),
                    "stand_sit_decision": item.get("stand_sit_decision", "NA"),
                    "stand_sit_reason": item.get("stand_sit_reason", ""),
                    "stand_sit_strong": item.get("stand_sit_strong", False),
                    "stand_sit_strong_margin": item.get("stand_sit_strong_margin", 0.0),
                    "stand_sit_required_frames": item.get("stand_sit_required_frames", 0),
                    "stand_sit_stable_count": item.get("stand_sit_stable_count", 0),
                    "moving_override_stable_count": item.get("moving_override_stable_count", 0),
                    "moving_override_required": item.get("moving_override_required", 0),
                    "moving_override_state": item.get("moving_override_state", "NONE"),
                    "moving_override_reason": item.get("moving_override_reason", ""),
                    "moving_override_blocked_by_body_still": item.get("moving_override_blocked_by_body_still", False),
                    "moving_translation_displacement_m": item.get("moving_translation_displacement_m", 0.0),
                    "moving_translation_confirmed": item.get("moving_translation_confirmed", False),
                    "stand_to_sit_gate": item.get("stand_to_sit_gate", "NA"),
                    "stand_to_sit_conf": item.get("stand_to_sit_conf", 0.0),
                    "stand_to_sit_margin": item.get("stand_to_sit_margin", 0.0),
                    "stand_to_sit_stable_count": item.get("stand_to_sit_stable_count", 0),
                    "stand_to_sit_required": item.get("stand_to_sit_required", 0),
                    "stand_to_sit_quality_ok": item.get("stand_to_sit_quality_ok", False),
                    "sitting_relative_gate_state": item.get("sitting_relative_gate_state", "NA"),
                    "sitting_relative_gate_stable_count": item.get("sitting_relative_gate_stable_count", 0),
                    "sitting_relative_gate_required_frames": item.get("sitting_relative_gate_required_frames", self.sitting_relative_frames),
                    "sitting_relative_gate_passed": item.get("sitting_relative_gate_passed", False),
                    "sitting_relative_gate_min_prob": item.get("sitting_relative_gate_min_prob", self.sitting_relative_min_prob),
                    "sitting_relative_gate_margin": item.get("sitting_relative_gate_margin", self.sitting_relative_margin),
                    "sitting_relative_gate_range_min_m": item.get("sitting_relative_gate_range_min_m", self.sitting_relative_range_min_m),
                    "sitting_relative_gate_range_ok": item.get("sitting_relative_gate_range_ok", False),
                    "sitting_relative_standing_veto_prob": item.get("sitting_relative_standing_veto_prob", self.sitting_relative_standing_veto_prob),
                    "sitting_relative_standing_veto_margin": item.get("sitting_relative_standing_veto_margin", self.sitting_relative_standing_veto_margin),
                    "sitting_relative_standing_veto_ok": item.get("sitting_relative_standing_veto_ok", False),
                    "sit_to_stand_recovery_count": item.get("sit_to_stand_recovery_count", 0),
                    "sit_to_stand_recovery_required": item.get("sit_to_stand_recovery_required", self.sit_to_stand_recovery_frames),
                    "baseline_ready": item.get("baseline_ready", False),
                    "baseline_frames": item.get("baseline_frames", 0),
                    "baseline_top_z": item.get("baseline_top_z", ""),
                    "baseline_centroid_z": item.get("baseline_centroid_z", ""),
                    "height_drop_from_baseline": item.get("height_drop_from_baseline", ""),
                    "centroid_drop": item.get("centroid_drop", ""),
                    "target_z_drop": item.get("target_z_drop", ""),
                    "geometry_decision": item.get("geometry_decision", "NA"),
                    "geometry_reason": item.get("geometry_reason", "NA"),
                    "geometry_range_threshold": item.get("geometry_range_threshold", 0.0),
                    "display_status": item.get("display_status", ""),
                    "transition_reason": item.get("transition_reason", ""),
                    "final_display_pose": item.get("final_display_pose", item.get("displayed_label", item.get("final_label", ""))),
                    "fall_gate_passed": item.get("fall_gate_passed", False),
                    "fall_gate_reason": item.get("fall_gate_reason", ""),
                    "sitting_gate_passed": item.get("sitting_gate_passed", False),
                    "sitting_gate_reason": item.get("sitting_gate_reason", ""),
                    "stability_count": item.get("stability_count", item.get("display_stability_count", 0)),
                    "stability_required": item.get("stability_required", item.get("display_stability_required", self.display_stability_frames)),
                    "stability_ratio": item.get("stability_ratio", item.get("display_stability_ratio", 0.0)),
                    "motion_state": item["motion_state"],
                    "model_asset_used": item.get("model_asset_used", ""),
                    "model_scale": item.get("model_scale", ""),
                    "ground_z": item.get("ground_z", self.ground_z),
                    **{f"prob_{name}": probs.get(name, 0.0) for name in self.class_names},
                }
            )
        if time.time() % 1.0 < 0.1 and self._log_file is not None:
            self._log_file.flush()


def _percent_text(confidence: Any) -> str:
    try:
        percent = float(confidence) * 100.0
    except Exception:
        return "-"
    if not math.isfinite(percent):
        return "-"
    if abs(percent - round(percent)) < 0.05:
        return f"{percent:.0f}"
    return f"{percent:.1f}"


def _fmt_float(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "NA"
    return f"{number:.2f}"


def _rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    try:
        array = np.asarray(value)
    except Exception:
        return []
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    rows: list[list[float]] = []
    for row in array:
        try:
            rows.append([float(item) for item in list(row)])
        except Exception:
            continue
    return rows


def _flat_values(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        array = np.asarray(value).reshape(-1)
    except Exception:
        return []
    values: list[float] = []
    for item in array:
        try:
            values.append(float(item))
        except Exception:
            continue
    return values


def _float_at(row: list[float], index: int, default: float = 0.0) -> float:
    try:
        return float(row[index])
    except Exception:
        return float(default)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _height_by_tid(value: Any) -> dict[int, float]:
    heights: dict[int, float] = {}
    for row in _rows(value):
        if len(row) < 2:
            continue
        try:
            heights[int(row[0])] = float(row[1])
        except Exception:
            continue
    return heights


def _track_position_by_tid(value: Any) -> dict[int, tuple[float, float, float]]:
    positions: dict[int, tuple[float, float, float]] = {}
    for row in _rows(value):
        if len(row) < 4:
            continue
        try:
            positions[int(row[0])] = (float(row[1]), float(row[2]), float(row[3]))
        except Exception:
            continue
    return positions
