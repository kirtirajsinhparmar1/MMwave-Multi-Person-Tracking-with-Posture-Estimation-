"""Launch the vendored TI Industrial Visualizer in People Tracking mode.

This file intentionally keeps the visualization path TI-owned: it imports the
vendored TI `gui_core.Window`, selects xWR6843 / 3D People Tracking, and calls
the same connect/config callbacks that the TI buttons use.
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from types import MethodType
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
VENDOR_ROOT = PROJECT_DIR / "ti_style_vendor"
VENDOR_COMMON = VENDOR_ROOT / "common"
VENDOR_INDUSTRIAL = VENDOR_ROOT / "Industrial_Visualizer"
DEFAULT_CFG = (
    REPO_ROOT
    / "source"
    / "ti"
    / "examples"
    / "Industrial_and_Personal_Electronics"
    / "People_Tracking"
    / "3D_People_Tracking"
    / "chirp_configs"
    / "ODS_6m_default.cfg"
)
DEFAULT_POSE_MODEL = (
    PROJECT_DIR
    / "model_experiments"
    / "outputs"
    / "ti_4class_clean_recording_robust_1600_fast"
    / "ti_pose_model.onnx"
)
DEFAULT_RGB_REPO = (
    REPO_ROOT
    / "RGB Posture Estmation"
    / "Human-Falling-Detect-Tracks"
)
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vendored TI Industrial Visualizer People Tracking UI."
    )
    parser.add_argument("--cli", default="COM7", help="CLI/config COM port.")
    parser.add_argument("--data", default="COM6", help="Data COM port.")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Path to .cfg file.")
    parser.add_argument(
        "--out",
        default=str(PROJECT_DIR / "logs" / "ti_style_ui_test1"),
        help="Output directory reserved for logs/saved TI data.",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Only prefill the TI UI; do not open COM ports or send the cfg.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print vendored paths, defaults, parser frames, and plot update calls.",
    )
    parser.add_argument(
        "--disable-gl-text",
        dest="disable_gl_text",
        action="store_true",
        default=None,
        help="Disable TI GLTextItem labels. Defaults on when the PySide6 shim is used.",
    )
    parser.add_argument(
        "--enable-gl-text",
        dest="disable_gl_text",
        action="store_false",
        help="Try TI GLTextItem labels even when the PySide6 shim is used.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch UI without hardware auto-start. Synthetic frames are not implemented.",
    )
    parser.add_argument(
        "--enable-rgb-panel",
        action="store_true",
        help="Add a live RGB camera panel beside the TI People Tracking view.",
    )
    parser.add_argument(
        "--rgb-source",
        default="0",
        help="OpenCV camera index or video path for the RGB panel.",
    )
    parser.add_argument(
        "--rgb-camera-backend",
        choices=("auto", "dshow", "msmf", "v4l2"),
        default="auto",
        help="OpenCV VideoCapture backend for camera sources.",
    )
    parser.add_argument(
        "--rgb-list-cameras",
        action="store_true",
        help="Probe RGB camera indices, print available cameras, and exit when used without the RGB UI.",
    )
    parser.add_argument(
        "--rgb-camera-probe-max-index",
        type=int,
        default=10,
        help="Highest OpenCV camera index to probe for --rgb-list-cameras or --rgb-prefer-external.",
    )
    parser.add_argument(
        "--rgb-prefer-external",
        action="store_true",
        help="Prefer the first available RGB camera index greater than 0; fall back to --rgb-source.",
    )
    parser.add_argument(
        "--rgb-width",
        type=int,
        default=640,
        help="Requested RGB camera capture width.",
    )
    parser.add_argument(
        "--rgb-height",
        type=int,
        default=480,
        help="Requested RGB camera capture height.",
    )
    parser.add_argument(
        "--rgb-fps",
        type=float,
        default=30,
        help="Requested RGB camera frame rate.",
    )
    parser.add_argument(
        "--rgb-mirror",
        action="store_true",
        help="Mirror RGB camera frames horizontally.",
    )
    parser.add_argument(
        "--enable-rgb-posture",
        action="store_true",
        help="Run the RGB detector/pose/tracker/action overlay inside the RGB panel.",
    )
    parser.add_argument(
        "--rgb-repo",
        default=str(DEFAULT_RGB_REPO),
        help="Path to the Human-Falling-Detect-Tracks RGB repo.",
    )
    parser.add_argument(
        "--rgb-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for RGB posture inference.",
    )
    parser.add_argument(
        "--rgb-detection-input-size",
        type=int,
        default=384,
        help="YOLO detector square input size; must be divisible by 32.",
    )
    parser.add_argument(
        "--rgb-pose-input-size",
        default="224x160",
        help="FastPose input size as HEIGHTxWIDTH; each value must be divisible by 32.",
    )
    parser.add_argument(
        "--rgb-pose-backbone",
        choices=("res50", "res101"),
        default="res50",
        help="FastPose backbone for RGB posture mode.",
    )
    parser.add_argument(
        "--rgb-show-skeleton",
        action="store_true",
        help="Draw tracked pose skeletons in RGB posture mode.",
    )
    parser.add_argument(
        "--rgb-show-detected",
        action="store_true",
        help="Draw raw detector boxes in RGB posture mode.",
    )
    parser.add_argument(
        "--rgb-no-action",
        action="store_true",
        help="Skip TSSTG action recognition in RGB posture mode.",
    )
    parser.add_argument(
        "--combined-session",
        action="store_true",
        help=(
            "Enable the normal combined experiment mode: RGB panel, RGB posture, "
            "combined logging, and combined status panel."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional combined experiment session id. Defaults to session_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Root directory for combined session logs. Defaults to ../logs.",
    )
    parser.add_argument(
        "--combined-log",
        action="store_true",
        help="Enable structured multimodal logging for mmWave and RGB outputs.",
    )
    parser.add_argument(
        "--combined-status-panel",
        action="store_true",
        help="Show a simple live mmWave/RGB summary under the RGB panel.",
    )
    parser.add_argument(
        "--rgb-log-keypoints",
        action="store_true",
        help="Log one RGB keypoint row per joint per track.",
    )
    parser.add_argument(
        "--rgb-log-frames",
        action="store_true",
        help="Create an RGB annotated frame folder; image saving is not implemented yet.",
    )
    parser.add_argument(
        "--rgb-record-video",
        action="store_true",
        help="Record the already-annotated RGB panel frames to a video file.",
    )
    parser.add_argument(
        "--rgb-video-output",
        default="",
        help=(
            "Optional output path for --rgb-record-video. Defaults to the combined "
            "session videos folder or logs/rgb_annotated_<timestamp>.mp4."
        ),
    )
    parser.add_argument(
        "--rgb-video-fps",
        type=float,
        default=0,
        help="Video FPS for --rgb-record-video. Use 0 for source FPS or 20 FPS fallback.",
    )
    parser.add_argument(
        "--rgb-video-codec",
        default="mp4v",
        help="OpenCV fourcc codec for --rgb-record-video. Defaults to mp4v.",
    )
    parser.add_argument(
        "--rgb-video-max-queue",
        type=int,
        default=120,
        help="Maximum queued annotated RGB frames before video frames are dropped.",
    )
    parser.add_argument(
        "--mmwave-log-points",
        action="store_true",
        help="Log mmWave point-cloud rows in combined sessions.",
    )
    parser.add_argument(
        "--enable-pose",
        action="store_true",
        help="Enable live ONNX pose classification per tracker TID.",
    )
    parser.add_argument(
        "--pose-model",
        default=str(DEFAULT_POSE_MODEL),
        help="Path to TI Pose/Fall ONNX model.",
    )
    parser.add_argument(
        "--pose-smoothing-window",
        type=int,
        default=7,
        help="Number of recent probability vectors to average per TID.",
    )
    parser.add_argument(
        "--pose-min-confidence",
        type=float,
        default=0.55,
        help="Minimum smoothed confidence before quality is marked LOW_CONF.",
    )
    parser.add_argument(
        "--pose-unknown-confidence",
        type=float,
        default=0.45,
        help="Smoothed confidence below this becomes UNKNOWN in the final pose label.",
    )
    parser.add_argument(
        "--pose-moving-speed-threshold",
        type=float,
        default=0.18,
        help="Horizontal speed threshold in m/s for derived MOVING state.",
    )
    parser.add_argument(
        "--pose-moving-confirm-frames",
        type=int,
        default=4,
        help="Consecutive frames above speed threshold before final label becomes MOVING.",
    )
    parser.add_argument(
        "--pose-fall-height-drop-threshold",
        type=float,
        default=0.35,
        help="Recent z-height drop threshold in meters for FALLING safety override.",
    )
    parser.add_argument(
        "--pose-fall-vertical-speed-threshold",
        type=float,
        default=0.35,
        help="Vertical speed threshold in m/s used to allow FALLING display.",
    )
    parser.add_argument(
        "--pose-fall-high-confidence",
        type=float,
        default=0.85,
        help="High ML confidence threshold used with mild height drop for FALLING.",
    )
    parser.add_argument(
        "--pose-fall-min-height-drop-with-high-confidence",
        type=float,
        default=0.20,
        help="Minimum height drop required when FALLING is accepted by high confidence.",
    )
    parser.add_argument(
        "--pose-min-associated-points-for-inference",
        type=int,
        default=1,
        help="Minimum associated points needed to add a frame to the pose window.",
    )
    parser.add_argument(
        "--pose-allow-target-only",
        action="store_true",
        help="Allow pose inference with zero associated points by zero-padding point slots.",
    )
    parser.add_argument(
        "--pose-3d-labels",
        action="store_true",
        help="Show per-TID posture labels above tracked targets in the 3D plot.",
    )
    parser.add_argument(
        "--pose-3d-label-format",
        default="{tid} | {final_label} {confidence_percent}%",
        help="Format string for 3D pose labels.",
    )
    parser.add_argument(
        "--pose-3d-label-z-offset",
        type=float,
        default=0.35,
        help="Meters to place the 3D pose label above the target/height box.",
    )
    parser.add_argument(
        "--pose-3d-label-min-confidence",
        type=float,
        default=0.45,
        help="Compatibility option; pose labels still show available predictions.",
    )
    parser.add_argument(
        "--pose-3d-label-max-distance",
        type=float,
        default=None,
        help="Optional maximum 3D distance in meters for showing pose labels.",
    )
    parser.add_argument(
        "--pose-3d-label-debug",
        action="store_true",
        help="Show warmup/extra confidence detail in 3D pose labels.",
    )
    parser.add_argument(
        "--pose-debug",
        action="store_true",
        help="Print pose classification status every 30 frames.",
    )
    parser.add_argument(
        "--pose-log",
        action="store_true",
        help="Write pose_predictions_ui.csv and pose_ui_metadata.json under --out.",
    )
    parser.add_argument(
        "--pose-log-associated-points",
        action="store_true",
        default=False,
        help="Write per-frame per-TID associated point rows to mmwave_associated_points.csv under --out.",
    )
    parser.add_argument(
        "--pose-associated-points-max-per-tid",
        type=int,
        default=64,
        help="Maximum associated points to write per TID per frame when associated point logging is enabled.",
    )
    parser.add_argument(
        "--pose-associated-points-format",
        choices=["csv"],
        default="csv",
        help="Associated point log format. CSV is currently supported.",
    )
    parser.add_argument(
        "--allow-missing-scaler",
        action="store_true",
        help="Allow a normalized pose model to run without scaler files. Debug use only.",
    )
    parser.add_argument(
        "--pose-human-models",
        action="store_true",
        help="Draw simple 3D human posture OBJ models for tracked people.",
    )
    parser.add_argument(
        "--pose-human-model-dir",
        default="ui_human_pose_models",
        help="Directory containing human_standing.obj, human_sitting.obj, and human_lying.obj.",
    )
    parser.add_argument(
        "--pose-human-model-mode",
        choices=["replace_box", "overlay_box", "model_only"],
        default="overlay_box",
        help="How human models relate to TI target boxes.",
    )
    parser.add_argument(
        "--pose-human-model-scale",
        type=float,
        default=1.0,
        help="Global scale multiplier for human posture models.",
    )
    parser.add_argument(
        "--pose-human-model-target-height",
        type=float,
        default=1.70,
        help="Standing/MOVING model target height in meters before global scale.",
    )
    parser.add_argument(
        "--pose-human-model-target-sitting-height",
        type=float,
        default=1.20,
        help="Sitting model target height in meters before global scale.",
    )
    parser.add_argument(
        "--pose-human-model-target-lying-length",
        type=float,
        default=1.70,
        help="Lying/FALLING model target body length in meters before global scale.",
    )
    parser.add_argument(
        "--pose-human-model-height-scale",
        default="auto",
        help="Use 'auto' for per-pose physical scaling, or provide an extra numeric multiplier.",
    )
    parser.add_argument(
        "--pose-human-model-debug",
        action="store_true",
        help="Print per-frame human model placement details.",
    )
    parser.add_argument(
        "--pose-human-model-stale-frames",
        type=int,
        default=10,
        help="Frames to keep a missing human model TID before removing its GL item.",
    )
    parser.add_argument(
        "--pose-human-model-ghost-distance-m",
        type=float,
        default=0.75,
        help="Legacy/debug distance setting; active ghosts are gated by track validation.",
    )
    parser.add_argument(
        "--pose-human-model-confirm-frames",
        type=int,
        default=5,
        help="Frames a new radar TID must persist before full human-model rendering.",
    )
    parser.add_argument(
        "--pose-human-model-confirm-min-geom-pts",
        type=int,
        default=3,
        help="Minimum associated geometry points for one good human-model confirmation frame.",
    )
    parser.add_argument(
        "--pose-human-model-confirm-min-quality-frames",
        type=int,
        default=3,
        help="Good evidence frames required in the confirmation window.",
    )
    parser.add_argument(
        "--pose-human-model-confirmed-grace-frames",
        type=int,
        default=30,
        help="Short weak-evidence grace period for already confirmed active tracks.",
    )
    parser.add_argument(
        "--pose-human-model-bad-evidence-demote-frames",
        type=int,
        default=60,
        help=(
            "Legacy/debug threshold for bad-evidence handling on unconfirmed tracks. "
            "Confirmed active tracks are retained and only disappear after lost/stale cleanup."
        ),
    )
    parser.add_argument(
        "--pose-human-model-ghost-min-bad-frames",
        type=int,
        default=8,
        help="Consecutive bad-evidence frames before a provisional active track is suspect.",
    )
    parser.add_argument(
        "--pose-human-model-ghost-no-points-frames",
        type=int,
        default=8,
        help="Consecutive no-point frames before a provisional active track is suspect.",
    )
    parser.add_argument(
        "--pose-human-model-show-provisional",
        action="store_true",
        help="Debug only: show provisional tracks without strong posture labels.",
    )
    parser.add_argument(
        "--pose-human-model-show-suspect",
        action="store_true",
        help="Debug only: show suspect ghost tracks without strong posture labels.",
    )
    parser.add_argument(
        "--pose-human-model-opacity",
        type=float,
        default=1.0,
        help="Human model opacity from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--pose-human-model-fallback",
        choices=["box", "standing", "none"],
        default="box",
        help="Fallback for unknown/warmup labels or model-load failure.",
    )
    parser.add_argument(
        "--pose-ground-z",
        type=float,
        default=0.0,
        help="World Z coordinate of the floor/ground used to anchor human models.",
    )
    parser.add_argument(
        "--pose-ground-plane",
        action="store_true",
        help="Draw a subtle ground plane/floor at --pose-ground-z.",
    )
    parser.add_argument(
        "--pose-ground-plane-size",
        type=float,
        default=8.0,
        help="Ground plane size in meters.",
    )
    parser.add_argument(
        "--no-pose-ground-plane-grid",
        dest="pose_ground_plane_grid",
        action="store_false",
        default=True,
        help="Disable the grid overlay on the pose ground plane.",
    )
    parser.add_argument(
        "--pose-ground-plane-alpha",
        type=float,
        default=0.18,
        help="Ground plane opacity from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--pose-display-stability-frames",
        type=int,
        default=16,
        help="Frames used for display-only pose stability/hysteresis.",
    )
    parser.add_argument(
        "--pose-display-min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence for a candidate pose to become the stable displayed pose.",
    )
    parser.add_argument(
        "--no-pose-display-hysteresis",
        dest="pose_display_hysteresis",
        action="store_false",
        default=True,
        help="Disable display-only pose hysteresis.",
    )
    parser.add_argument(
        "--pose-display-stability-ratio",
        type=float,
        default=0.70,
        help="Majority ratio required over the display-stability window.",
    )
    parser.add_argument(
        "--no-pose-falling-fast-update",
        dest="pose_falling_fast_update",
        action="store_false",
        default=True,
        help="Disable the shorter display threshold for high-confidence FALLING.",
    )
    parser.add_argument(
        "--pose-fall-stability-frames",
        "--pose-falling-stability-frames",
        dest="pose_fall_stability_frames",
        type=int,
        default=4,
        help="Display-stability frames required for high-confidence FALLING.",
    )
    parser.add_argument(
        "--pose-standing-stability-frames",
        type=int,
        default=12,
        help="Display-stability frames required for STANDING.",
    )
    parser.add_argument(
        "--pose-sitting-stability-frames",
        type=int,
        default=8,
        help="Display-stability frames required for SITTING.",
    )
    parser.add_argument(
        "--pose-lying-stability-frames",
        type=int,
        default=14,
        help="Display-stability frames required for LYING.",
    )
    parser.add_argument(
        "--pose-moving-stability-frames",
        type=int,
        default=4,
        help="Display-stability frames required for MOVING.",
    )
    parser.add_argument(
        "--pose-unknown-stability-frames",
        type=int,
        default=6,
        help="Display-stability frames required for UNKNOWN.",
    )
    parser.add_argument(
        "--pose-sitting-stability-ratio",
        type=float,
        default=0.50,
        help="Majority ratio required for SITTING display updates.",
    )
    parser.add_argument(
        "--pose-sitting-min-confidence",
        type=float,
        default=0.45,
        help="Minimum confidence for SITTING display updates.",
    )
    parser.add_argument(
        "--pose-sitting-max-speed",
        type=float,
        default=0.25,
        help="Maximum horizontal speed in m/s for accepting SITTING.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-min-confidence",
        type=float,
        default=0.65,
        help="Minimum SITTING confidence required before displayed STANDING can switch to SITTING.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-margin",
        type=float,
        default=0.15,
        help="Minimum SITTING minus STANDING probability margin required for STANDING to SITTING.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-frames",
        type=int,
        default=12,
        help="Consecutive gated SITTING frames required before displayed STANDING switches to SITTING.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-allow-target-only",
        action="store_true",
        default=False,
        help="Allow target-only or NO_POINTS evidence to switch displayed STANDING to SITTING.",
    )
    parser.add_argument(
        "--pose-sitting-relative-gate",
        dest="pose_sitting_relative_gate",
        action="store_true",
        default=True,
        help=(
            "Enable the offline-sweep-selected relative sitting transition route: "
            "SITTING can replace displayed STANDING when sit probability consistently "
            "exceeds stand probability at the configured range, margin, and stability."
        ),
    )
    parser.add_argument(
        "--pose-disable-sitting-relative-gate",
        dest="pose_sitting_relative_gate",
        action="store_false",
        help="Disable the relative sit-vs-stand transition route.",
    )
    parser.add_argument(
        "--pose-sitting-relative-range-min-m",
        type=float,
        default=3.0,
        help=(
            "Minimum target range for the refined relative sitting gate. The default "
            "protects near-range standing frames found in offline regression mining."
        ),
    )
    parser.add_argument(
        "--pose-sitting-relative-min-prob",
        type=float,
        default=0.55,
        help=(
            "Minimum SITTING probability for the refined relative sitting gate; "
            "selected by offline sweep, not random threshold tuning."
        ),
    )
    parser.add_argument(
        "--pose-sitting-relative-margin",
        type=float,
        default=0.12,
        help="Minimum SITTING minus STANDING probability margin for the relative sitting gate.",
    )
    parser.add_argument(
        "--pose-sitting-relative-frames",
        type=int,
        default=16,
        help="Consecutive frames required before the relative sitting gate can switch STANDING to SITTING.",
    )
    parser.add_argument(
        "--pose-sitting-relative-standing-veto-prob",
        type=float,
        default=0.50,
        help="Block the relative sitting gate when STANDING probability is at or above this value.",
    )
    parser.add_argument(
        "--pose-sitting-relative-standing-veto-margin",
        type=float,
        default=0.05,
        help="Block the relative sitting gate when STANDING exceeds SITTING by this margin.",
    )
    parser.add_argument(
        "--pose-sit-to-stand-recovery-margin",
        type=float,
        default=0.10,
        help="Minimum STANDING minus SITTING probability margin for SITTING lock recovery.",
    )
    parser.add_argument(
        "--pose-sit-to-stand-recovery-frames",
        type=int,
        default=6,
        help="Consecutive STANDING-favoring frames required to recover from displayed SITTING.",
    )
    parser.add_argument(
        "--pose-standing-min-confidence",
        type=float,
        default=0.70,
        help="Minimum confidence for STANDING display updates.",
    )
    parser.add_argument(
        "--pose-lying-min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence for LYING display updates.",
    )
    parser.add_argument(
        "--pose-falling-min-confidence",
        type=float,
        default=0.70,
        help="Minimum confidence for FALLING display updates.",
    )
    parser.add_argument(
        "--pose-moving-min-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for MOVING display updates.",
    )
    parser.add_argument(
        "--pose-range-near-max",
        type=float,
        default=2.0,
        help="Maximum target range in meters for near stand/sit calibration.",
    )
    parser.add_argument(
        "--pose-range-mid-max",
        type=float,
        default=4.0,
        help="Maximum target range in meters for mid stand/sit calibration; farther targets use far.",
    )
    parser.add_argument(
        "--pose-stand-sit-near-margin",
        type=float,
        default=0.06,
        help="Standing/sitting probability margin required at near range.",
    )
    parser.add_argument(
        "--pose-stand-sit-mid-margin",
        type=float,
        default=0.10,
        help="Standing/sitting probability margin required at mid range.",
    )
    parser.add_argument(
        "--pose-stand-sit-far-margin",
        type=float,
        default=0.15,
        help="Standing/sitting probability margin required at far range.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-near-frames",
        type=int,
        default=6,
        help="Stable frames required for STANDING to SITTING at near range.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-mid-frames",
        type=int,
        default=8,
        help="Stable frames required for STANDING to SITTING at mid range.",
    )
    parser.add_argument(
        "--pose-stand-to-sit-far-frames",
        type=int,
        default=12,
        help="Stable frames required for STANDING to SITTING at far range.",
    )
    parser.add_argument(
        "--pose-sit-to-stand-near-frames",
        type=int,
        default=8,
        help="Stable frames required for SITTING to STANDING at near range.",
    )
    parser.add_argument(
        "--pose-sit-to-stand-mid-frames",
        type=int,
        default=10,
        help="Stable frames required for SITTING to STANDING at mid range.",
    )
    parser.add_argument(
        "--pose-sit-to-stand-far-frames",
        type=int,
        default=14,
        help="Stable frames required for SITTING to STANDING at far range.",
    )
    parser.add_argument(
        "--pose-moving-override-near-frames",
        type=int,
        default=3,
        help="Consecutive motion frames required before MOVING overrides stand/sit at near range.",
    )
    parser.add_argument(
        "--pose-moving-override-mid-frames",
        type=int,
        default=4,
        help="Consecutive motion frames required before MOVING overrides stand/sit at mid range.",
    )
    parser.add_argument(
        "--pose-moving-override-far-frames",
        type=int,
        default=5,
        help="Consecutive motion frames required before MOVING overrides stand/sit at far range.",
    )
    parser.add_argument(
        "--pose-strong-stand-sit-near-margin",
        type=float,
        default=0.12,
        help="Strong stand/sit margin at near range; blocks speed-only MOVING override.",
    )
    parser.add_argument(
        "--pose-strong-stand-sit-mid-margin",
        type=float,
        default=0.18,
        help="Strong stand/sit margin at mid range; blocks speed-only MOVING override.",
    )
    parser.add_argument(
        "--pose-strong-stand-sit-far-margin",
        type=float,
        default=0.25,
        help="Strong stand/sit margin at far range; blocks speed-only MOVING override.",
    )
    parser.add_argument(
        "--pose-moving-require-translation",
        dest="pose_moving_require_translation",
        action="store_true",
        default=True,
        help="Require translation evidence before MOVING overrides strong stand/sit.",
    )
    parser.add_argument(
        "--no-pose-moving-require-translation",
        dest="pose_moving_require_translation",
        action="store_false",
        help="Allow sustained speed-only MOVING override for strong stand/sit.",
    )
    parser.add_argument(
        "--pose-moving-override-require-body-translation-for-sitting",
        dest="pose_moving_override_require_body_translation_for_sitting",
        action="store_true",
        default=True,
        help=(
            "While relative sitting evidence is stable, require body/target "
            "translation before MOVING can override SITTING."
        ),
    )
    parser.add_argument(
        "--pose-disable-moving-override-body-translation-guard",
        dest="pose_moving_override_require_body_translation_for_sitting",
        action="store_false",
        help="Disable the sitting-specific MOVING override body-translation guard.",
    )
    parser.add_argument(
        "--pose-moving-translation-window",
        type=int,
        default=8,
        help="History window for translation-confirmed MOVING override.",
    )
    parser.add_argument(
        "--pose-moving-translation-min-m",
        type=float,
        default=0.25,
        help="Minimum displacement in meters for translation-confirmed MOVING override.",
    )
    parser.add_argument(
        "--pose-sensor-height-m",
        type=float,
        default=1.25,
        help="Sensor mount height used for opt-in pose calibration debug values.",
    )
    parser.add_argument(
        "--pose-sensor-pitch-deg",
        type=float,
        default=0.0,
        help="Sensor pitch in degrees for opt-in pose calibration debug values.",
    )
    parser.add_argument(
        "--pose-sensor-roll-deg",
        type=float,
        default=0.0,
        help="Sensor roll in degrees for opt-in pose calibration debug values.",
    )
    parser.add_argument(
        "--pose-sensor-yaw-deg",
        type=float,
        default=0.0,
        help="Sensor yaw in degrees for opt-in pose calibration debug values.",
    )
    parser.add_argument(
        "--pose-use-sensor-calibration",
        action="store_true",
        help="Use sensor calibration for logged calibrated/floor-relative pose geometry.",
    )
    parser.add_argument(
        "--pose-floor-z-m",
        type=float,
        default=0.0,
        help="Floor z in calibrated pose coordinates.",
    )
    parser.add_argument(
        "--pose-assoc-debug",
        action="store_true",
        help="Print throttled point-association diagnostics.",
    )
    parser.add_argument(
        "--pose-assoc-method",
        choices=("auto", "target_index", "nearest", "hybrid"),
        default="auto",
        help="Point association method for pose features and geometry.",
    )
    parser.add_argument(
        "--pose-assoc-nearest-radius-m",
        type=float,
        default=0.75,
        help="Nearest-neighbor association radius in meters.",
    )
    parser.add_argument(
        "--pose-assoc-nearest-z-min",
        type=float,
        default=-0.5,
        help="Minimum z for nearest-neighbor association.",
    )
    parser.add_argument(
        "--pose-assoc-nearest-z-max",
        type=float,
        default=2.5,
        help="Maximum z for nearest-neighbor association.",
    )
    parser.add_argument(
        "--pose-assoc-min-points-good",
        type=int,
        default=3,
        help="Associated point count considered good for geometry diagnostics.",
    )
    parser.add_argument(
        "--pose-use-standing-baseline",
        action="store_true",
        help="Enable per-TID standing baseline for upright sitting drop logic.",
    )
    parser.add_argument(
        "--pose-standing-baseline-min-frames",
        type=int,
        default=20,
        help="Stable standing frames required before sitting-drop baseline is ready.",
    )
    parser.add_argument(
        "--pose-sitting-drop-near-m",
        type=float,
        default=0.20,
        help="Near-range geometry drop threshold for upright sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-mid-m",
        type=float,
        default=0.25,
        help="Mid-range geometry drop threshold for upright sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-far-m",
        type=float,
        default=0.35,
        help="Far-range geometry drop threshold for upright sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-min-sit-prob",
        type=float,
        default=0.30,
        help="Minimum sitting probability for geometry-supported sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-centroid-m",
        type=float,
        default=0.25,
        help="Centroid drop threshold for geometry-supported sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-top-m",
        type=float,
        default=0.25,
        help="Top-height drop threshold for geometry-supported sitting.",
    )
    parser.add_argument(
        "--pose-sitting-drop-target-z-m",
        type=float,
        default=0.20,
        help="Target-z fallback drop threshold for geometry-supported sitting.",
    )
    args = parser.parse_args()
    if args.combined_session:
        args.enable_rgb_panel = True
        args.enable_rgb_posture = True
        args.combined_log = True
        args.combined_status_panel = True
    if args.enable_rgb_posture and not args.enable_rgb_panel:
        print(
            "[ti-style-warning] --enable-rgb-posture requires --enable-rgb-panel; "
            "enabling the RGB panel.",
            flush=True,
        )
        args.enable_rgb_panel = True
    if (args.combined_log or args.combined_status_panel) and args.session_id is None:
        args.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    if args.pose_log_associated_points and not args.enable_pose:
        print(
            "[ti-style-warning] --pose-log-associated-points requires --enable-pose; "
            "associated point logging will stay disabled.",
            flush=True,
        )
    return args


def parse_rgb_source_value(value: object):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def rgb_backend_flag(cv2_module, backend: str):
    if str(backend).lower() == "auto":
        return None
    names = {
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
    }
    return getattr(cv2_module, names[str(backend).lower()], None)


def open_rgb_capture_for_probe(cv2_module, source, backend: str):
    flag = rgb_backend_flag(cv2_module, backend)
    if str(backend).lower() != "auto" and flag is None:
        raise RuntimeError(f"OpenCV backend is unavailable: {backend}")
    if flag is None:
        return cv2_module.VideoCapture(source)
    return cv2_module.VideoCapture(source, flag)


def probe_rgb_camera_source(source, backend: str) -> dict[str, object]:
    import cv2

    capture = None
    info = {
        "source": str(source),
        "resolved_source": str(source),
        "backend": str(backend),
        "opened": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "name": "UNKNOWN",
    }
    try:
        capture = open_rgb_capture_for_probe(cv2, source, backend)
        opened = bool(capture.isOpened())
        info["opened"] = opened
        if opened:
            info["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            info["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception as exc:
        info["error"] = str(exc)
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
    return info


def probe_rgb_camera_indices(max_index: int, backend: str) -> list[dict[str, object]]:
    max_index = max(0, int(max_index))
    print(f"[RGB_CAMERA] probing camera indices 0..{max_index} backend={backend}", flush=True)
    results = []
    for index in range(max_index + 1):
        info = probe_rgb_camera_source(index, backend)
        results.append(info)
        line = (
            "[RGB_CAMERA] index={index} opened={opened} width={width} height={height} "
            "fps={fps:.1f} backend={backend} name={name}"
        ).format(
            index=index,
            opened=str(bool(info.get("opened"))),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            fps=float(info.get("fps") or 0.0),
            backend=info.get("backend", backend),
            name=info.get("name", "UNKNOWN"),
        )
        if info.get("error"):
            line += f" error={info['error']}"
        print(line, flush=True)
    external_candidates = [
        index
        for index, info in enumerate(results)
        if index > 0 and bool(info.get("opened"))
    ]
    print(f"[RGB_CAMERA] recommended external candidates: {external_candidates}", flush=True)
    return results


def should_exit_after_rgb_camera_list(args: argparse.Namespace) -> bool:
    return (
        bool(args.rgb_list_cameras)
        and not args.enable_rgb_panel
        and not args.enable_rgb_posture
        and not args.combined_log
        and not args.combined_status_panel
        and not args.enable_pose
    )


def resolve_rgb_camera_source(args: argparse.Namespace) -> bool:
    original_source = parse_rgb_source_value(args.rgb_source)
    args._rgb_original_source = original_source
    args._rgb_camera_probe_results = []
    args._rgb_selected_camera_info = {
        "source": str(original_source),
        "resolved_source": str(original_source),
        "backend": args.rgb_camera_backend,
        "opened": None,
        "width": None,
        "height": None,
        "fps": None,
        "prefer_external": bool(args.rgb_prefer_external),
    }

    if args.rgb_list_cameras or args.rgb_prefer_external:
        args._rgb_camera_probe_results = probe_rgb_camera_indices(
            args.rgb_camera_probe_max_index,
            args.rgb_camera_backend,
        )
        if args.rgb_list_cameras and should_exit_after_rgb_camera_list(args):
            return True

    if args.rgb_prefer_external:
        print("[RGB_CAMERA] --rgb-prefer-external enabled", flush=True)
        selected = None
        for index, info in enumerate(args._rgb_camera_probe_results):
            if index > 0 and bool(info.get("opened")):
                selected = (index, info)
                break
        if selected is not None:
            index, info = selected
            args.rgb_source = index
            args._rgb_selected_camera_info = {
                **info,
                "source": str(original_source),
                "resolved_source": str(index),
                "prefer_external": True,
            }
            print(
                "[RGB_CAMERA] selected external camera index={} width={} height={} fps={:.1f}".format(
                    index,
                    int(info.get("width") or 0),
                    int(info.get("height") or 0),
                    float(info.get("fps") or 0.0),
                ),
                flush=True,
            )
            return True
        print(
            f"[RGB_CAMERA] no external camera found; falling back to --rgb-source {original_source}",
            flush=True,
        )

    args.rgb_source = original_source
    if args.rgb_list_cameras and not should_exit_after_rgb_camera_list(args):
        info = probe_rgb_camera_source(original_source, args.rgb_camera_backend)
        args._rgb_selected_camera_info = {
            **info,
            "source": str(original_source),
            "resolved_source": str(original_source),
            "prefer_external": bool(args.rgb_prefer_external),
        }
        if not bool(info.get("opened")):
            print(
                f"[RGB_CAMERA] selected source failed to open; source={original_source}",
                flush=True,
            )
            return False
    else:
        args._rgb_selected_camera_info = {
            "source": str(original_source),
            "resolved_source": str(original_source),
            "backend": args.rgb_camera_backend,
            "opened": None,
            "width": None,
            "height": None,
            "fps": None,
            "prefer_external": bool(args.rgb_prefer_external),
        }

    print(f"[RGB_CAMERA] selected_source={args.rgb_source}", flush=True)
    print(f"[RGB_CAMERA] backend={args.rgb_camera_backend}", flush=True)
    return True


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[ti-style-debug] {message}", flush=True)


def combined_print(message: str) -> None:
    print(f"[COMBINED] {message}", flush=True)


def combined_error(message: str) -> None:
    print(f"[COMBINED][ERROR] {message}", flush=True)


def wall_time_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def resolve_log_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def git_info(path: Path) -> dict[str, object]:
    info = {"commit": None, "branch": None, "dirty": None, "error": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info.update({"commit": commit or None, "branch": branch or None, "dirty": bool(status)})
    except Exception as exc:
        info["error"] = str(exc)
    return info


def create_combined_logger(args: argparse.Namespace, debug: bool):
    if not args.combined_log:
        return None

    try:
        from combined_session_logger import CombinedSessionLogger

        log_root = resolve_log_root(args.log_root)
        combined_print(f"log_root={log_root} session_id={args.session_id}")
        rgb_repo_path = Path(args.rgb_repo).expanduser()
        if not rgb_repo_path.is_absolute():
            rgb_repo_path = (PROJECT_DIR / rgb_repo_path).resolve()
        else:
            rgb_repo_path = rgb_repo_path.resolve()

        mmwave_git = git_info(PROJECT_DIR)
        rgb_git = git_info(rgb_repo_path)
        rgb_camera_info = dict(getattr(args, "_rgb_selected_camera_info", {}) or {})
        metadata = {
            "session_id": args.session_id,
            "created_wall_time_iso": wall_time_iso(),
            "created_monotonic_ns": time.monotonic_ns(),
            "workspace_root": str(REPO_ROOT),
            "mmwave_repo_path": str(PROJECT_DIR),
            "mmwave_git_commit": mmwave_git["commit"],
            "mmwave_git_branch": mmwave_git["branch"],
            "mmwave_git_dirty": mmwave_git["dirty"],
            "rgb_repo_path": str(rgb_repo_path),
            "rgb_git_commit": rgb_git["commit"],
            "rgb_git_branch": rgb_git["branch"],
            "rgb_git_dirty": rgb_git["dirty"],
            "mmwave_cli_port": args.cli,
            "mmwave_data_port": args.data,
            "mmwave_cfg_path": str(resolve_project_path(args.cfg)),
            "mmwave_pose_enabled": bool(args.enable_pose),
            "mmwave_pose_log_associated_points": bool(args.pose_log_associated_points),
            "mmwave_pose_associated_points_max_per_tid": (
                args.pose_associated_points_max_per_tid
            ),
            "mmwave_pose_associated_points_format": args.pose_associated_points_format,
            "mmwave_human_models_enabled": bool(args.pose_human_models),
            "mmwave_human_model_stale_frames": args.pose_human_model_stale_frames,
            "mmwave_human_model_ghost_distance_m": args.pose_human_model_ghost_distance_m,
            "mmwave_human_model_confirm_frames": args.pose_human_model_confirm_frames,
            "mmwave_human_model_confirm_min_geom_pts": (
                args.pose_human_model_confirm_min_geom_pts
            ),
            "mmwave_human_model_confirm_min_quality_frames": (
                args.pose_human_model_confirm_min_quality_frames
            ),
            "mmwave_human_model_confirmed_grace_frames": (
                args.pose_human_model_confirmed_grace_frames
            ),
            "mmwave_human_model_bad_evidence_demote_frames": (
                args.pose_human_model_bad_evidence_demote_frames
            ),
            "mmwave_human_model_ghost_min_bad_frames": (
                args.pose_human_model_ghost_min_bad_frames
            ),
            "mmwave_human_model_ghost_no_points_frames": (
                args.pose_human_model_ghost_no_points_frames
            ),
            "mmwave_human_model_show_provisional": bool(
                args.pose_human_model_show_provisional
            ),
            "mmwave_human_model_show_suspect": bool(args.pose_human_model_show_suspect),
            "rgb_panel_enabled": bool(args.enable_rgb_panel),
            "rgb_posture_enabled": bool(args.enable_rgb_posture),
            "rgb_source": str(getattr(args, "_rgb_original_source", args.rgb_source)),
            "rgb_resolved_source": str(args.rgb_source),
            "rgb_device": args.rgb_device,
            "rgb_camera_backend": args.rgb_camera_backend,
            "rgb_prefer_external": bool(args.rgb_prefer_external),
            "rgb_camera_selected": rgb_camera_info,
            "combined_status_panel": bool(args.combined_status_panel),
            "rgb_log_keypoints": bool(args.rgb_log_keypoints),
            "rgb_log_frames": bool(args.rgb_log_frames),
            "rgb_record_video": bool(args.rgb_record_video),
            "rgb_video_output": str(args.rgb_video_output),
            "rgb_video_fps": args.rgb_video_fps,
            "rgb_video_codec": args.rgb_video_codec,
            "rgb_video_max_queue": args.rgb_video_max_queue,
            "mmwave_log_points": bool(args.mmwave_log_points),
            "notes": "",
        }
        logger = CombinedSessionLogger(
            log_root=log_root,
            session_id=args.session_id,
            metadata=metadata,
            log_rgb_keypoints=args.rgb_log_keypoints,
            log_mmwave_points=args.mmwave_log_points,
        )
        if mmwave_git["error"]:
            logger.log_event("git_info_unavailable", {"repo": "mmwave", "error": mmwave_git["error"]})
        if rgb_git["error"]:
            logger.log_event("git_info_unavailable", {"repo": "rgb", "error": rgb_git["error"]})
        try:
            logger.log_event("rgb_camera_selected", rgb_camera_info)
        except Exception:
            pass
        if args.rgb_log_frames:
            frames_dir = logger.session_dir / "rgb_annotated_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            logger.log_event(
                "rgb_frame_saving_not_implemented",
                {"directory": str(frames_dir)},
            )
        combined_print(f"logger created: {logger.session_dir}")
        debug_print(debug, f"combined session log directory: {logger.session_dir}")
        return logger
    except Exception as exc:
        args.combined_log = False
        combined_error(f"logger creation failed; combined logging disabled: {exc}")
        return None


def _rows(value) -> list:
    if value is None:
        return []
    try:
        return value.tolist()
    except Exception:
        try:
            return list(value)
        except Exception:
            return []


def _float_at(row, index: int, default=None):
    try:
        return float(row[index])
    except Exception:
        return default


def _int_at(row, index: int, default=None):
    value = _float_at(row, index, default)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _height_by_tid(height_data) -> dict[int, dict[str, object]]:
    result = {}
    for row in _rows(height_data):
        tid = _int_at(row, 0)
        if tid is None:
            continue
        max_z = _float_at(row, 1)
        min_z = _float_at(row, 2)
        height = None
        if max_z is not None and min_z is not None:
            height = max_z - min_z
        result[tid] = {"height_max_z_m": max_z, "height_min_z_m": min_z, "height_m": height}
    return result


def _associated_point_counts(points) -> dict[int, int]:
    counts = {}
    for point in _rows(points):
        track_index = _int_at(point, 6)
        if track_index is None or track_index in {253, 254, 255}:
            continue
        counts[track_index] = counts.get(track_index, 0) + 1
    return counts


def _pose_probability(probabilities: dict, *names: str):
    if not isinstance(probabilities, dict):
        return None
    lower_map = {str(key).lower(): value for key, value in probabilities.items()}
    for name in names:
        value = lower_map.get(name.lower())
        if value is not None:
            return value
    return None


def log_mmwave_output(combined_logger, status_panel, output_dict: dict, demo_instance=None) -> None:
    if not isinstance(output_dict, dict):
        return

    now_ns = time.monotonic_ns()
    now_iso = wall_time_iso()
    frame_num = output_dict.get("frameNum")
    tracks = _rows(output_dict.get("trackData"))
    points = _rows(output_dict.get("pointCloud"))
    heights = _height_by_tid(output_dict.get("heightData"))
    point_counts = _associated_point_counts(points)
    error_value = output_dict.get("error", 0)
    parse_ok = int(error_value or 0) == 0

    if combined_logger is not None:
        combined_logger.log_mmwave_frame(
            {
                "host_wall_time_iso": now_iso,
                "host_monotonic_ns": now_ns,
                "source": "mmwave",
                "mmwave_frame_num": frame_num,
                "num_tracks": len(tracks),
                "num_points": len(points),
                "parse_ok": parse_ok,
                "error_count": 0 if parse_ok else 1,
            }
        )
    for track in tracks:
        tid = _int_at(track, 0)
        height = heights.get(tid, {})
        if combined_logger is not None:
            combined_logger.log_mmwave_track(
                {
                    "host_wall_time_iso": now_iso,
                    "host_monotonic_ns": now_ns,
                    "source": "mmwave",
                    "mmwave_frame_num": frame_num,
                    "tid": tid,
                    "x_m": _float_at(track, 1),
                    "y_m": _float_at(track, 2),
                    "z_m": _float_at(track, 3),
                    "vx_mps": _float_at(track, 4),
                    "vy_mps": _float_at(track, 5),
                    "vz_mps": _float_at(track, 6),
                    "ax_mps2": _float_at(track, 7),
                    "ay_mps2": _float_at(track, 8),
                    "az_mps2": _float_at(track, 9),
                    "g": _float_at(track, 10),
                    "confidence": _float_at(track, 11),
                    "num_associated_points": point_counts.get(tid, 0),
                    "height_min_z_m": height.get("height_min_z_m"),
                    "height_max_z_m": height.get("height_max_z_m"),
                    "height_m": height.get("height_m"),
                }
            )
    if combined_logger is not None and getattr(combined_logger, "log_mmwave_points", False):
        for point_index, point in enumerate(points):
            combined_logger.log_mmwave_point(
                {
                    "host_wall_time_iso": now_iso,
                    "host_monotonic_ns": now_ns,
                    "source": "mmwave",
                    "mmwave_frame_num": frame_num,
                    "point_index": point_index,
                    "track_index": _int_at(point, 6),
                    "x_m": _float_at(point, 0),
                    "y_m": _float_at(point, 1),
                    "z_m": _float_at(point, 2),
                    "doppler": _float_at(point, 3),
                    "snr": _float_at(point, 4),
                    "noise": _float_at(point, 5),
                }
            )

    pose_results = getattr(demo_instance, "latestPoseResults", {}) if demo_instance is not None else {}
    pose_summaries = []
    if isinstance(pose_results, dict):
        for tid, pose in pose_results.items():
            if not isinstance(pose, dict):
                continue
            probabilities = pose.get("probabilities", {})
            final_label = pose.get("final_label")
            pose_summaries.append(f"TID {tid} {final_label}")
            if combined_logger is not None:
                combined_logger.log_mmwave_pose(
                    {
                        "host_wall_time_iso": now_iso,
                        "host_monotonic_ns": now_ns,
                        "source": "mmwave",
                        "mmwave_frame_num": frame_num,
                        "tid": tid,
                        "window_ready": bool(pose.get("window_ready", False)),
                        "ml_label": pose.get("smoothed_label") or pose.get("raw_label"),
                        "ml_confidence": pose.get("smoothed_confidence") or pose.get("raw_confidence"),
                        "final_label": final_label,
                        "motion_label": pose.get("motion_state"),
                        "speed_mps": pose.get("horizontal_speed"),
                        "height_drop_flag": bool(float(pose.get("height_drop", 0.0) or 0.0) > 0.0),
                        "quality_flag": pose.get("quality"),
                        "num_points": pose.get("num_points"),
                        "prob_standing": _pose_probability(probabilities, "STANDING", "standing"),
                        "prob_sitting": _pose_probability(probabilities, "SITTING", "sitting"),
                        "prob_lying": _pose_probability(probabilities, "LYING", "lying", "Lying Down"),
                        "prob_falling": _pose_probability(probabilities, "FALLING", "falling", "Fall Down"),
                    }
                )
    if status_panel is not None:
        status_panel.update_mmwave(frame_num, len(tracks), pose_summaries)


def log_rgb_result(combined_logger, status_panel, result: dict, log_keypoints: bool) -> None:
    if not isinstance(result, dict):
        return

    tracks = result.get("tracks") or []
    action_count = sum(1 for track in tracks if track.get("action_label"))
    frame_row = {
        "host_wall_time_iso": result.get("host_wall_time_iso"),
        "host_monotonic_ns": result.get("host_monotonic_ns"),
        "source": result.get("source", "rgb"),
        "rgb_frame_num": result.get("rgb_frame_num"),
        "width": result.get("width"),
        "height": result.get("height"),
        "fps_estimate": result.get("fps_estimate"),
        "frame_read_ok": result.get("frame_read_ok"),
        "num_detections": result.get("num_detections"),
        "num_tracks": result.get("num_tracks", len(tracks)),
        "num_actions": result.get("num_actions", action_count),
        "error_count": len(result.get("errors") or []),
    }
    if combined_logger is not None:
        combined_logger.log_rgb_frame(frame_row)

    action_summaries = []
    for track in tracks:
        track_row = {
            "host_wall_time_iso": result.get("host_wall_time_iso"),
            "host_monotonic_ns": result.get("host_monotonic_ns"),
            "source": result.get("source", "rgb"),
            "rgb_frame_num": result.get("rgb_frame_num"),
            "rgb_track_id": track.get("rgb_track_id"),
            "bbox_x1_px": track.get("bbox_x1_px"),
            "bbox_y1_px": track.get("bbox_y1_px"),
            "bbox_x2_px": track.get("bbox_x2_px"),
            "bbox_y2_px": track.get("bbox_y2_px"),
            "bbox_confidence": track.get("bbox_confidence"),
            "pose_confidence": track.get("pose_confidence"),
            "tracker_state": track.get("tracker_state"),
            "track_age": track.get("track_age"),
            "time_since_update": track.get("time_since_update"),
            "action_window_ready": track.get("action_window_ready"),
            "action_label": track.get("action_label"),
            "action_confidence": track.get("action_confidence"),
        }
        if combined_logger is not None:
            combined_logger.log_rgb_track(track_row)
        if track.get("action_label"):
            action_summaries.append(f"ID {track.get('rgb_track_id')} {track.get('action_label')}")
            probs = track.get("action_probs") or {}
            if combined_logger is not None:
                combined_logger.log_rgb_action(
                    {
                        "host_wall_time_iso": result.get("host_wall_time_iso"),
                        "host_monotonic_ns": result.get("host_monotonic_ns"),
                        "source": result.get("source", "rgb"),
                        "rgb_frame_num": result.get("rgb_frame_num"),
                        "rgb_track_id": track.get("rgb_track_id"),
                        "action_window_ready": track.get("action_window_ready"),
                        "action_label": track.get("action_label"),
                        "action_confidence": track.get("action_confidence"),
                        "prob_standing": probs.get("Standing"),
                        "prob_walking": probs.get("Walking"),
                        "prob_sitting": probs.get("Sitting"),
                        "prob_lying_down": probs.get("Lying Down"),
                        "prob_stand_up": probs.get("Stand up"),
                        "prob_sit_down": probs.get("Sit down"),
                        "prob_fall_down": probs.get("Fall Down"),
                    }
                )
        if combined_logger is not None and log_keypoints:
            for keypoint in track.get("keypoints") or []:
                keypoint_row = {
                    "host_wall_time_iso": result.get("host_wall_time_iso"),
                    "host_monotonic_ns": result.get("host_monotonic_ns"),
                    "source": result.get("source", "rgb"),
                    "rgb_frame_num": result.get("rgb_frame_num"),
                    "rgb_track_id": track.get("rgb_track_id"),
                }
                keypoint_row.update(keypoint)
                combined_logger.log_rgb_keypoint(keypoint_row)
    if status_panel is not None:
        status_panel.update_rgb(result.get("rgb_frame_num"), len(tracks), action_summaries)


class CombinedStatusPanel:
    def __init__(self, widget) -> None:
        self.widget = widget
        self.mmwave_text = "Latest mmWave:\n  frame: -\n  tracks: 0\n  poses: -"
        self.rgb_text = "Latest RGB:\n  frame: -\n  tracks: 0\n  actions: -"
        self._last_render_ns = 0
        self._render_interval_ns = 250_000_000
        self.render()

    def update_mmwave(self, frame_num, tracks_count: int, pose_summaries: list[str]) -> None:
        poses = ", ".join(pose_summaries[:6]) if pose_summaries else "-"
        self.mmwave_text = (
            "Latest mmWave:\n"
            f"  frame: {frame_num if frame_num is not None else '-'}\n"
            f"  tracks: {tracks_count}\n"
            f"  poses: {poses}"
        )
        self.render()

    def update_rgb(self, frame_num, tracks_count: int, action_summaries: list[str]) -> None:
        actions = ", ".join(action_summaries[:6]) if action_summaries else "-"
        self.rgb_text = (
            "Latest RGB:\n"
            f"  frame: {frame_num if frame_num is not None else '-'}\n"
            f"  tracks: {tracks_count}\n"
            f"  actions: {actions}"
        )
        self.render()

    def render(self) -> None:
        now_ns = time.monotonic_ns()
        if self._last_render_ns and now_ns - self._last_render_ns < self._render_interval_ns:
            return
        try:
            self.widget.setPlainText(self.mmwave_text + "\n\n" + self.rgb_text)
            self._last_render_ns = now_ns
        except Exception as exc:
            combined_error(f"status panel update failed: {exc}")


def add_import_paths(debug: bool) -> list[Path]:
    paths = [
        VENDOR_ROOT,
        VENDOR_COMMON,
        VENDOR_INDUSTRIAL,
        VENDOR_COMMON / "Common_Tabs",
        VENDOR_COMMON / "Demo_Classes",
        VENDOR_COMMON / "Demo_Classes" / "Helper_Classes",
    ]
    added: list[Path] = []
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added.append(path)
    for path in added:
        debug_print(debug, f"sys.path added: {path}")
    return added


def check_pyside2_shim(debug: bool) -> bool:
    try:
        import PySide2
        from PySide2 import QtCore
        from PySide6 import __version__ as pyside6_version
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name.startswith("PySide6"):
            raise SystemExit(
                "PySide6 is required for the local PySide2 compatibility shim. "
                "Install it with: python -m pip install -r requirements_ti_style.txt"
            ) from exc
        raise

    print(f"PySide2 compatibility shim resolved to PySide6 {pyside6_version}", flush=True)
    pyside2_path = Path(PySide2.__file__).resolve()
    qtcore_path = Path(QtCore.__file__).resolve()
    debug_print(debug, f"PySide2 shim package: {pyside2_path}")
    debug_print(debug, f"PySide2.QtCore shim: {qtcore_path}")
    return VENDOR_ROOT in pyside2_path.parents


def configure_gl_text(args: argparse.Namespace, using_pyside2_shim: bool, debug: bool) -> bool:
    disable_gl_text = args.disable_gl_text
    if disable_gl_text is None:
        disable_gl_text = using_pyside2_shim

    os.environ["TI_STYLE_DISABLE_GL_TEXT"] = "1" if disable_gl_text else "0"
    if disable_gl_text:
        debug_print(debug, "GL text labels disabled for PySide6 compatibility.")
    else:
        debug_print(debug, "GL text labels enabled.")
    return disable_gl_text


def safe_len(obj) -> int:
    if obj is None:
        return 0
    try:
        return len(obj)
    except Exception:
        return 0


def ensure_vendor_runtime_dirs() -> None:
    (VENDOR_INDUSTRIAL / "cache").mkdir(parents=True, exist_ok=True)
    (VENDOR_INDUSTRIAL / "binData").mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def normalize_windows_com(value: str) -> str:
    text = value.strip()
    if os.name == "nt" and text.upper().startswith("COM"):
        return text[3:]
    return text


def set_combo_text(combo, text: str) -> bool:
    index = combo.findText(text)
    if index >= 0:
        combo.setCurrentIndex(index)
        return True
    return False


def configure_business_demo_list() -> None:
    from demo_defines import BUSINESS_DEMOS, DEVICE_DEMO_DICT

    for device_name in DEVICE_DEMO_DICT:
        DEVICE_DEMO_DICT[device_name]["demos"] = [
            demo
            for demo in DEVICE_DEMO_DICT[device_name]["demos"]
            if demo in BUSINESS_DEMOS["Industrial"]
        ]


def install_debug_hooks(debug: bool, gl_text_disabled: bool) -> None:
    if not debug:
        return

    from Common_Tabs.plot_3d import Plot3D
    from Demo_Classes.people_tracking import PeopleTracking
    from gui_parser import UARTParser
    from gui_threads import updateQTTargetThread3D

    original_send_cfg = UARTParser.sendCfg
    original_read_double = UARTParser.readAndParseUartDoubleCOMPort
    original_update_graph = PeopleTracking.updateGraph
    original_update_point_cloud = Plot3D.updatePointCloud
    original_run = updateQTTargetThread3D.run

    def visual_status(self, output_dict):
        if not isinstance(output_dict, dict):
            return
        frame_num = output_dict.get("frameNum")
        try:
            should_log = int(frame_num) % 30 == 0
        except Exception:
            should_log = False
        if not should_log:
            return

        points = safe_len(output_dict.get("pointCloud"))
        targets = safe_len(output_dict.get("trackData"))
        heights = safe_len(output_dict.get("heightData"))
        point_item = "yes" if getattr(self, "scatter", None) is not None else "no"
        target_markers = "yes" if safe_len(getattr(self, "ellipsoids", None)) else "no"
        boxes = "yes" if (
            safe_len(getattr(self, "boundaryBoxList", None))
            or safe_len(getattr(self, "boundaryBoxViz", None))
        ) else "no"
        gl_text = "disabled" if gl_text_disabled else "enabled"
        print(
            "[ti-style-debug] visual status "
            f"frame={frame_num} points={points} targets={targets} heights={heights} "
            f"pointItem={point_item} targetMarkers={target_markers} boxes={boxes} "
            f"glText={gl_text}",
            flush=True,
        )

    def send_cfg_debug(self, cfg):
        print(f"[ti-style-debug] UARTParser.sendCfg lines={len(cfg)}", flush=True)
        result = original_send_cfg(self, cfg)
        print(
            f"[ti-style-debug] UARTParser.sendCfg complete comError={self.comError}",
            flush=True,
        )
        return result

    def read_double_debug(self, demo):
        output = original_read_double(self, demo)
        frame = output[0] if isinstance(output, tuple) else output
        if isinstance(frame, dict):
            points = safe_len(frame.get("pointCloud"))
            targets = safe_len(frame.get("trackData"))
            heights = safe_len(frame.get("heightData"))
            print(
                "[ti-style-debug] frame "
                f"num={frame.get('frameNum', 'n/a')} points={points} "
                f"targets={targets} heights={heights}",
                flush=True,
            )
        return output

    def update_graph_debug(self, output_dict):
        points = safe_len(output_dict.get("pointCloud")) if isinstance(output_dict, dict) else 0
        targets = safe_len(output_dict.get("trackData")) if isinstance(output_dict, dict) else 0
        print(
            f"[ti-style-debug] PeopleTracking.updateGraph points={points} targets={targets}",
            flush=True,
        )
        result = original_update_graph(self, output_dict)
        visual_status(self, output_dict)
        return result

    def update_point_cloud_debug(self, output_dict):
        points = safe_len(output_dict.get("pointCloud")) if isinstance(output_dict, dict) else 0
        print(f"[ti-style-debug] Plot3D.updatePointCloud points={points}", flush=True)
        return original_update_point_cloud(self, output_dict)

    def run_debug(self):
        points = safe_len(getattr(self, "pointCloud", None))
        targets = safe_len(getattr(self, "targets", None))
        heights = safe_len(getattr(self, "heightData", None))
        print(
            "[ti-style-debug] updateQTTargetThread3D.run "
            f"points={points} targets={targets} heights={heights}",
            flush=True,
        )
        return original_run(self)

    UARTParser.sendCfg = send_cfg_debug
    UARTParser.readAndParseUartDoubleCOMPort = read_double_debug
    PeopleTracking.updateGraph = update_graph_debug
    Plot3D.updatePointCloud = update_point_cloud_debug
    updateQTTargetThread3D.run = run_debug


def import_ti_qt():
    from PySide2.QtCore import QTimer
    from PySide2.QtGui import QColor, QPalette
    from PySide2.QtWidgets import QApplication

    from demo_defines import DEMO_3D_PEOPLE_TRACKING
    from gui_core import Window

    return QApplication, QTimer, QPalette, QColor, Window, DEMO_3D_PEOPLE_TRACKING


def apply_ti_dark_palette(app, QPalette, QColor) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


def configure_window(window, args: argparse.Namespace, demo_name: str, debug: bool) -> None:
    cfg_path = resolve_project_path(args.cfg)
    out_dir = resolve_project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_combo_text(window.deviceList, "xWR6843")
    window.onChangeDevice()
    set_combo_text(window.demoList, demo_name)
    window.onChangeDemo()

    window.cliCom.setText(normalize_windows_com(args.cli))
    window.dataCom.setText(normalize_windows_com(args.data))
    window.filename_edit.setText(str(cfg_path))
    window.core.parser.filepath = out_dir.name
    window.core.parseCfg(str(cfg_path))

    debug_print(debug, f"CLI port default: {args.cli}")
    debug_print(debug, f"Data port default: {args.data}")
    debug_print(debug, f"Cfg path: {cfg_path}")
    debug_print(debug, f"Output directory: {out_dir}")
    debug_print(debug, f"Selected device: {window.core.device}")
    debug_print(debug, f"Selected demo: {window.core.demo}")
    debug_print(debug, f"Selected demo class: {type(window.core.demoClassDict[window.core.demo]).__name__}")


def create_pose_manager_before_qt(args: argparse.Namespace, debug: bool):
    if not args.enable_pose:
        debug_print(debug, "pose model disabled")
        return None

    model_path = resolve_project_path(args.pose_model)
    out_dir = resolve_project_path(args.out)
    pose_debug = args.pose_debug or debug
    if pose_debug:
        print(f"[pose-runtime] resolved model path: {model_path}", flush=True)

    try:
        from pose_model_runtime import prewarm_onnxruntime

        prewarm_onnxruntime(pose_debug)
        from ti_style_pose_overlay import TiStylePoseManager

        pose_manager = TiStylePoseManager(
            model_path=model_path,
            smoothing_window=args.pose_smoothing_window,
            min_confidence=args.pose_min_confidence,
            unknown_confidence=args.pose_unknown_confidence,
            moving_speed_threshold=args.pose_moving_speed_threshold,
            moving_confirm_frames=args.pose_moving_confirm_frames,
            fall_height_drop_threshold=args.pose_fall_height_drop_threshold,
            fall_vertical_speed_threshold=args.pose_fall_vertical_speed_threshold,
            fall_high_confidence=args.pose_fall_high_confidence,
            fall_min_height_drop_with_high_confidence=(
                args.pose_fall_min_height_drop_with_high_confidence
            ),
            min_associated_points_for_inference=(
                args.pose_min_associated_points_for_inference
            ),
            allow_target_only=args.pose_allow_target_only,
            enable_3d_labels=args.pose_3d_labels,
            label_format=args.pose_3d_label_format,
            label_z_offset=args.pose_3d_label_z_offset,
            label_min_confidence=args.pose_3d_label_min_confidence,
            label_max_distance=args.pose_3d_label_max_distance,
            label_debug=args.pose_3d_label_debug,
            enable_human_models=args.pose_human_models,
            human_model_debug=args.pose_human_model_debug,
            human_model_stale_frames=args.pose_human_model_stale_frames,
            human_model_ghost_distance_m=args.pose_human_model_ghost_distance_m,
            human_model_confirm_frames=args.pose_human_model_confirm_frames,
            human_model_confirm_min_geom_pts=args.pose_human_model_confirm_min_geom_pts,
            human_model_confirm_min_quality_frames=(
                args.pose_human_model_confirm_min_quality_frames
            ),
            human_model_confirmed_grace_frames=(
                args.pose_human_model_confirmed_grace_frames
            ),
            human_model_bad_evidence_demote_frames=(
                args.pose_human_model_bad_evidence_demote_frames
            ),
            human_model_ghost_min_bad_frames=args.pose_human_model_ghost_min_bad_frames,
            human_model_ghost_no_points_frames=(
                args.pose_human_model_ghost_no_points_frames
            ),
            human_model_show_provisional=args.pose_human_model_show_provisional,
            human_model_show_suspect=args.pose_human_model_show_suspect,
            display_stability_frames=args.pose_display_stability_frames,
            display_min_confidence=args.pose_display_min_confidence,
            display_hysteresis=args.pose_display_hysteresis,
            display_stability_ratio=args.pose_display_stability_ratio,
            falling_fast_update=args.pose_falling_fast_update,
            fall_stability_frames=args.pose_fall_stability_frames,
            standing_stability_frames=args.pose_standing_stability_frames,
            sitting_stability_frames=args.pose_sitting_stability_frames,
            lying_stability_frames=args.pose_lying_stability_frames,
            moving_stability_frames=args.pose_moving_stability_frames,
            unknown_stability_frames=args.pose_unknown_stability_frames,
            sitting_stability_ratio=args.pose_sitting_stability_ratio,
            sitting_min_confidence=args.pose_sitting_min_confidence,
            sitting_max_speed=args.pose_sitting_max_speed,
            standing_min_confidence=args.pose_standing_min_confidence,
            lying_min_confidence=args.pose_lying_min_confidence,
            falling_min_confidence=args.pose_falling_min_confidence,
            moving_min_confidence=args.pose_moving_min_confidence,
            range_near_max=args.pose_range_near_max,
            range_mid_max=args.pose_range_mid_max,
            stand_sit_near_margin=args.pose_stand_sit_near_margin,
            stand_sit_mid_margin=args.pose_stand_sit_mid_margin,
            stand_sit_far_margin=args.pose_stand_sit_far_margin,
            stand_to_sit_near_frames=args.pose_stand_to_sit_near_frames,
            stand_to_sit_mid_frames=args.pose_stand_to_sit_mid_frames,
            stand_to_sit_far_frames=args.pose_stand_to_sit_far_frames,
            sit_to_stand_near_frames=args.pose_sit_to_stand_near_frames,
            sit_to_stand_mid_frames=args.pose_sit_to_stand_mid_frames,
            sit_to_stand_far_frames=args.pose_sit_to_stand_far_frames,
            moving_override_near_frames=args.pose_moving_override_near_frames,
            moving_override_mid_frames=args.pose_moving_override_mid_frames,
            moving_override_far_frames=args.pose_moving_override_far_frames,
            strong_stand_sit_near_margin=args.pose_strong_stand_sit_near_margin,
            strong_stand_sit_mid_margin=args.pose_strong_stand_sit_mid_margin,
            strong_stand_sit_far_margin=args.pose_strong_stand_sit_far_margin,
            moving_require_translation=args.pose_moving_require_translation,
            moving_translation_window=args.pose_moving_translation_window,
            moving_translation_min_m=args.pose_moving_translation_min_m,
            sensor_height_m=args.pose_sensor_height_m,
            sensor_pitch_deg=args.pose_sensor_pitch_deg,
            sensor_roll_deg=args.pose_sensor_roll_deg,
            sensor_yaw_deg=args.pose_sensor_yaw_deg,
            use_sensor_calibration=args.pose_use_sensor_calibration,
            floor_z_m=args.pose_floor_z_m,
            assoc_debug=args.pose_assoc_debug,
            assoc_method=args.pose_assoc_method,
            assoc_nearest_radius_m=args.pose_assoc_nearest_radius_m,
            assoc_nearest_z_min=args.pose_assoc_nearest_z_min,
            assoc_nearest_z_max=args.pose_assoc_nearest_z_max,
            assoc_min_points_good=args.pose_assoc_min_points_good,
            use_standing_baseline=args.pose_use_standing_baseline,
            standing_baseline_min_frames=args.pose_standing_baseline_min_frames,
            sitting_drop_near_m=args.pose_sitting_drop_near_m,
            sitting_drop_mid_m=args.pose_sitting_drop_mid_m,
            sitting_drop_far_m=args.pose_sitting_drop_far_m,
            sitting_drop_min_sit_prob=args.pose_sitting_drop_min_sit_prob,
            sitting_drop_centroid_m=args.pose_sitting_drop_centroid_m,
            sitting_drop_top_m=args.pose_sitting_drop_top_m,
            sitting_drop_target_z_m=args.pose_sitting_drop_target_z_m,
            stand_to_sit_min_confidence=args.pose_stand_to_sit_min_confidence,
            stand_to_sit_margin=args.pose_stand_to_sit_margin,
            stand_to_sit_frames=args.pose_stand_to_sit_frames,
            stand_to_sit_allow_target_only=args.pose_stand_to_sit_allow_target_only,
            sitting_relative_gate=args.pose_sitting_relative_gate,
            sitting_relative_range_min_m=args.pose_sitting_relative_range_min_m,
            sitting_relative_min_prob=args.pose_sitting_relative_min_prob,
            sitting_relative_margin=args.pose_sitting_relative_margin,
            sitting_relative_frames=args.pose_sitting_relative_frames,
            sitting_relative_standing_veto_prob=(
                args.pose_sitting_relative_standing_veto_prob
            ),
            sitting_relative_standing_veto_margin=(
                args.pose_sitting_relative_standing_veto_margin
            ),
            moving_override_require_body_translation_for_sitting=(
                args.pose_moving_override_require_body_translation_for_sitting
            ),
            sit_to_stand_recovery_margin=args.pose_sit_to_stand_recovery_margin,
            sit_to_stand_recovery_frames=args.pose_sit_to_stand_recovery_frames,
            ground_z=args.pose_ground_z,
            human_model_target_height=args.pose_human_model_target_height,
            human_model_target_sitting_height=args.pose_human_model_target_sitting_height,
            human_model_target_lying_length=args.pose_human_model_target_lying_length,
            debug=pose_debug,
            log_dir=out_dir if args.pose_log else None,
            associated_points_log_dir=(
                out_dir if args.pose_log_associated_points else None
            ),
            associated_points_session_id=args.session_id or out_dir.name,
            associated_points_max_per_tid=args.pose_associated_points_max_per_tid,
            associated_points_format=args.pose_associated_points_format,
            cfg_path=resolve_project_path(args.cfg),
            cli_port=args.cli,
            data_port=args.data,
            allow_missing_scaler=args.allow_missing_scaler,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        message = str(exc)
        if getattr(exc, "name", "") == "onnxruntime":
            message = (
                "onnxruntime is required for --enable-pose. "
                "Install it with: python -m pip install onnxruntime"
            )
        else:
            message = (
                "ONNX Runtime failed to import inside the TI-style UI process. "
                "Standalone import may work, but Qt/PySide DLL load order can break it. "
                "This launcher now preloads ONNX Runtime before Qt; if it still fails, "
                "reinstall onnxruntime or use a torch fallback. "
                f"Original exception: {exc}"
            )
        raise SystemExit(message) from exc

    debug_print(pose_debug, f"pose model loaded before Qt: {model_path}")
    if args.pose_log:
        debug_print(pose_debug, f"pose logging directory: {out_dir}")
    if args.pose_log_associated_points:
        debug_print(
            pose_debug,
            f"associated point logging file: {out_dir / 'mmwave_associated_points.csv'}",
        )
    return pose_manager


def attach_pose_manager(window, pose_manager, args: argparse.Namespace, debug: bool):
    if pose_manager is None:
        return None

    demo_instance = window.core.demoClassDict.get(window.core.demo)
    if demo_instance is None or not hasattr(demo_instance, "setPoseManager"):
        raise RuntimeError("Selected TI demo class does not support pose manager attachment")

    demo_instance.setPoseManager(pose_manager)
    if args.pose_human_models:
        try:
            from human_model_renderer import HumanPoseModelRenderer

            renderer = HumanPoseModelRenderer(
                getattr(demo_instance, "plot_3d"),
                model_dir=resolve_project_path(args.pose_human_model_dir),
                scale=args.pose_human_model_scale,
                height_scale=args.pose_human_model_height_scale,
                target_height=args.pose_human_model_target_height,
                target_sitting_height=args.pose_human_model_target_sitting_height,
                target_lying_length=args.pose_human_model_target_lying_length,
                ground_z=args.pose_ground_z,
                opacity=args.pose_human_model_opacity,
                fallback=args.pose_human_model_fallback,
                debug=args.pose_human_model_debug or debug,
                stale_ttl_frames=args.pose_human_model_stale_frames,
                ghost_distance_m=args.pose_human_model_ghost_distance_m,
            )
            pose_manager._human_model_renderer = renderer
            demo_instance.setPoseHumanModelRenderer(
                renderer,
                mode=args.pose_human_model_mode,
            )
            atexit.register(renderer.clear_all_human_models)
            debug_print(
                debug or args.pose_human_model_debug,
                f"human posture models attached: {resolve_project_path(args.pose_human_model_dir)}",
            )
        except Exception as exc:
            print(
                f"[human-model] disabled; keeping TI target boxes. Reason: {exc}",
                flush=True,
            )
    if args.pose_ground_plane and hasattr(demo_instance, "setPoseGroundPlane"):
        demo_instance.setPoseGroundPlane(
            enabled=True,
            ground_z=args.pose_ground_z,
            size=args.pose_ground_plane_size,
            grid=args.pose_ground_plane_grid,
            alpha=args.pose_ground_plane_alpha,
        )
        debug_print(
            debug or args.pose_debug,
            "[ground-plane] enabled z={} size={}".format(
                args.pose_ground_z,
                args.pose_ground_plane_size,
            ),
        )
    atexit.register(pose_manager.close)
    debug_print(debug or getattr(pose_manager, "debug", False), "[pose-runtime] pose manager attached")
    debug_print(debug or getattr(pose_manager, "debug", False), f"pose manager attached to: {type(demo_instance).__name__}")
    return pose_manager


def clear_pose_human_models(pose_manager) -> None:
    renderer = getattr(pose_manager, "_human_model_renderer", None)
    if renderer is None:
        return
    try:
        if hasattr(renderer, "clear_all_human_models"):
            renderer.clear_all_human_models()
        else:
            renderer.clear()
    except Exception as exc:
        print(f"[human-model-warning] shutdown cleanup failed: {exc}", flush=True)


def resolve_rgb_video_output(args: argparse.Namespace, combined_logger) -> Path | None:
    if not args.rgb_record_video:
        return None

    if args.rgb_video_output:
        path = Path(args.rgb_video_output).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_DIR / path).resolve()

    if combined_logger is not None and hasattr(combined_logger, "session_dir"):
        return Path(combined_logger.session_dir) / "videos" / "rgb_annotated.mp4"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_DIR / "logs" / f"rgb_annotated_{timestamp}.mp4"


def attach_rgb_panel(window, args: argparse.Namespace, debug: bool):
    if not args.enable_rgb_panel:
        debug_print(debug, "RGB panel disabled")
        return None

    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import QSplitter

    from rgb_camera_panel import RgbCameraPanel

    video_output = getattr(args, "_resolved_rgb_video_output", None)
    panel = RgbCameraPanel(
        source=args.rgb_source,
        backend=args.rgb_camera_backend,
        width=args.rgb_width,
        height=args.rgb_height,
        fps=args.rgb_fps,
        mirror=args.rgb_mirror,
        posture_enabled=args.enable_rgb_posture,
        rgb_repo=args.rgb_repo,
        device=args.rgb_device,
        detection_input_size=args.rgb_detection_input_size,
        pose_input_size=args.rgb_pose_input_size,
        pose_backbone=args.rgb_pose_backbone,
        show_skeleton=args.rgb_show_skeleton,
        show_detected=args.rgb_show_detected,
        no_action=args.rgb_no_action,
        record_video=args.rgb_record_video,
        video_output=video_output or "",
        video_fps=args.rgb_video_fps,
        video_codec=args.rgb_video_codec,
        video_max_queue=args.rgb_video_max_queue,
    )
    splitter = QSplitter(Qt.Horizontal)
    window.gridLayout.removeWidget(window.demoTabs)
    splitter.addWidget(window.demoTabs)
    splitter.addWidget(panel)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([1200, 420])

    window.gridLayout.addWidget(splitter, 0, 1, 8, 1)
    window.rgb_splitter = splitter
    window.rgb_camera_panel = panel
    atexit.register(panel.stop)

    debug_print(
        debug,
        "RGB panel attached beside TI demoTabs "
        f"(source={args.rgb_source}, backend={args.rgb_camera_backend})",
    )
    if args.rgb_record_video:
        debug_print(debug, f"RGB annotated video output: {video_output}")
    return panel


def attach_combined_status_panel(rgb_panel, args: argparse.Namespace, debug: bool):
    if not args.combined_status_panel:
        return None
    if rgb_panel is None:
        debug_print(debug, "combined status panel requested, but RGB panel is not enabled")
        return None

    try:
        from PySide2.QtWidgets import QPlainTextEdit, QSizePolicy

        status_widget = QPlainTextEdit()
        status_widget.setReadOnly(True)
        status_widget.setMaximumHeight(150)
        status_widget.setMinimumHeight(115)
        status_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_widget.setStyleSheet(
            "QPlainTextEdit { background: #151515; color: #e6e6e6; "
            "border: 1px solid #3a3a3a; font-family: Consolas, monospace; }"
        )
        rgb_panel.layout().addWidget(status_widget)
        panel = CombinedStatusPanel(status_widget)
        rgb_panel.combined_status_panel = panel
        combined_print("status panel created")
        debug_print(debug, "combined status panel attached under RGB panel")
        return panel
    except Exception as exc:
        combined_error(f"status panel creation failed; continuing without it: {exc}")
        return None


def attach_combined_mmwave_logging(window, combined_logger, status_panel, debug: bool) -> None:
    if combined_logger is None and status_panel is None:
        return

    try:
        original_update_graph = window.core.updateGraph
    except Exception as exc:
        combined_error(f"mmWave logging hook attach failed: {exc}")
        return

    def update_graph_with_combined_logging(core_self, output_dict):
        result = original_update_graph(output_dict)
        try:
            demo_instance = core_self.demoClassDict.get(core_self.demo)
            log_mmwave_output(combined_logger, status_panel, output_dict, demo_instance)
        except Exception as exc:
            if combined_logger is not None:
                combined_logger.log_event("mmwave_log_error", {"error": str(exc)})
            combined_error(f"MMWAVE logging failed: {exc}")
        return result

    try:
        window.core.updateGraph = MethodType(update_graph_with_combined_logging, window.core)
        combined_print("mmwave logging hook attached")
        debug_print(debug, "combined mmWave logging hook attached to gui_core.Window.core.updateGraph")
    except Exception as exc:
        combined_error(f"mmWave logging hook attach failed: {exc}")


def attach_combined_rgb_logging(rgb_panel, combined_logger, status_panel, args: argparse.Namespace, debug: bool) -> None:
    if rgb_panel is None or (combined_logger is None and status_panel is None):
        return

    from PySide2.QtCore import QTimer

    pending_results = deque(maxlen=90)
    state = {"dropped": 0, "last_error_ns": 0}

    def on_rgb_result(result):
        if len(pending_results) == pending_results.maxlen:
            state["dropped"] += 1
        pending_results.append(result)

    def drain_rgb_results():
        drained = 0
        while pending_results and drained < 2:
            result = pending_results.popleft()
            drained += 1
            try:
                log_rgb_result(combined_logger, status_panel, result, args.rgb_log_keypoints)
            except Exception as exc:
                now_ns = time.monotonic_ns()
                if combined_logger is not None:
                    combined_logger.log_event("rgb_log_error", {"error": str(exc)})
                if now_ns - state["last_error_ns"] > 1_000_000_000:
                    combined_error(f"RGB logging failed: {exc}")
                    state["last_error_ns"] = now_ns
        if state["dropped"]:
            dropped = state["dropped"]
            state["dropped"] = 0
            if combined_logger is not None:
                combined_logger.log_event("rgb_log_queue_dropped", {"dropped_results": dropped})

    try:
        timer = QTimer(rgb_panel)
        timer.setInterval(50)
        timer.timeout.connect(drain_rgb_results)
        timer.start()
        rgb_panel._combined_rgb_log_timer = timer
        rgb_panel._combined_rgb_log_queue = pending_results
        rgb_panel.resultReady.connect(on_rgb_result)
        combined_print("rgb logging hook attached")
        debug_print(debug, "combined RGB logging/status hook attached to RGB panel results")
    except Exception as exc:
        combined_error(f"RGB logging hook attach failed: {exc}")
        try:
            if combined_logger is not None:
                combined_logger.log_event("rgb_log_hook_error", {"error": str(exc)})
        except Exception:
            pass


def attach_rgb_video_event_logging(rgb_panel, combined_logger, args: argparse.Namespace, debug: bool) -> None:
    if rgb_panel is None:
        return

    def on_video_event(event_type, payload):
        try:
            if combined_logger is not None:
                combined_logger.log_event(str(event_type), dict(payload or {}))
        except Exception as exc:
            debug_print(debug, f"RGB video event logging failed: {exc}")

    try:
        rgb_panel.videoEvent.connect(on_video_event)
        debug_print(debug, "RGB camera/video event hook attached")
    except Exception as exc:
        combined_error(f"RGB video event hook attach failed: {exc}")


def auto_start(window, debug: bool) -> None:
    debug_print(debug, "auto-start: connecting COM ports via TI Window.onConnect()")
    window.onConnect()
    connected = window.connectStatus.text() == "Connected"
    debug_print(debug, f"auto-start: connectStatus={window.connectStatus.text()}")
    if not connected:
        debug_print(debug, "auto-start: config not sent because COM connection failed")
        return

    debug_print(debug, "auto-start: sending cfg via TI Window.sendCfg()")
    window.sendCfg()
    debug_print(debug, "auto-start: cfg send requested; TI parse timer should now be active")


def main() -> int:
    args = parse_args()
    rgb_source_ok = resolve_rgb_camera_source(args)
    if should_exit_after_rgb_camera_list(args):
        return 0
    if not rgb_source_ok:
        return 2
    rgb_panel = None
    status_panel = None
    if args.combined_log or args.combined_status_panel:
        combined_print(
            "combined_log="
            f"{bool(args.combined_log)} combined_status_panel={bool(args.combined_status_panel)} "
            f"rgb_log_keypoints={bool(args.rgb_log_keypoints)}"
        )
    combined_logger = create_combined_logger(args, args.debug)
    args._resolved_rgb_video_output = resolve_rgb_video_output(args, combined_logger)
    pose_manager = create_pose_manager_before_qt(args, args.debug)
    add_import_paths(args.debug)
    using_pyside2_shim = check_pyside2_shim(args.debug)
    gl_text_disabled = configure_gl_text(args, using_pyside2_shim, args.debug)
    ensure_vendor_runtime_dirs()

    original_cwd = Path.cwd()
    os.chdir(VENDOR_INDUSTRIAL)
    debug_print(args.debug, f"cwd changed from {original_cwd} to {VENDOR_INDUSTRIAL}")

    configure_business_demo_list()
    install_debug_hooks(args.debug, gl_text_disabled)

    QApplication, QTimer, QPalette, QColor, Window, demo_name = import_ti_qt()
    app = QApplication(sys.argv[:1])
    apply_ti_dark_palette(app, QPalette, QColor)

    screen = app.primaryScreen()
    size = screen.size() if screen is not None else []
    window = Window(size=size, title="Industrial Visualizer - TI Style (Vendored)")
    configure_window(window, args, demo_name, args.debug)
    attach_pose_manager(window, pose_manager, args, args.debug)
    rgb_panel = attach_rgb_panel(window, args, args.debug)
    status_panel = attach_combined_status_panel(rgb_panel, args, args.debug)
    attach_combined_rgb_logging(rgb_panel, combined_logger, status_panel, args, args.debug)
    attach_rgb_video_event_logging(rgb_panel, combined_logger, args, args.debug)
    attach_combined_mmwave_logging(window, combined_logger, status_panel, args.debug)
    window.show()

    auto_start_enabled = not args.no_auto_start and not args.demo
    if auto_start_enabled:
        if combined_logger is not None:
            combined_logger.log_event(
                "mmwave_started",
                {"mode": "auto_start", "cli": args.cli, "data": args.data, "cfg": args.cfg},
            )
        QTimer.singleShot(500, lambda: auto_start(window, args.debug))
    else:
        if combined_logger is not None:
            combined_logger.log_event(
                "mmwave_start_skipped",
                {"demo": bool(args.demo), "no_auto_start": bool(args.no_auto_start)},
            )
        debug_print(args.debug, "auto-start disabled; use Connect then Start and Send Configuration")
    if combined_logger is not None and rgb_panel is not None:
        combined_logger.log_event(
            "rgb_worker_started",
            {
                "posture_enabled": bool(args.enable_rgb_posture),
                "source": str(getattr(args, "_rgb_original_source", args.rgb_source)),
                "resolved_source": str(args.rgb_source),
                "backend": args.rgb_camera_backend,
                "prefer_external": bool(args.rgb_prefer_external),
            },
        )

    try:
        return app.exec_()
    finally:
        if rgb_panel is not None:
            if combined_logger is not None:
                combined_logger.log_event("rgb_worker_stopped", {})
            rgb_panel.stop()
        if pose_manager is not None:
            clear_pose_human_models(pose_manager)
            pose_manager.close()
        if combined_logger is not None:
            combined_logger.log_event("mmwave_stopped", {})
            combined_logger.log_event("app_shutdown", {})
            combined_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
