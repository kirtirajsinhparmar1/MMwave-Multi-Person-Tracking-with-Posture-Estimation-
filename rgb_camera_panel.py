"""Qt RGB camera panel for side-by-side mmWave/RGB viewing."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PySide2.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


RgbSource = Union[int, str]


ACTION_PROBABILITY_FIELDS = {
    "Standing": "prob_standing",
    "Walking": "prob_walking",
    "Sitting": "prob_sitting",
    "Lying Down": "prob_lying_down",
    "Stand up": "prob_stand_up",
    "Sit down": "prob_sit_down",
    "Fall Down": "prob_fall_down",
}


def _wall_time_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _track_state_name(track) -> str:
    if track.is_confirmed():
        return "confirmed"
    if track.is_tentative():
        return "tentative"
    if track.is_deleted():
        return "deleted"
    return "unknown"


def _keypoint_records(keypoints: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    records = []
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return records
    for index, row in enumerate(keypoints):
        x_px = float(row[0])
        y_px = float(row[1])
        score = float(row[2]) if keypoints.shape[1] > 2 else None
        records.append(
            {
                "joint_index": index,
                "x_px": x_px,
                "y_px": y_px,
                "score": score,
                "x_norm": x_px / width if width else None,
                "y_norm": y_px / height if height else None,
            }
        )
    return records


def parse_rgb_source(value: object) -> RgbSource:
    """Return an int camera index when possible, otherwise a path/string source."""
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _backend_flag(cv2_module, backend: str) -> Optional[int]:
    backend = backend.lower()
    if backend == "auto":
        return None

    names = {
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
    }
    return getattr(cv2_module, names[backend], None)


def _open_capture(cv2_module, source: RgbSource, backend: str):
    backend_flag = _backend_flag(cv2_module, backend)
    if backend != "auto" and backend_flag is None:
        raise RuntimeError(f"OpenCV backend is unavailable: {backend}")

    if backend_flag is None:
        return cv2_module.VideoCapture(source)
    return cv2_module.VideoCapture(source, backend_flag)


def _configure_capture(cv2_module, capture, source: RgbSource, width: int, height: int, fps: float) -> None:
    if not isinstance(source, int):
        return
    if width > 0:
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        capture.set(cv2_module.CAP_PROP_FPS, fps)
    capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)


def _rgb_array_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


def _parse_pose_size(value: str) -> tuple[int, int]:
    try:
        parts = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError("--rgb-pose-input-size must use HEIGHTxWIDTH, for example 224x160") from exc

    if len(parts) != 2 or any(part <= 0 or part % 32 != 0 for part in parts):
        raise ValueError("--rgb-pose-input-size dimensions must be positive and divisible by 32")
    return parts


def _pose_backbone_name(value: str) -> str:
    return {
        "res50": "resnet50",
        "res101": "resnet101",
    }[value]


def _required_model_files(pose_backbone: str, no_action: bool) -> list[Path]:
    pose_file = (
        Path("Models/sppe/fast_res101_320x256.pth")
        if pose_backbone == "res101"
        else Path("Models/sppe/fast_res50_256x192.pth")
    )
    required = [
        Path("Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg"),
        Path("Models/yolo-tiny-onecls/best-model.pth"),
        pose_file,
    ]
    if not no_action:
        required.append(Path("Models/TSSTG/tsstg-model.pth"))
    return required


def _kpt_to_bbox(kpt: np.ndarray, ex: int = 20) -> np.ndarray:
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        ),
        dtype=np.float32,
    )


def _poses_to_detections(poses, frame_shape, Detection):
    height, width = frame_shape[:2]
    detections = []
    for pose in poses:
        keypoints = pose["keypoints"].detach().cpu().numpy()
        scores = pose["kp_score"].detach().cpu().numpy()
        if (
            keypoints.ndim != 2
            or keypoints.shape[1] != 2
            or scores.ndim != 2
            or scores.shape[0] != keypoints.shape[0]
            or not np.all(np.isfinite(keypoints))
            or not np.all(np.isfinite(scores))
        ):
            continue

        bbox = _kpt_to_bbox(keypoints)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        detection = Detection(
            bbox,
            np.concatenate((keypoints, scores), axis=1),
            float(scores.mean()),
        )
        if detection.is_valid():
            detections.append(detection)
    return detections


def _draw_label(cv2_module, frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2_module.putText(
        frame,
        text,
        (max(0, x), max(16, y)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2_module.LINE_AA,
    )


class RgbCaptureWorker(QObject):
    frameReady = Signal(QImage)
    statusChanged = Signal(str)
    resultReady = Signal(dict)
    finished = Signal()

    def __init__(
        self,
        source: RgbSource,
        backend: str,
        width: int,
        height: int,
        fps: float,
        mirror: bool,
    ) -> None:
        super().__init__()
        self.source = source
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self._running = False

    @Slot()
    def run(self) -> None:
        capture = None
        self._running = True

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - depends on local install
            self.statusChanged.emit(f"OpenCV import failed: {exc}")
            self.finished.emit()
            return

        try:
            capture = _open_capture(cv2, self.source, self.backend)
            if not capture.isOpened():
                self.statusChanged.emit("RGB source unavailable")
                self.finished.emit()
                return

            _configure_capture(cv2, capture, self.source, self.width, self.height, self.fps)
            self.statusChanged.emit(f"RGB source active: {self.source}")
            delay_ms = max(1, int(1000.0 / self.fps)) if self.fps > 0 else 1
            frame_count = 0
            fps_time = time.time()

            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.statusChanged.emit("RGB frame read failed")
                    break

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_count += 1
                now = time.time()
                elapsed = max(now - fps_time, 1e-6)
                fps_estimate = 1.0 / elapsed
                fps_time = now
                height_px, width_px = rgb.shape[:2]
                monotonic = time.monotonic_ns()
                self.resultReady.emit(
                    {
                        "type": "rgb_frame",
                        "schema_version": 1,
                        "rgb_frame_num": frame_count,
                        "host_wall_time_iso": _wall_time_iso(),
                        "host_monotonic_ns": monotonic,
                        "source": str(self.source),
                        "width": width_px,
                        "height": height_px,
                        "fps_estimate": fps_estimate,
                        "frame_read_ok": True,
                        "num_detections": 0,
                        "num_tracks": 0,
                        "tracks": [],
                        "errors": [],
                    }
                )
                self.frameReady.emit(_rgb_array_to_qimage(rgb))
                QThread.msleep(delay_ms)
        except Exception as exc:  # pragma: no cover - defensive UI isolation
            self.statusChanged.emit(f"RGB capture error: {exc}")
        finally:
            if capture is not None:
                capture.release()
            self._running = False
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False


class RgbPostureWorker(QObject):
    frameReady = Signal(QImage)
    statusChanged = Signal(str)
    resultReady = Signal(dict)
    finished = Signal()

    def __init__(
        self,
        source: RgbSource,
        backend: str,
        width: int,
        height: int,
        fps: float,
        mirror: bool,
        rgb_repo: object,
        device: str,
        detection_input_size: int,
        pose_input_size: str,
        pose_backbone: str,
        show_skeleton: bool,
        show_detected: bool,
        no_action: bool,
    ) -> None:
        super().__init__()
        self.source = source
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.rgb_repo = Path(str(rgb_repo)).expanduser()
        self.device = device
        self.detection_input_size = detection_input_size
        self.pose_input_size = pose_input_size
        self.pose_backbone = pose_backbone
        self.show_skeleton = show_skeleton
        self.show_detected = show_detected
        self.no_action = no_action
        self._running = False

    @Slot()
    def run(self) -> None:
        capture = None
        self._running = True

        try:
            import cv2
            import torch
        except Exception as exc:  # pragma: no cover - depends on local install
            self.statusChanged.emit(f"RGB model load failed: import failed: {exc}")
            self.finished.emit()
            return

        try:
            self.statusChanged.emit("Loading RGB posture models...")
            components = self._load_components(torch)
            self.statusChanged.emit("RGB models loaded; opening source...")
        except Exception as exc:
            self.statusChanged.emit(f"RGB model load failed: {exc}")
            self.finished.emit()
            return

        try:
            capture = _open_capture(cv2, self.source, self.backend)
            if not capture.isOpened():
                self.statusChanged.emit("RGB source unavailable")
                self.finished.emit()
                return

            _configure_capture(cv2, capture, self.source, self.width, self.height, self.fps)
            reported_fps = capture.get(cv2.CAP_PROP_FPS)
            self.statusChanged.emit(
                "RGB posture active "
                f"({components['device']}, action={'off' if self.no_action else 'on'}, "
                f"camera_fps={reported_fps:.1f})"
            )

            frame_count = 0
            fps_time = time.time()
            while self._running:
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    self.statusChanged.emit("RGB source unavailable")
                    break

                try:
                    annotated, stats = self._process_frame(cv2, torch, components, bgr)
                except Exception as exc:
                    self.statusChanged.emit(f"RGB inference failed: {exc}")
                    break

                frame_count += 1
                now = time.time()
                elapsed = max(now - fps_time, 1e-6)
                fps = 1.0 / elapsed
                fps_time = now
                stats["result"].update(
                    {
                        "rgb_frame_num": frame_count,
                        "host_wall_time_iso": _wall_time_iso(),
                        "host_monotonic_ns": time.monotonic_ns(),
                        "fps_estimate": fps,
                    }
                )
                if frame_count == 1 or frame_count % 30 == 0:
                    self.statusChanged.emit(
                        "RGB posture frame "
                        f"{frame_count}: detected={stats['detected']} "
                        f"poses={stats['poses']} tracks={stats['tracks']} "
                        f"action={stats['action']} fps={fps:.1f}"
                    )
                _draw_label(
                    cv2,
                    annotated,
                    f"RGB {frame_count} | FPS: {fps:.1f}",
                    (10, 20),
                    (0, 255, 0),
                )
                self.resultReady.emit(stats["result"])
                self.frameReady.emit(_rgb_array_to_qimage(annotated))
        finally:
            if capture is not None:
                capture.release()
            self._running = False
            self.finished.emit()

    def _load_components(self, torch_module):
        repo = self.rgb_repo.resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"RGB repo not found: {repo}")

        if self.detection_input_size <= 0 or self.detection_input_size % 32 != 0:
            raise ValueError("--rgb-detection-input-size must be positive and divisible by 32")

        pose_height, pose_width = _parse_pose_size(self.pose_input_size)
        missing = [
            repo / rel_path
            for rel_path in _required_model_files(self.pose_backbone, self.no_action)
            if not (repo / rel_path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing required model file(s): " + "; ".join(str(path) for path in missing)
            )

        device = self.device
        if device == "auto":
            device = "cuda" if torch_module.cuda.is_available() else "cpu"
        if device == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        previous_cwd = Path.cwd()
        os.chdir(repo)
        try:
            from Detection.Utils import ResizePadding
            from DetectorLoader import TinyYOLOv3_onecls
            from PoseEstimateLoader import SPPE_FastPose
            from Track.Tracker import Detection, Tracker
            from fn import draw_single

            TSSTG = None
            if not self.no_action:
                from ActionsEstLoader import TSSTG as _TSSTG

                TSSTG = _TSSTG

            detect_model = TinyYOLOv3_onecls(self.detection_input_size, device=device)
            pose_model = SPPE_FastPose(
                _pose_backbone_name(self.pose_backbone),
                pose_height,
                pose_width,
                device=device,
            )
            tracker = Tracker(max_age=30, n_init=3)
            action_model = None if self.no_action else TSSTG(device=device)
            resize_fn = ResizePadding(self.detection_input_size, self.detection_input_size)
        finally:
            os.chdir(previous_cwd)

        return {
            "device": device,
            "detect_model": detect_model,
            "pose_model": pose_model,
            "tracker": tracker,
            "action_model": action_model,
            "resize_fn": resize_fn,
            "Detection": Detection,
            "draw_single": draw_single,
        }

    def _process_frame(self, cv2_module, torch_module, components, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if self.mirror:
            bgr = cv2_module.flip(bgr, 1)

        frame = components["resize_fn"](bgr)
        frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        detect_model = components["detect_model"]
        pose_model = components["pose_model"]
        tracker = components["tracker"]
        Detection = components["Detection"]
        draw_single = components["draw_single"]

        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        detected_count = 0 if detected is None else int(detected.shape[0])

        tracker.predict()
        for track in tracker.tracks:
            det_device = detected.device if detected is not None else torch_module.device("cpu")
            det = torch_module.tensor(
                [track.to_tlbr().tolist() + [0.5, 1.0, 0.0]],
                dtype=torch_module.float32,
                device=det_device,
            )
            detected = torch_module.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = _poses_to_detections(poses, frame.shape, Detection)

            if self.show_detected:
                for bb in detected[:, 0:5].detach().cpu().numpy():
                    x1, y1, x2, y2 = bb[:4].astype(int)
                    cv2_module.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        tracker.update(detections)

        action_model = components["action_model"]
        action_state = "off" if self.no_action else "pending"
        structured_tracks = []
        action_count = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update != 0:
                continue

            bbox = track.to_tlbr().astype(int)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, frame.shape[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, frame.shape[0] - 1)
            center = track.get_center().astype(int)
            center[0] = np.clip(center[0], 0, frame.shape[1] - 1)
            center[1] = np.clip(center[1], 0, frame.shape[0] - 1)

            action = "action off" if self.no_action else "pending"
            action_label = None
            action_confidence = None
            action_probs = {}
            action_window_ready = bool(len(track.keypoints_list) == 30)
            label_color = (0, 255, 0)
            if action_model is not None and len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                probs = out[0]
                confidence = float(probs.max())
                action_name = action_model.class_names[probs.argmax()]
                action = f"{action_name}: {confidence * 100.0:.1f}%"
                action_state = action
                action_label = action_name
                action_confidence = confidence
                action_probs = {
                    class_name: float(probs[index])
                    for index, class_name in enumerate(action_model.class_names)
                }
                action_count += 1
                if action_name == "Fall Down":
                    label_color = (255, 0, 0)
                elif action_name == "Lying Down":
                    label_color = (255, 200, 0)

            keypoints = (
                np.array(track.keypoints_list[-1], dtype=np.float32)
                if len(track.keypoints_list) > 0
                else np.empty((0, 3), dtype=np.float32)
            )
            pose_confidence = (
                float(np.nanmean(keypoints[:, 2]))
                if keypoints.ndim == 2 and keypoints.shape[1] > 2 and keypoints.size
                else None
            )
            if self.show_skeleton and len(track.keypoints_list) > 0:
                frame = draw_single(frame, track.keypoints_list[-1])

            cv2_module.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                1,
            )
            _draw_label(cv2_module, frame, f"ID {track.track_id}", (center[0], center[1]), (0, 128, 255))
            _draw_label(cv2_module, frame, action, (bbox[0] + 5, bbox[1] + 15), label_color)
            structured_tracks.append(
                {
                    "rgb_track_id": int(track.track_id),
                    "bbox_x1_px": float(bbox[0]),
                    "bbox_y1_px": float(bbox[1]),
                    "bbox_x2_px": float(bbox[2]),
                    "bbox_y2_px": float(bbox[3]),
                    "bbox_confidence": None,
                    "pose_confidence": pose_confidence,
                    "tracker_state": _track_state_name(track),
                    "track_age": int(getattr(track, "age", 0)),
                    "time_since_update": int(getattr(track, "time_since_update", 0)),
                    "action_window_ready": action_window_ready,
                    "action_label": action_label,
                    "action_confidence": action_confidence,
                    "action_probs": action_probs,
                    "keypoints": _keypoint_records(keypoints, frame_width, frame_height),
                }
            )

        return frame, {
            "detected": detected_count,
            "poses": len(detections),
            "tracks": len(tracker.tracks),
            "action": action_state,
            "result": {
                "type": "rgb_frame",
                "schema_version": 1,
                "rgb_frame_num": None,
                "host_wall_time_iso": None,
                "host_monotonic_ns": None,
                "source": str(self.source),
                "width": frame_width,
                "height": frame_height,
                "fps_estimate": None,
                "frame_read_ok": True,
                "num_detections": detected_count,
                "num_tracks": len(structured_tracks),
                "num_actions": action_count,
                "tracks": structured_tracks,
                "errors": [],
            },
        }

    @Slot()
    def stop(self) -> None:
        self._running = False


class RgbCameraPanel(QWidget):
    resultReady = Signal(dict)

    def __init__(
        self,
        source: object = 0,
        backend: str = "auto",
        width: int = 640,
        height: int = 480,
        fps: float = 30,
        mirror: bool = False,
        posture_enabled: bool = False,
        rgb_repo: object = "",
        device: str = "auto",
        detection_input_size: int = 384,
        pose_input_size: str = "224x160",
        pose_backbone: str = "res50",
        show_skeleton: bool = False,
        show_detected: bool = False,
        no_action: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.source = parse_rgb_source(source)
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.posture_enabled = posture_enabled
        self.rgb_repo = rgb_repo
        self.device = device
        self.detection_input_size = detection_input_size
        self.pose_input_size = pose_input_size
        self.pose_backbone = pose_backbone
        self.show_skeleton = show_skeleton
        self.show_detected = show_detected
        self.no_action = no_action
        self._last_image: Optional[QImage] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None

        self.title_label = QLabel("RGB Posture" if posture_enabled else "RGB Camera")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 14px;")

        self.image_label = QLabel("RGB panel disabled")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #111111; color: #dddddd; border: 1px solid #333333; }"
        )

        self.status_label = QLabel("Starting RGB posture..." if posture_enabled else "Starting RGB camera...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.start()

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = QThread(self)
        if self.posture_enabled:
            self._worker = RgbPostureWorker(
                source=self.source,
                backend=self.backend,
                width=self.width,
                height=self.height,
                fps=self.fps,
                mirror=self.mirror,
                rgb_repo=self.rgb_repo,
                device=self.device,
                detection_input_size=self.detection_input_size,
                pose_input_size=self.pose_input_size,
                pose_backbone=self.pose_backbone,
                show_skeleton=self.show_skeleton,
                show_detected=self.show_detected,
                no_action=self.no_action,
            )
        else:
            self._worker = RgbCaptureWorker(
                source=self.source,
                backend=self.backend,
                width=self.width,
                height=self.height,
                fps=self.fps,
                mirror=self.mirror,
            )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frameReady.connect(self._on_frame)
        self._worker.statusChanged.connect(self._on_status)
        if hasattr(self._worker, "resultReady"):
            self._worker.resultReady.connect(self._on_result)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def stop(self) -> None:
        worker = self._worker
        thread = self._thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(1500)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._render_last_image()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.stop()
        super().closeEvent(event)

    @Slot(QImage)
    def _on_frame(self, image: QImage) -> None:
        self._last_image = image
        self._render_last_image()

    @Slot(str)
    def _on_status(self, message: str) -> None:
        print(f"[rgb-panel] {message}", flush=True)
        self.status_label.setText(message)
        lower_message = message.lower()
        if self._last_image is None and any(
            marker in lower_message
            for marker in ("could not", "unavailable", "error", "failed", "missing")
        ):
            self.image_label.setText(message)

    @Slot(dict)
    def _on_result(self, result: dict) -> None:
        self.resultReady.emit(result)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _render_last_image(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
