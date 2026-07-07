"""Optional OpenGL human posture model renderer for the TI-style visualizer.

The renderer intentionally uses a small OBJ parser instead of adding a GLB
dependency. If loading or drawing fails, callers should keep the normal TI
target boxes visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph.opengl as gl


UNKNOWN_LABELS = {"UNKNOWN", "LOW_QUALITY", "LOW_POINTS", "NO_POINTS", "NO_POSE", "WARMUP"}
HUMAN_UI_DIAG_INTERVAL_FRAMES = 30


@dataclass
class ObjMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    size: np.ndarray
    width: float
    depth: float
    height: float
    horizontal_length: float


def load_obj_mesh(path: str | Path) -> ObjMesh:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                indexes = [_parse_face_index(token, len(vertices)) for token in parts[1:]]
                for offset in range(1, len(indexes) - 1):
                    faces.append([indexes[0], indexes[offset], indexes[offset + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no usable vertices/faces: {path}")

    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)

    mins = vertex_array.min(axis=0)
    maxs = vertex_array.max(axis=0)
    center_x = (mins[0] + maxs[0]) * 0.5
    center_y = (mins[1] + maxs[1]) * 0.5
    vertex_array[:, 0] -= center_x
    vertex_array[:, 1] -= center_y
    vertex_array[:, 2] -= mins[2]

    size = maxs - mins
    width = float(size[0])
    depth = float(size[1])
    height = float(size[2])
    horizontal_length = float(max(width, depth))
    return ObjMesh(
        path.stem,
        vertex_array,
        face_array,
        mins.astype(np.float32),
        maxs.astype(np.float32),
        size.astype(np.float32),
        width,
        depth,
        height,
        horizontal_length,
    )


def _parse_face_index(token: str, vertex_count: int) -> int:
    text = token.split("/")[0]
    value = int(text)
    if value < 0:
        return vertex_count + value
    return value - 1


class HumanPoseModelRenderer:
    def __init__(
        self,
        gl_view,
        model_dir: str | Path,
        scale: float = 1.0,
        height_scale: str | float = "auto",
        target_height: float = 1.70,
        target_sitting_height: float = 1.20,
        target_lying_length: float = 1.70,
        ground_z: float = 0.0,
        opacity: float = 1.0,
        fallback: str = "box",
        debug: bool = False,
        stale_ttl_frames: int = 10,
        ghost_distance_m: float = 0.75,
    ):
        self.gl_view = gl_view
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.scale = float(scale)
        self.height_scale = height_scale
        self.target_height = float(target_height)
        self.target_sitting_height = float(target_sitting_height)
        self.target_lying_length = float(target_lying_length)
        self.ground_z = float(ground_z)
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.fallback = str(fallback)
        self.debug = bool(debug)
        self.stale_ttl_frames = max(0, int(stale_ttl_frames))
        self.ghost_distance_m = max(0.0, float(ghost_distance_m))
        self.meshes: dict[str, ObjMesh] = {}
        self.model_scales: dict[str, float] = {}
        self.items: dict[int, dict[str, Any]] = {}
        self.last_seen_frame: dict[int, int] = {}
        self.last_positions: dict[int, tuple[float, float, float]] = {}
        self.stale_ages: dict[int, int] = {}
        self.disabled = False
        self._warned = False
        self._last_diag_frame: int | None = None
        self._last_frame = 0
        self._last_summary: dict[str, Any] = {
            "frame": 0,
            "active_tracks": 0,
            "active_tids": [],
            "renderer_items": 0,
            "renderer_tids": [],
            "stale_tids": [],
        }
        self._load_meshes()
        self._compute_model_scales()

    def update_models(
        self,
        records: list[dict] | None,
        current_frame: int | None = None,
        active_tids: set[int] | None = None,
    ) -> None:
        if self.disabled:
            return

        try:
            frame = self._resolve_frame(current_frame, records)
            active_tids = self._resolve_active_tids(records, active_tids)
            active_positions: dict[int, tuple[float, float, float]] = {}
            records_by_tid = self._records_by_tid(records)

            for tid in sorted(active_tids):
                record = records_by_tid.get(tid)
                if record is None:
                    self._remove_tid(tid, "missing render record")
                    self.last_seen_frame[tid] = frame
                    self.stale_ages[tid] = 0
                    continue

                tid = int(record.get("tid"))
                model_name = self._model_name_for_label(record.get("final_label"))
                if model_name is None:
                    self._remove_tid(tid, "no model for label")
                    self.last_seen_frame[tid] = frame
                    self.stale_ages[tid] = 0
                    continue

                mesh = self.meshes.get(model_name)
                if mesh is None:
                    self._warn_once(f"human model '{model_name}' was not loaded")
                    self._remove_tid(tid, "mesh missing")
                    self.last_seen_frame[tid] = frame
                    self.stale_ages[tid] = 0
                    continue

                x = float(record.get("x", 0.0))
                y = float(record.get("y", 0.0))
                z = float(record.get("ground_z", self.ground_z))
                if not np.all(np.isfinite([x, y, z])):
                    self._remove_tid(tid, "invalid position")
                    self.last_seen_frame[tid] = frame
                    self.stale_ages[tid] = 0
                    continue

                item = self._item_for_tid(tid, model_name, mesh, record)
                item.resetTransform()
                scale = self._scale_for_record(mesh, record)
                item.scale(scale, scale, scale)
                item.translate(x, y, z)
                item.setVisible(True)
                self.last_seen_frame[tid] = frame
                self.last_positions[tid] = (x, y, z)
                self.stale_ages[tid] = 0
                active_positions[tid] = (x, y, z)

                if self.debug:
                    print(
                        "[HUMAN_UI] tid={} pos=({:.2f},{:.2f},{:.2f}) pose={} last_seen={} age={} visible={} item_id={}".format(
                            tid,
                            x,
                            y,
                            z,
                            record.get("final_label", ""),
                            self.last_seen_frame.get(tid, frame),
                            frame - self.last_seen_frame.get(tid, frame),
                            True,
                            id(item),
                        ),
                        flush=True,
                    )

            stale_tids = self._reconcile_stale_items(frame, active_tids, active_positions)
            self._update_summary(frame, active_tids, stale_tids)
            self._print_summary_if_due(frame)
        except Exception as exc:
            self.disabled = True
            self._warn_once(f"human model rendering disabled after failure: {exc}")
            self.clear_all_human_models()

    def clear(self) -> None:
        self.clear_all_human_models()

    def clear_all_human_models(self) -> None:
        for tid in list(self.items):
            self._remove_tid(tid, "clear")
        self.items.clear()
        self.last_seen_frame.clear()
        self.last_positions.clear()
        self.stale_ages.clear()
        self._last_summary = {
            "frame": self._last_frame,
            "active_tracks": 0,
            "active_tids": [],
            "renderer_items": 0,
            "renderer_tids": [],
            "stale_tids": [],
        }

    def get_debug_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    def _load_meshes(self) -> None:
        file_map = {
            "standing": "human_standing.obj",
            "sitting": "human_sitting.obj",
            "lying": "human_lying.obj",
        }
        for name, filename in file_map.items():
            path = self.model_dir / filename
            if path.exists():
                self.meshes[name] = load_obj_mesh(path)
                continue
            glb_path = path.with_suffix(".glb")
            if glb_path.exists():
                self._warn_once(f"GLB loading is not implemented; missing OBJ: {path}")
            else:
                self._warn_once(f"human model asset missing: {path}")
        if self.debug:
            print(f"[human-model] loaded OBJ models: {sorted(self.meshes)}", flush=True)

    def _compute_model_scales(self) -> None:
        targets = {
            "standing": ("height", self.target_height),
            "sitting": ("height", self.target_sitting_height),
            "lying": ("length", self.target_lying_length),
        }
        for name, mesh in self.meshes.items():
            mode, target = targets.get(name, ("height", self.target_height))
            raw = mesh.horizontal_length if mode == "length" else mesh.height
            if raw > 0.001 and np.isfinite(raw):
                self.model_scales[name] = float(target) / float(raw)
            else:
                self.model_scales[name] = 1.0
            if self.debug:
                raw_bounds = (
                    tuple(float(v) for v in mesh.bounds_min),
                    tuple(float(v) for v in mesh.bounds_max),
                )
                if mode == "length":
                    print(
                        "[human-model-scale] {} raw_bounds={} raw_length={:.3f} scale={:.3f}".format(
                            name, raw_bounds, raw, self.model_scales[name]
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "[human-model-scale] {} raw_bounds={} raw_height={:.3f} scale={:.3f}".format(
                            name, raw_bounds, raw, self.model_scales[name]
                        ),
                        flush=True,
                    )

    def _item_for_tid(self, tid: int, model_name: str, mesh: ObjMesh, record: dict):
        entry = self.items.get(tid)
        if entry is not None and entry.get("model") == model_name:
            item = entry["item"]
            try:
                item.setColor(self._color_for_label(record.get("final_label"), record.get("quality")))
            except Exception:
                pass
            return item

        if entry is not None:
            try:
                self.gl_view.removeItem(entry["item"])
            except Exception:
                entry["item"].setVisible(False)

        mesh_data = gl.MeshData(vertexes=mesh.vertices, faces=mesh.faces)
        item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=True,
            drawFaces=True,
            edgeColor=(0.05, 0.05, 0.05, min(1.0, self.opacity)),
            color=self._color_for_label(record.get("final_label"), record.get("quality")),
        )
        item.setGLOptions("translucent")
        self.gl_view.addItem(item)
        self.items[tid] = {"item": item, "model": model_name}
        return item

    def _resolve_frame(self, current_frame: int | None, records: list[dict] | None) -> int:
        if current_frame is not None:
            try:
                self._last_frame = int(current_frame)
                return self._last_frame
            except Exception:
                pass
        for record in records or []:
            try:
                self._last_frame = int(record.get("frame"))
                return self._last_frame
            except Exception:
                continue
        self._last_frame += 1
        return self._last_frame

    def _resolve_active_tids(
        self, records: list[dict] | None, active_tids: set[int] | None
    ) -> set[int]:
        if active_tids is not None:
            result = set()
            for tid in active_tids:
                try:
                    result.add(int(tid))
                except Exception:
                    continue
            return result
        result = set()
        for record in records or []:
            try:
                result.add(int(record.get("tid")))
            except Exception:
                continue
        return result

    def _records_by_tid(self, records: list[dict] | None) -> dict[int, dict]:
        result: dict[int, dict] = {}
        for record in records or []:
            try:
                result[int(record.get("tid"))] = record
            except Exception:
                continue
        return result

    def _reconcile_stale_items(
        self,
        frame: int,
        active_tids: set[int],
        active_positions: dict[int, tuple[float, float, float]],
    ) -> list[int]:
        stale_tids: list[int] = []
        for tid in list(self.items):
            if tid in active_tids:
                continue
            last_seen = int(self.last_seen_frame.get(tid, frame))
            age = frame - last_seen
            self.stale_ages[tid] = age
            stale_tids.append(tid)
            if self.debug:
                entry = self.items.get(tid, {})
                position = self.last_positions.get(tid, (float("nan"), float("nan"), float("nan")))
                print(
                    "[HUMAN_UI] tid={} pos=({:.2f},{:.2f},{:.2f}) pose={} last_seen={} age={} visible={} item_id={}".format(
                        tid,
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        entry.get("model", ""),
                        last_seen,
                        age,
                        True,
                        id(entry.get("item")),
                    ),
                    flush=True,
                )
            if age >= self.stale_ttl_frames:
                if self.debug:
                    print(
                        f"[HUMAN_UI] removed stale tid={tid} age={age} reason=removed_stale_not_active",
                        flush=True,
                    )
                self._remove_tid(tid, "removed_stale_not_active")
        return stale_tids

    def _update_summary(
        self,
        frame: int,
        active_tids: set[int],
        stale_tids: list[int],
    ) -> None:
        renderer_tids = sorted(int(tid) for tid in self.items)
        self._last_summary = {
            "frame": int(frame),
            "active_tracks": len(active_tids),
            "active_tids": sorted(int(tid) for tid in active_tids),
            "renderer_items": len(renderer_tids),
            "renderer_tids": renderer_tids,
            "stale_tids": sorted(int(tid) for tid in stale_tids),
        }

    def _print_summary_if_due(self, frame: int) -> None:
        if (
            self._last_diag_frame is not None
            and frame - self._last_diag_frame < HUMAN_UI_DIAG_INTERVAL_FRAMES
        ):
            return
        self._last_diag_frame = frame
        summary = self._last_summary
        print(
            "[HUMAN_UI] frame={} active_tracks={} active_tids={} renderer_items={} renderer_tids={} stale_tids={}".format(
                summary["frame"],
                summary["active_tracks"],
                summary["active_tids"],
                summary["renderer_items"],
                summary["renderer_tids"],
                summary["stale_tids"],
            ),
            flush=True,
        )

    def _model_name_for_label(self, label) -> str | None:
        text = str(label).upper()
        if text in {"SITTING"}:
            return "sitting"
        if text in {"LYING", "FALLING"}:
            return "lying"
        if text in UNKNOWN_LABELS:
            if self.fallback == "standing":
                return "standing"
            return None
        return "standing"

    def _scale_for_record(self, mesh: ObjMesh, record: dict) -> float:
        model_name = self._model_name_for_label(record.get("final_label")) or mesh.name
        scale = self.scale * float(self.model_scales.get(model_name, 1.0))
        if str(self.height_scale).lower() != "auto":
            try:
                scale *= float(self.height_scale)
            except Exception:
                pass
        return scale

    def _hide_tid(self, tid: int) -> None:
        entry = self.items.get(int(tid))
        if entry is not None:
            try:
                entry["item"].setVisible(False)
            except Exception:
                pass

    def _remove_tid(self, tid: int, reason: str = "") -> None:
        tid = int(tid)
        entry = self.items.pop(tid, None)
        if entry is None:
            return
        item = entry.get("item")
        try:
            self.gl_view.removeItem(item)
        except Exception as exc:
            try:
                item.setVisible(False)
            except Exception:
                pass
            print(
                f"[human-model-warning] failed to remove TID {tid} ({reason}): {exc}",
                flush=True,
            )
        self.last_positions.pop(tid, None)
        self.last_seen_frame.pop(tid, None)
        self.stale_ages.pop(tid, None)

    def _color_for_label(self, label, quality):
        alpha = self.opacity
        if str(quality).upper() != "OK":
            alpha = min(alpha, 0.55)
        text = str(label).upper()
        if text == "FALLING":
            return (1.0, 0.12, 0.08, alpha)
        if text == "MOVING":
            return (1.0, 0.72, 0.12, alpha)
        if text == "SITTING":
            return (0.18, 0.48, 1.0, alpha)
        if text == "LYING":
            return (0.62, 0.34, 1.0, alpha)
        if text in UNKNOWN_LABELS:
            return (0.55, 0.55, 0.55, min(alpha, 0.35))
        return (0.18, 0.85, 0.35, alpha)

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(f"[human-model] {message}", flush=True)
            self._warned = True
