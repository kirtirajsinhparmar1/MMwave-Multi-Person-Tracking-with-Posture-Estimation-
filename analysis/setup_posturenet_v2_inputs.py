from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from posturenet_v2_common import (
    DEFAULT_CFG,
    REPO_ROOT,
    SEGMENT_FIELDS,
    SESSION_SPECS,
    cfg_family,
    discover_session_path,
    ensure_dir,
    git_value,
    inventory_session,
    protocol_segments,
    search_roots,
    timestamp_utc,
    write_csv,
)


REGISTRY_FIELDS = [
    "session_id",
    "session_path",
    "cfg_path",
    "recording_date",
    "people_count",
    "positions",
    "sequence_description",
    "distances_m",
    "poses_subposes",
    "has_rgb_video",
    "segment_file",
    "trust_level",
    "notes",
]

DISCOVERY_FIELDS = [
    "session_id",
    "exists",
    "selected_path",
    "searched_roots",
    "csv_files",
    "has_rgb_video",
    "has_metadata",
    "cfg_path",
    "recording_date",
    "notes",
]

OLD_CODE_FILES = [
    "ti_style_pose_overlay.py",
    "pose_feature_extractor.py",
    "pose_model_runtime.py",
    "run_ti_style_visualizer.py",
    "human_model_renderer.py",
    "pose_data_logger.py",
    "analysis/replay_posture_decision_fix.py",
    "analysis/analyze_distance_posture_session.py",
]


def copy_file_once(src: Path, dst: Path) -> tuple[str, int]:
    ensure_dir(dst.parent)
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    if dst.exists():
        return ("copied" if src.exists() else "existing_dest_only", dst.stat().st_size)
    return ("missing", 0)


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def old_model_info(model_dir: Path) -> dict:
    metadata = read_json(model_dir / "model_metadata.json")
    classes = metadata.get("class_names") or ["STANDING", "SITTING", "LYING", "FALLING"]
    input_shape = f"[1, {metadata.get('input_size', 176)}]"
    output_shape = f"[1, {metadata.get('num_classes', len(classes))}]"
    onnx_path = model_dir / "ti_pose_model.onnx"
    try:
        import onnx  # type: ignore

        model = onnx.load(str(onnx_path))
        if model.graph.input:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            input_shape = str([d.dim_value or d.dim_param or "?" for d in dims])
        if model.graph.output:
            dims = model.graph.output[0].type.tensor_type.shape.dim
            output_shape = str([d.dim_value or d.dim_param or "?" for d in dims])
    except Exception:
        pass
    return {
        "old_onnx_model_path": str(onnx_path),
        "old_model_input_shape": input_shape,
        "old_model_output_shape": output_shape,
        "old_model_classes": ";".join(str(c) for c in classes),
    }


def snapshot_old_architecture() -> tuple[list[dict], dict]:
    root = ensure_dir(REPO_ROOT / "old_architecture")
    code_snapshot = ensure_dir(root / "code_snapshot")
    artifacts_dir = ensure_dir(root / "model_artifacts")
    reports_dir = ensure_dir(root / "reports")
    manifests_dir = ensure_dir(root / "manifests")
    rows: list[dict] = []

    for rel in OLD_CODE_FILES:
        src = REPO_ROOT / rel
        dst = code_snapshot / rel
        status, size = copy_file_once(src, dst)
        rows.append(
            {
                "kind": "code",
                "status": status,
                "source_path": str(src),
                "destination_path": str(dst),
                "size_bytes": size,
            }
        )

    model_dir = REPO_ROOT / "model_experiments" / "outputs" / "ti_4class_clean_recording_robust_1600_fast"
    for pattern in ["*.onnx", "*.json", "*.csv", "*.txt"]:
        for src in sorted(model_dir.glob(pattern)):
            dst = artifacts_dir / src.name
            status, size = copy_file_once(src, dst)
            rows.append(
                {
                    "kind": "artifact",
                    "status": status,
                    "source_path": str(src),
                    "destination_path": str(dst),
                    "size_bytes": size,
                }
            )

    info = old_model_info(model_dir)
    commit = git_value(["rev-parse", "HEAD"])
    branch = git_value(["branch", "--show-current"])
    timestamp = timestamp_utc()

    manifest = manifests_dir / "OLD_ARCHITECTURE_MANIFEST.md"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("# Old Posture Architecture Manifest\n\n")
        handle.write(f"- timestamp_utc: {timestamp}\n")
        handle.write(f"- git_commit: {commit}\n")
        handle.write(f"- current_branch: {branch}\n")
        handle.write(f"- old_onnx_model_path: {info['old_onnx_model_path']}\n")
        handle.write(f"- old_model_input_shape: {info['old_model_input_shape']}\n")
        handle.write(f"- old_model_output_shape: {info['old_model_output_shape']}\n")
        handle.write(f"- old_model_classes: {info['old_model_classes']}\n\n")
        handle.write("| kind | status | size_bytes | source_path | destination_path |\n")
        handle.write("| --- | --- | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['kind']} | {row['status']} | {row['size_bytes']} | "
                f"`{row['source_path']}` | `{row['destination_path']}` |\n"
            )

    summary = reports_dir / "OLD_POSTURE_ARCHITECTURE_SUMMARY.md"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("# Old Posture Architecture Summary\n\n")
        handle.write(
            "The old runtime builds a TI-style 176-feature window per TID from target metadata and the selected highest-z associated points. "
            "The feature vector is 22 channels over 8 frames: target z/velocity/acceleration plus five selected point y/z/SNR triplets. "
            "The ONNX model in `ti_4class_clean_recording_robust_1600_fast` predicts STANDING, SITTING, LYING, and FALLING.\n\n"
        )
        handle.write(
            "Runtime posture is not the raw model output. `ti_style_pose_overlay.py` applies per-TID smoothing, confidence thresholds, "
            "motion overrides, standing/sitting stability counts, height-drop/fall gates, sitting gates, range-zone logic, and a relative "
            "sitting gate before deciding the displayed posture and human model asset.\n\n"
        )
        handle.write(
            "Known limitations: the old model was trained on prior TI-style feature windows, not the current user-collected standing/sitting "
            "session registry; current logs often record only model probabilities and track metadata, not raw associated point tensors; "
            "absolute/range-sensitive behavior caused standing/sitting confusion at specific distances; and UI rendering/dropout evidence "
            "is mixed into runtime behavior rather than modeled as a separate reliability output.\n\n"
        )
        handle.write(
            "A new architecture is needed to use user-provided protocols as ground truth, validate by held-out sessions/positions/person counts, "
            "separate posture from visibility/reliability, and avoid absolute-coordinate shortcuts as the main posture signal.\n"
        )

    return rows, info


def generate_registry_and_segments() -> tuple[list[dict], list[dict]]:
    registry_rows: list[dict] = []
    discovery_rows: list[dict] = []
    segments_root = ensure_dir(REPO_ROOT / "analysis_inputs" / "posture_segments")
    searched = ";".join(str(root) for root in search_roots())

    for spec in SESSION_SPECS:
        path = discover_session_path(spec.session_id)
        inventory = inventory_session(path)
        cfg_path = str(inventory.get("cfg_path") or spec.cfg_hint or DEFAULT_CFG)
        recording_date = str(inventory.get("recording_date") or spec.recording_date)
        segment_file = segments_root / f"{spec.session_id}_segments.csv"
        segment_rows = protocol_segments(spec.session_id)
        write_csv(segment_file, segment_rows, SEGMENT_FIELDS)

        notes = spec.notes
        inv_notes = str(inventory.get("notes") or "")
        if inv_notes:
            notes = f"{notes} Discovery: {inv_notes}"
        registry_rows.append(
            {
                "session_id": spec.session_id,
                "session_path": str(path or ""),
                "cfg_path": cfg_path,
                "recording_date": recording_date,
                "people_count": spec.people_count,
                "positions": spec.positions,
                "sequence_description": spec.sequence_description,
                "distances_m": spec.distances_m,
                "poses_subposes": spec.poses_subposes,
                "has_rgb_video": bool(inventory.get("has_rgb_video")),
                "segment_file": str(segment_file.relative_to(REPO_ROOT)),
                "trust_level": spec.trust_level,
                "notes": notes,
            }
        )
        discovery_rows.append(
            {
                "session_id": spec.session_id,
                "exists": bool(inventory.get("exists")),
                "selected_path": str(path or ""),
                "searched_roots": searched,
                "csv_files": inventory.get("csv_files", ""),
                "has_rgb_video": bool(inventory.get("has_rgb_video")),
                "has_metadata": bool(inventory.get("has_metadata")),
                "cfg_path": cfg_path,
                "recording_date": recording_date,
                "notes": inv_notes,
            }
        )

    write_csv(REPO_ROOT / "analysis_inputs" / "posture_session_registry_full.csv", registry_rows, REGISTRY_FIELDS)
    out_dir = ensure_dir(REPO_ROOT / "analysis_outputs" / "posture_registry_full")
    write_csv(out_dir / "session_discovery.csv", discovery_rows, DISCOVERY_FIELDS)

    with (out_dir / "SESSION_DISCOVERY_REPORT.md").open("w", encoding="utf-8") as handle:
        found = sum(1 for row in discovery_rows if row["exists"])
        handle.write("# Session Discovery Report\n\n")
        handle.write(f"Sessions requested: {len(discovery_rows)}\n\n")
        handle.write(f"Sessions found: {found}\n\n")
        handle.write(f"Search roots: `{searched}`\n\n")
        handle.write("| session_id | exists | selected_path | csv_files | notes |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for row in discovery_rows:
            handle.write(
                f"| {row['session_id']} | {row['exists']} | `{row['selected_path']}` | "
                f"{row['csv_files']} | {row['notes']} |\n"
            )
    return registry_rows, discovery_rows


def write_modality_audit(discovery_rows: list[dict]) -> None:
    has_raw_points = False
    has_point_xyz = False
    has_track_indexes = False
    has_associated_points = False
    has_pose_probs = False
    has_features176 = False
    evidence: list[str] = []

    for row in discovery_rows:
        csv_files = str(row.get("csv_files") or "")
        csv_names = [name.strip() for name in csv_files.split(";") if name.strip()]
        lower = csv_files.lower()
        if "mmwave_pose.csv" in lower or "pose_predictions_ui.csv" in lower:
            has_pose_probs = True
        if "features_176" in lower:
            has_features176 = True
        point_like = [
            name
            for name in csv_names
            if not name.lower().startswith("rgb_")
            and (
                name.lower()
                in {"points.csv", "raw_points.csv", "mmwave_points.csv", "point_cloud.csv", "pointcloud.csv"}
                or "point" in name.lower()
                or name.lower() in {"features_176.csv", "features_22.csv"}
            )
        ]
        if point_like:
            evidence.append(f"{row['session_id']}: mmWave point-candidate csv names: {';'.join(point_like)}")
        session_path = Path(str(row.get("selected_path") or ""))
        meta = read_json(session_path / "session_metadata.json") if session_path.exists() else {}
        if meta.get("mmwave_log_points") is False:
            evidence.append(f"{row['session_id']}: session_metadata.json has mmwave_log_points=false")
        if (session_path / "points.csv").exists() or (session_path / "raw_points.csv").exists():
            has_raw_points = True
        for candidate in ["points.csv", "raw_points.csv", "mmwave_points.csv", "point_cloud.csv"]:
            path = session_path / candidate
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8-sig") as handle:
                        header = next(csv.reader(handle), [])
                    lower_header = {h.lower() for h in header}
                    has_point_xyz = {"x", "y", "z"}.issubset(lower_header) or {"x_m", "y_m", "z_m"}.issubset(lower_header)
                    has_track_indexes = bool({"track_index", "target_index", "tid"} & lower_header)
                    has_associated_points = has_point_xyz and has_track_indexes
                except Exception:
                    pass

    report = REPO_ROOT / "POSTURE_DATA_MODALITY_AUDIT.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Posture Data Modality Audit\n\n")
        qa = [
            ("1. Do the logs contain raw pointCloud rows per frame?", "yes" if has_raw_points else "no"),
            ("2. Do the logs contain per-point x/y/z/snr/doppler?", "yes" if has_point_xyz else "no"),
            ("3. Do the logs contain trackIndexes or point-to-TID association?", "yes" if has_track_indexes else "no"),
            ("4. Do the logs contain associated points per TID?", "yes" if has_associated_points else "no"),
            ("5. Do the logs contain only mmwave_pose probabilities and track metadata?", "yes" if has_pose_probs and not has_associated_points else "no"),
            ("6. Do the logs contain old 176-feature vectors?", "yes" if has_features176 else "no"),
            ("7. Can target-centered point-cloud tensors be reconstructed?", "yes" if has_associated_points else "no"),
            ("8. Is full RadarPostureNet-v2 trainable from current logs?", "yes" if has_associated_points else "no"),
            ("9. Is only RadarPostureNet-v2-lite trainable from current logs?", "yes" if has_pose_probs and not has_associated_points else "no"),
            (
                "10. What exact logging must be added for full point-cloud training?",
                "Log per-frame per-point x/y/z/doppler/SNR plus target index/TID association, frame number, timestamp, target pose/track rows, and point quality for every TID.",
            ),
        ]
        for question, answer in qa:
            handle.write(f"## {question}\n\n{answer}\n\n")
        handle.write("## Evidence\n\n")
        if evidence:
            for item in evidence:
                handle.write(f"- {item}\n")
        else:
            handle.write("- No point-cloud evidence found in discovered session files.\n")
        handle.write("\n## Decision\n\n")
        if has_associated_points:
            handle.write("Per-point associated point data exists, so the full dataset path should be built.\n")
        else:
            handle.write("Per-point associated point data is not available. Build the lite dataset and treat full point-cloud training as blocked until logging is added.\n")


def write_architecture_spec() -> None:
    path = REPO_ROOT / "RADAR_POSTURENET_V2_ARCHITECTURE.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 Architecture\n\n")
        sections = [
            ("1. Problem definition", "Classify per-TID posture as STANDING or SITTING, optionally with sitting subtypes, while predicting reliability/visibility from mmWave/RGB-aligned logs."),
            ("2. Why tracking is not the problem", "Tracking already produces TIDs and target kinematics; the failure mode is posture interpretation under range, side position, two-person association, and disappearance uncertainty."),
            ("3. Why absolute coordinates should not be the main posture signal", "Absolute x/y/z encode room placement and distance labels, so they can shortcut validation and fail when the person moves to a new range or side."),
            ("4. Coordinate-invariant target-centered normalization", "Use per-TID target-centered points, relative heights, spreads, velocities, and temporal deltas; retain range as context, not as the main label cue."),
            ("5. Handling distance 1m-5m", "Report metrics by distance and hold out 5m separately. Normalize geometry relative to the tracked target and include quality/range as covariates."),
            ("6. Handling center/left/right position", "Train and validate center-to-side generalization. Use lateral sign/order for assignment only when data separation verifies it."),
            ("7. Handling two-person sessions", "Keep left/right person instances as separate labels with shared segment times. Validate single-person to two-person transfer."),
            ("8. Handling disappearance/render uncertainty", "Model missing pose rows, low/no points, TID switches, UI visibility, and render confirmation as reliability evidence instead of silently discarding them."),
            ("9. Coarse labels vs fine sitting subtype labels", "The coarse head predicts STANDING/SITTING. The subtype head predicts STANDING, SITTING_LEAN_BACK, SITTING_UPRIGHT, and SITTING_LEAN_FORWARD where labels exist."),
            ("10. How old data and new data are combined", "Old ONNX outputs and runtime logs are auxiliary signals. User protocol segments are the only ground-truth posture labels."),
            ("11. How old ONNX model is used as auxiliary teacher/input", "Old model probabilities, margins, and stability rates can be input features or distillation targets. They are never labels."),
            ("12. Full architecture if point-cloud data exists", "RadarPostureNet-v2-full: per-TID associated point sequence -> target-centered point normalization -> PointNet/Point Transformer point encoder -> track + quality MLP encoder -> old ONNX probability embedding -> fusion -> TCN + Temporal Transformer -> coarse posture head -> sitting subtype head -> reliability/visibility head."),
            ("13. Lite architecture if only logs/probabilities exist", "RadarPostureNet-v2-lite: temporal windows of old ONNX probabilities, track metadata, range, velocity, geom_pts, NO_POINTS/LOW_POINTS/OK, display stability, render visibility, and TID stability -> TCN/Transformer or strong tabular temporal model -> coarse posture head -> subtype head if labels exist -> reliability/visibility head."),
            ("14. Training strategy", "Choose full only if point tensors exist; otherwise train lite. Run baselines plus at most three bounded model families with early stopping and no unbounded search."),
            ("15. Validation strategy", "Use leave-one-session-out, old/single-person to newer sessions, center to left/right, single-person to two-person, 5m reporting, and standing_3m false-sitting checks."),
            ("16. UI/UX integration plan", "Only accepted models get runtime flags. Shadow mode logs v2 predictions while old posture controls UI. Replace mode is allowed only after acceptance."),
            ("17. Runtime rollback plan", "Default remains old behavior. Disable `--pose-v2-enable` or switch from replace to shadow to roll back."),
            ("18. Acceptance criteria", "Standing accuracy >=95%, false SITTING on standing_3m <=5%, sitting and sitting subtypes improve over old runtime, left/right gap is not severe, two-person accuracy does not collapse, and 5m is reported separately."),
        ]
        for title, text in sections:
            handle.write(f"## {title}\n\n{text}\n\n")
        handle.write("## Current-pass decision\n\nThe full architecture is preferred, but the current discovered combined logs do not contain per-point associated point rows. This pass therefore trains the lite fallback unless later discovery finds point tensors.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-old-snapshot", action="store_true")
    args = parser.parse_args()

    snapshot_rows = []
    old_info = {}
    if not args.skip_old_snapshot:
        snapshot_rows, old_info = snapshot_old_architecture()
    registry_rows, discovery_rows = generate_registry_and_segments()
    write_modality_audit(discovery_rows)
    write_architecture_spec()

    print(f"Old files considered: {len(snapshot_rows)}")
    print(f"Old model classes: {old_info.get('old_model_classes', 'not inspected')}")
    print(f"Sessions found: {sum(1 for row in discovery_rows if row['exists'])}/{len(discovery_rows)}")
    print(f"Registry rows: {len(registry_rows)}")
    print("Registry: analysis_inputs/posture_session_registry_full.csv")
    print("Segment templates: analysis_inputs/posture_segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
