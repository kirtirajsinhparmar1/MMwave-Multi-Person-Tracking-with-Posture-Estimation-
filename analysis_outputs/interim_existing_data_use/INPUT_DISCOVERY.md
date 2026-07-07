# Input Discovery

Generated: 2026-07-06

This task reuses the existing bounded PostureNet-v2 lite pass outputs. No UI runtime, tracking, cfg, ONNX model, or RGB repository code is modified.

| input | status | evidence |
|---|---|---|
| Full session registry | found | `analysis_inputs\posture_session_registry_full.csv` exists and lists all 9 user-provided sessions. |
| Cleaned segments | found | `analysis_outputs\posture_cleaning\filled_segments` exists with one filled segment CSV per registered session. |
| Lite dataset | found | `analysis_outputs\posturenet_v2_dataset\posturenet_lite_windows.csv` exists and contains protocol labels, old probabilities, old display rates, range, quality, visibility, distance, position, and people-count fields. |
| Prior model reports | found | `analysis_outputs\posturenet_v2_model\model_comparison.csv` and `POSTURENET_V2_END_TO_END_REPORT.md` exist. The prior lite replacement model did not pass acceptance. |
| RGB files | found | The 9 registered session folders contain RGB sidecars such as `rgb_keypoints.csv`, `rgb_tracks.csv`, `rgb_frames.csv`, `sync_index.csv`, and `videos\rgb_annotated.mp4`. |
| Full point-cloud tensors | missing | `analysis_outputs\posturenet_v2_dataset\pointcloud_availability_report.csv` reports missing per-point xyz/signal rows with point-to-target association for all 9 sessions. |

## Sessions Reused

- `session_20260703_205540`
- `sitting_ab_default_cfg`
- `sitting_ab_static_retention_cfg`
- `sitting_relative_gate_refined_live_test`
- `session_20260704_145249`
- `session_20260704_150636`
- `session_20260704_152302`
- `session_20260706_173741`
- `session_20260706_175519`

## Decision

Existing data is usable for failure analysis, RGB teacher audit, and an offline conservative reliability gate. It is not sufficient for a replacement full RadarPostureNet-v2 posture model because associated per-TID point-cloud tensors are absent.
