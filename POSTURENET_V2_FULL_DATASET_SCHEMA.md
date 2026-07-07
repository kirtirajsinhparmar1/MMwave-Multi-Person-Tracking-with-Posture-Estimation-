# PostureNet-v2 Full Point-Cloud Dataset Schema

## 1. Purpose

This schema defines the full RadarPostureNet-v2 Sparse-MoE training dataset. It uses full per-TID associated point-cloud logs instead of only old ONNX probabilities and track summaries.

The required source for point geometry is:

```text
mmwave_associated_points.csv
```

Labels still come from cleaned segment protocols and the session registry. Runtime posture labels and old ONNX predictions are inputs or baselines, not ground truth.

## 2. Required Inputs

Required files:

- `mmwave_associated_points.csv`
- `mmwave_pose.csv`
- `mmwave_tracks.csv`
- cleaned segment labels from `analysis_outputs/posture_cleaning/filled_segments`
- session registry from `analysis_inputs/posture_session_registry_full.csv`

Useful supporting files:

- `analysis_outputs/posture_cleaning/segment_quality.csv`
- `analysis_outputs/posture_cleaning/disappearance_events.csv`
- `pose_predictions_ui.csv` if `mmwave_pose.csv` is not present
- `session_metadata.json` for cfg, RGB availability, sensor setup, and logging metadata

## 3. Source File Roles

### mmwave_associated_points.csv

Primary source for:

- per-frame point rows,
- TID association,
- target-centered coordinates,
- SNR and Doppler,
- target metadata copied onto each point row,
- no-points summary rows,
- old model probabilities when available.

Important columns:

- `session_id`
- `frame`
- `tid`
- `point_x_m`, `point_y_m`, `point_z_m`
- `relative_x_m`, `relative_y_m`, `relative_z_m`
- `height_above_ground_m`
- `point_range_m`
- `point_doppler_mps`
- `point_snr`
- `is_valid_point`
- `geom_pts_for_tid`
- `points_total_frame`
- `quality_label_for_tid`
- `target_x_m`, `target_y_m`, `target_z_m`
- `target_range_m`
- `target_vx_mps`, `target_vy_mps`, `target_vz_mps`
- `old_display_pose`
- `old_model_stand_prob`
- `old_model_sit_prob`
- `old_model_move_prob`
- `old_model_lie_prob`
- `old_model_fall_prob`

### mmwave_pose.csv

Primary source for:

- old runtime posture probabilities,
- old displayed label,
- geometry quality flags,
- point-count summaries,
- display stability,
- motion state.

Fallback file:

```text
pose_predictions_ui.csv
```

### mmwave_tracks.csv

Primary source for:

- target position,
- target range,
- target velocity,
- track existence,
- track persistence.

Fallback file:

```text
targets.csv
```

### Cleaned Segment Labels

Primary source for labels and evaluation grouping:

- `session_id`
- `segment_id`
- `person_slot`
- `expected_pose`
- `expected_subpose`
- `expected_distance_m`
- `expected_position`
- `start_time_s`
- `end_time_s`
- `assigned_tid`
- `label_confidence`
- `assignment_confidence`

### Session Registry

Primary source for:

- session path,
- people count,
- position coverage,
- distance coverage,
- recording date,
- RGB availability,
- trust level,
- notes.

## 4. Window Definition

Each training example is a per-TID temporal window:

```text
session_id + assigned_tid + start_frame + end_frame
```

Recommended window lengths:

- `T=32` frames for fast response,
- `T=48` frames for balanced response,
- `T=64` frames for maximum stability.

Recommended stride:

- 8 to 16 frames for training,
- non-overlapping or session-grouped windows for validation summaries.

Windows must not cross cleaned segment boundaries. Windows with low label confidence can be kept for reliability learning but should not dominate posture loss.

## 5. Output Tensors

### point_tensor

```text
point_tensor:
  [num_windows, T, N, F]
```

Defaults:

- `T = 32, 48, or 64`
- `N = 64`
- `F = 8`

Feature order:

1. `relative_x_m`
2. `relative_y_m`
3. `relative_z_m`
4. `height_above_ground_m`
5. `point_range_m`
6. `point_doppler_mps`
7. `point_snr`
8. `valid_mask`

Point selection when a frame has more than `N` associated points:

- keep high-SNR points,
- keep high-z and low-z points,
- keep spatially spread points,
- avoid using only top-five height points.

Padding:

- padded points are zeros,
- `valid_mask=0`.

No-points frame:

- all point rows padded,
- `valid_mask=0`,
- sparsity features record the dropout.

### track_tensor

```text
track_tensor:
  [num_windows, T, K]
```

Recommended `K` features:

1. `target_range_m`
2. `target_z_m`
3. `target_vx_mps`
4. `target_vy_mps`
5. `target_vz_mps`
6. `speed`
7. `geom_pts`
8. `NO_POINTS_flag`
9. `LOW_POINTS_flag`
10. `OK_flag`
11. `track_age`
12. `pose_switch_count`
13. `ui_visible_flag`
14. `people_count`
15. `position_code`

The preview script currently verifies a compact `K=12` subset because not every source field exists in the smoke log.

### sparsity_tensor

```text
sparsity_tensor:
  [num_windows, S]
```

Recommended `S` features:

1. `range_band_code`
2. `point_count_mean`
3. `point_count_std`
4. `NO_POINTS_rate_window`
5. `LOW_POINTS_rate_window`
6. `SNR_mean`
7. `SNR_std`
8. `valid_frame_rate`
9. `target_range_mean`
10. `target_range_std`
11. `range_jitter`
12. `ui_visible_rate`

Range-band encoding:

- `NEAR = 0`
- `FAR = 1`
- `EDGE = 2`
- `UNKNOWN = -1`

### old_model_tensor

```text
old_model_tensor:
  [num_windows, T, P]
```

Recommended `P=5`:

1. `old_stand_prob`
2. `old_sit_prob`
3. `old_move_prob`
4. `old_lie_prob`
5. `old_fall_prob`

If a probability is missing, fill with `0.0` and add a missing-feature mask if missingness becomes common.

### labels

```text
labels:
  coarse_pose
  subtype_pose
  reliability
  range_band
  sparsity_level
```

`coarse_pose`:

- `STANDING`
- `SITTING`
- `MOVING`
- `UNKNOWN`

`subtype_pose`:

- `STANDING`
- `SITTING_LEAN_BACK`
- `SITTING_UPRIGHT`
- `SITTING_LEAN_FORWARD`
- `UNKNOWN`

`reliability`:

- `HIGH`
- `MEDIUM`
- `LOW`

`range_band`:

- `NEAR`
- `FAR`
- `EDGE`

`sparsity_level`:

- `DENSE`
- `MODERATE`
- `SPARSE`
- `EXTREME_SPARSE`

## 6. Label Assignment

For each cleaned segment row:

1. Resolve `session_id` to the session folder.
2. Use `assigned_tid` for the person instance.
3. Use `start_time_s` and `end_time_s` to choose frames.
4. Keep windows fully inside the segment.
5. Assign `expected_pose` to `coarse_pose`.
6. Assign `expected_subpose` to `subtype_pose`.
7. Assign `range_band` from `expected_distance_m` or measured target range if the expected distance is missing.
8. Assign `sparsity_level` from point-count and valid-frame statistics.
9. Assign `reliability` from evidence quality and label confidence.

Old displayed posture is never used as the target.

## 7. Reliability Label Rule

Initial rule:

```text
HIGH:
  label_confidence HIGH
  assignment_confidence HIGH or MEDIUM
  valid_frame_rate >= 0.9
  mean points >= 8
  NO_POINTS_rate < 0.05
  LOW_POINTS_rate < 0.20

MEDIUM:
  valid_frame_rate >= 0.6
  mean points >= 2
  track remains stable

LOW:
  valid_frame_rate < 0.6
  mean points < 2
  frequent disappearance
  repeated NO_POINTS frames
  high ambiguity or low label/assignment confidence
```

The reliability rule should be audited after collecting 4m, 5m, and 6m associated-point sessions.

## 8. Fallback Policy For Missing Fields

Missing `mmwave_associated_points.csv`:

- do not build a full point-cloud tensor,
- mark the session as full-model unavailable,
- optionally build the older lite dataset only for baseline comparison.

Missing `mmwave_pose.csv`:

- use `pose_predictions_ui.csv`,
- set missing old probabilities to `0.0`,
- keep a missing-probability mask if needed.

Missing `mmwave_tracks.csv`:

- use `targets.csv`,
- if target velocity is missing, set velocity to `0.0` and mark missing velocity.

Missing SNR:

- set `point_snr=0.0`,
- add an SNR-missing flag if many sessions lack SNR.

Missing Doppler:

- set `point_doppler_mps=0.0`,
- do not infer Doppler from target speed.

Missing `track_age`:

- approximate from consecutive frames seen for the same TID inside the window.

Missing `ui_visible`:

- derive from old display label when available,
- otherwise set to `0.0` and mark missing.

Missing `people_count`:

- use registry value,
- otherwise default to `1` and flag missing registry context.

## 9. Current Preview Result

The optional offline preview script is:

```powershell
python analysis\build_sparse_moe_dataset_preview.py `
  --out analysis_outputs\sparse_moe_dataset_preview
```

Current smoke result:

- associated point logs found: 1,
- windows constructed: 200,
- `point_tensor`: `[num_windows, 32, 64, 8]`,
- `track_tensor`: `[num_windows, 32, 12]`,
- `sparsity_tensor`: `[num_windows, 10]`,
- `old_model_tensor`: `[num_windows, 32, 5]`.

This verifies tensor construction only. It is not enough for training because it is a near-range standing smoke log, not a balanced far-range dataset.

## 10. Dataset Manifest

The final builder should write:

- `window_manifest.csv`
- `point_tensor.npy` or chunked equivalent
- `track_tensor.npy`
- `sparsity_tensor.npy`
- `old_model_tensor.npy`
- `labels.csv`
- `dataset_build_report.md`

Required manifest columns:

- `window_id`
- `session_id`
- `segment_id`
- `person_slot`
- `tid`
- `start_frame`
- `end_frame`
- `start_time_s`
- `end_time_s`
- `expected_pose`
- `expected_subpose`
- `range_band`
- `sparsity_level`
- `reliability`
- `expected_distance_m`
- `expected_position`
- `people_count`
- `label_confidence`
- `assignment_confidence`
- `source_point_log`
