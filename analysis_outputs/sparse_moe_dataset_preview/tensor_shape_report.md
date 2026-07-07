# Sparse-MoE Dataset Preview

This report verifies tensor construction only. No model was trained.

## Inputs

- Associated point logs found: 1
- Window frames: 32
- Stride frames: 16
- Max points per TID/frame: 64

## Tensor Shapes

- `point_tensor`: `[num_windows, 32, 64, 8]`
- `track_tensor`: `[num_windows, 32, 12]`
- `sparsity_tensor`: `[num_windows, 10]`
- `old_model_tensor`: `[num_windows, 32, 5]`

## Feature Lists

- Point features: `relative_x_m`, `relative_y_m`, `relative_z_m`, `height_above_ground_m`, `point_range_m`, `point_doppler_mps`, `point_snr`, `valid_mask`
- Track features: `target_range_m`, `target_z_m`, `target_vx_mps`, `target_vy_mps`, `target_vz_mps`, `speed_mps`, `geom_pts_for_tid`, `NO_POINTS_flag`, `LOW_POINTS_flag`, `OK_flag`, `track_age_norm`, `ui_visible_flag`
- Sparsity features: `range_band_code`, `point_count_mean`, `point_count_std`, `NO_POINTS_rate_window`, `LOW_POINTS_rate_window`, `SNR_mean`, `SNR_std`, `valid_frame_rate`, `target_range_mean`, `target_range_std`
- Old model features: `old_stand_prob`, `old_sit_prob`, `old_move_prob`, `old_lie_prob`, `old_fall_prob`

## Result

- Windows constructed: 200
- Sessions represented: 1
- TIDs represented: 2
- Mean point_count_mean: 2.980
- Mean valid_frame_rate: 0.468

## Label Note

The preview does not require cleaned labels. If a session name exposes a posture, the manifest marks that as a heuristic label source; otherwise labels remain `UNKNOWN`.
