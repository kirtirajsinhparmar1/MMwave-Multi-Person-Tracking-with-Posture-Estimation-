# Geometry, Association, and Upright Sitting Fix

## Root problem found

Upright sitting was still being treated too much like standing because the mmWave display path did not have enough reliable body-height evidence. The logs also showed repeated `pts=0` / `quality=NO_POINTS` cases while the TI parser exposes point-to-track association through `trackIndexes`, so the overlay was often missing usable point geometry. A second issue was that sustained `speed_high` could still let `MOVING` override a strong standing/sitting decision.

## Files modified

- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `GEOMETRY_ASSOC_UPRIGHT_SITTING_FIX.md`

## MOVING override fix

Strong standing/sitting decisions now block speed-only `MOVING` overrides. The strong decision margin is range aware:

- near: `0.12`
- mid: `0.18`
- far: `0.25`

When `--pose-moving-require-translation` is enabled, `MOVING` can override strong `STANDING`/`SITTING` only when translation is confirmed across the configured history window. New reasons include:

- `moving_override_blocked_by_strong_stand_sit`
- `moving_override_translation_confirmed`
- `moving_override_speed_only_rejected`

## New geometry features

The pose overlay now computes per-target geometry from associated points when available:

- `geom_pts`
- `geom_centroid_z`
- `geom_top_z`
- `geom_bottom_z`
- `geom_height`
- `geom_floor_centroid_z`
- `geom_quality`
- point range min/max and centroid x/y/z

It also logs target fallback geometry:

- `target_x`
- `target_y`
- `target_z`
- `target_range_m`
- `target_speed`

## Sensor calibration flags

Calibration is opt-in and debug-safe. Raw coordinates remain available, and calibrated/floor-relative values are logged separately.

- `--pose-sensor-height-m` default `1.25`
- `--pose-sensor-pitch-deg` default `0.0`
- `--pose-sensor-roll-deg` default `0.0`
- `--pose-sensor-yaw-deg` default `0.0`
- `--pose-use-sensor-calibration`
- `--pose-floor-z-m` default `0.0`

## Point association diagnostics and fallback

Association now uses layered methods:

1. `trackIndexes` / target index association when present.
2. Existing point target-id/index fields when usable.
3. Nearest-neighbor fallback around the target position.
4. Target-only fallback when no usable points exist.

New CLI flags:

- `--pose-assoc-debug`
- `--pose-assoc-method auto|target_index|nearest|hybrid` default `auto`
- `--pose-assoc-nearest-radius-m` default `0.75`
- `--pose-assoc-nearest-z-min` default `-0.5`
- `--pose-assoc-nearest-z-max` default `2.5`
- `--pose-assoc-min-points-good` default `3`

Diagnostics include:

- `[POINT_ASSOC] frame=... points_total=... tracks_total=... has_target_index=...`
- `[POINT_ASSOC] tid=... points_by_target_index=... points_by_nearest=... final_assoc=... final_points=...`

## Standing baseline logic

When enabled, each TID maintains its own standing baseline only after stable `STANDING` frames:

- `standing_baseline_centroid_z`
- `standing_baseline_top_z`
- `standing_baseline_height_extent`
- `standing_baseline_target_z`
- `standing_baseline_range_m`
- `standing_baseline_frames`

Baseline updates only when the displayed pose and stand/sit resolver agree on `STANDING`, the target is mostly stationary, and the target is not being displayed as `MOVING`.

## Upright sitting logic

When baseline is ready, ambiguous or close standing/sitting cases can become `SITTING` from geometry drop evidence:

- centroid drop
- top-height drop
- height extent reduction
- target-z fallback drop when point geometry is unavailable

Range-aware sitting drop thresholds:

- near: `0.20 m`
- mid: `0.25 m`
- far: `0.35 m`

Other defaults:

- `--pose-standing-baseline-min-frames` default `20`
- `--pose-sitting-drop-min-sit-prob` default `0.30`
- `--pose-sitting-drop-target-z-m` default `0.20`

New reasons include:

- `geometry_sitting_drop`
- `geometry_sitting_drop_target_only`
- `upright_sitting_geometry_supported`
- `geometry_no_sitting_drop`

## New CLI flags and defaults

- `--pose-strong-stand-sit-near-margin 0.12`
- `--pose-strong-stand-sit-mid-margin 0.18`
- `--pose-strong-stand-sit-far-margin 0.25`
- `--pose-moving-require-translation` enabled by default
- `--no-pose-moving-require-translation`
- `--pose-moving-translation-window 8`
- `--pose-moving-translation-min-m 0.25`
- `--pose-sensor-height-m 1.25`
- `--pose-sensor-pitch-deg 0.0`
- `--pose-sensor-roll-deg 0.0`
- `--pose-sensor-yaw-deg 0.0`
- `--pose-use-sensor-calibration`
- `--pose-floor-z-m 0.0`
- `--pose-assoc-debug`
- `--pose-assoc-method auto`
- `--pose-assoc-nearest-radius-m 0.75`
- `--pose-assoc-nearest-z-min -0.5`
- `--pose-assoc-nearest-z-max 2.5`
- `--pose-assoc-min-points-good 3`
- `--pose-use-standing-baseline`
- `--pose-standing-baseline-min-frames 20`
- `--pose-sitting-drop-near-m 0.20`
- `--pose-sitting-drop-mid-m 0.25`
- `--pose-sitting-drop-far-m 0.35`
- `--pose-sitting-drop-min-sit-prob 0.30`
- `--pose-sitting-drop-centroid-m 0.25`
- `--pose-sitting-drop-top-m 0.25`
- `--pose-sitting-drop-target-z-m 0.20`

## Debug fields added

The pose debug line and CSV include additional fields for:

- association: `assoc_method`, `points_total`, `points_by_target_index`, `points_by_nearest`
- geometry: `geom_pts`, `geom_quality`, `geom_centroid_z`, `geom_top_z`, `geom_bottom_z`, `geom_height`
- calibration: `raw_target_z`, `cal_target_z`, `floor_z`, `floor_relative_z`
- baseline: `baseline_ready`, `baseline_frames`, `baseline_top_z`, `baseline_centroid_z`
- drop evidence: `height_drop`, `centroid_drop`, `target_z_drop`, `geometry_range_threshold`
- moving override: `moving_override_reason`, `translation_m`, `translation_confirmed`
- final decision: `geometry_decision`, `geometry_reason`, `final_reason`

Example:

```text
[pose] tid=2 raw=STANDING 0.58 smooth=STANDING 0.57 cand=SITTING display=SITTING stand_prob=0.41 sit_prob=0.48 stand_sit_margin=-0.07 stand_sit_decision=SITTING range_m=2.31 range_zone=mid geom_pts=6 geom_quality=POINT_GEOMETRY geom_top_z=1.19 geom_centroid_z=0.82 baseline_ready=true baseline_frames=24 baseline_top_z=1.48 baseline_centroid_z=1.11 height_drop=0.29 centroid_drop=0.29 geometry_decision=SITTING geometry_reason=upright_sitting_geometry_supported moving_override_reason=moving_override_speed_only_rejected final_reason=stand_sit_hysteresis_update
```

## Compile status

Passed:

```powershell
python -m py_compile run_ti_style_visualizer.py ti_style_pose_overlay.py pose_model_runtime.py pose_feature_extractor.py combined_session_logger.py
```

## Help status

Passed:

```powershell
python run_ti_style_visualizer.py --help
```

The new CLI flags appeared in help output.

## Final git status

Scoped files changed by this pass:

- `run_ti_style_visualizer.py`
- `ti_style_pose_overlay.py`
- `GEOMETRY_ASSOC_UPRIGHT_SITTING_FIX.md`

The worktree also contains pre-existing generated/cache/log and RGB/combined-session related changes that were not part of this pass.

## Exact live command to run next

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\geometry_assoc_upright_sitting_test `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-log `
  --pose-debug `
  --pose-3d-labels `
  --pose-min-associated-points-for-inference 1 `
  --pose-allow-target-only `
  --pose-human-models `
  --pose-human-model-mode overlay_box `
  --pose-human-model-dir "ui_human_pose_models" `
  --pose-ground-plane `
  --pose-ground-z 0.0 `
  --pose-assoc-debug `
  --pose-assoc-method hybrid `
  --pose-assoc-nearest-radius-m 0.75 `
  --pose-assoc-min-points-good 3 `
  --pose-moving-require-translation `
  --pose-moving-translation-window 8 `
  --pose-moving-translation-min-m 0.25 `
  --pose-strong-stand-sit-near-margin 0.12 `
  --pose-strong-stand-sit-mid-margin 0.18 `
  --pose-strong-stand-sit-far-margin 0.25 `
  --pose-use-standing-baseline `
  --pose-standing-baseline-min-frames 20 `
  --pose-sitting-drop-near-m 0.20 `
  --pose-sitting-drop-mid-m 0.25 `
  --pose-sitting-drop-far-m 0.35 `
  --pose-sitting-drop-min-sit-prob 0.30 `
  --pose-use-sensor-calibration `
  --pose-sensor-height-m 1.25 `
  --pose-sensor-pitch-deg 0.0 `
  --pose-sensor-roll-deg 0.0 `
  --pose-sensor-yaw-deg 0.0 `
  --pose-floor-z-m 0.0 `
  --pose-range-near-max 2.0 `
  --pose-range-mid-max 4.0 `
  --enable-rgb-panel `
  --enable-rgb-posture `
  --rgb-repo "C:\Users\UBESC\Desktop\Combined MMwave and RGB\RGB Posture Estmation\Human-Falling-Detect-Tracks" `
  --rgb-source 0 `
  --rgb-camera-backend auto `
  --rgb-device cpu `
  --rgb-no-action `
  --rgb-show-skeleton `
  --rgb-show-detected `
  --combined-log `
  --combined-status-panel `
  --log-root "..\logs" `
  --rgb-log-keypoints
```

## Physical test protocol

1. Place radar in current vertical setup.
2. Stand still at about 2 m for 20 seconds so baseline can form.
3. Sit upright, not leaning back, for 20 seconds.
4. Stand again for 20 seconds.
5. Repeat at about 3 m.
6. Repeat at about 4 m.
7. Repeat once with radar tilted slightly downward if possible.

## Console fields to inspect

Check for:

- `baseline_ready=true`
- `baseline_frames>=20`
- `geom_pts>0`
- `geom_quality=POINT_GEOMETRY`
- `geometry_decision=SITTING`
- `geometry_reason=geometry_sitting_drop` or `upright_sitting_geometry_supported`
- `moving_override_reason=moving_override_speed_only_rejected` when stand/sit is strong
- `translation_confirmed=true` only when the person is actually walking/translating

## Limitations and next step

This pass does not retrain the model and does not add RGB fusion. If live testing still shows `geom_pts=0`, the next step is to inspect the `[POINT_ASSOC]` diagnostics against the raw TI point-cloud/track-index fields for that specific cfg and firmware output.
