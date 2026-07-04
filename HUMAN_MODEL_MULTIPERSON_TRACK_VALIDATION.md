# Human Model Multi-Person Track Validation

## 1. Why single-person mode is not the deployment fix

Single-person rendering hides the duplicate/shadow symptom by choosing one track, but it breaks the actual system goal: multi-person tracking. The deploy-ready fix keeps all real people renderable and suppresses only weak, provisional, or suspect radar tracks from becoming full posture human models.

The human-model UI now renders every confirmed track. A one-person scene with a radar ghost renders one confirmed person. A two-person scene with two confirmed radar tracks renders two people.

## 2. Track states

- `STATIONARY`: not a validation state. A stationary active radar track can be valid and must stay visible if confirmed.
- `STALE`: a renderer item whose TID is no longer present in the current active radar TID set.
- `PROVISIONAL`: a newly active radar TID that has not yet collected enough good geometry evidence.
- `CONFIRMED`: an active radar TID with enough recent evidence to render as a full posture human model.
- `SUSPECT_GHOST`: an active radar TID with repeated weak evidence, usually no associated points or target-only geometry.
- `LOST`: a validation state for a TID that disappeared from the active radar TID set and is waiting for stale TTL cleanup.

## 3. Files modified

- `human_model_renderer.py`
- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `ti_style_vendor/common/Demo_Classes/people_tracking.py`
- `HUMAN_MODEL_MULTIPERSON_TRACK_VALIDATION.md`

## 4. Correct stale cleanup logic

Renderer stale status is now based only on current radar activity:

```python
stale = tid not in active_tids_this_frame
```

Stationary position, low speed, and low movement are not stale signals.

If `active_tids_this_frame` is empty, every cached renderer item becomes stale. If a renderer item stays stale until `age >= --pose-human-model-stale-frames`, the renderer removes it from the GL view, hides it safely if removal fails, deletes the TID from the renderer cache, and deletes stale bookkeeping.

Debug removal reason:

```text
[HUMAN_UI] tid=<tid> removed_stale_not_active age=<age>
```

## 5. Multi-person validation state machine

New active TIDs enter as `PROVISIONAL`.

```text
NEW -> PROVISIONAL
PROVISIONAL -> CONFIRMED       when enough recent good evidence exists
PROVISIONAL -> SUSPECT_GHOST   when repeated bad/no-point evidence exists
CONFIRMED -> SUSPECT_GHOST     only after long bad evidence
ACTIVE -> LOST                 when TID is absent from current active radar tracks
LOST -> removed                after stale TTL
```

The validation state is UI-side only. It does not modify radar tracks, parser output, ONNX inference, posture smoothing, raw combined logs, raw radar logs, or raw pose CSV values.

## 6. Confirmation rules

Defaults:

```text
--pose-human-model-confirm-frames 5
--pose-human-model-confirm-min-geom-pts 3
--pose-human-model-confirm-min-quality-frames 3
```

Good evidence requires:

- `geom_pts >= confirm_min_geom_pts`
- association is not `auto_none`
- `quality` is not `NO_POINTS` or `TARGET_ONLY`
- `geom_quality` is not `TARGET_ONLY`
- position/range values are finite
- body geometry is not impossibly small
- the TID has persisted long enough

Good evidence is counted over a recent window. It does not need to be good on every single frame.

## 7. Ghost and suspect rules

Defaults:

```text
--pose-human-model-ghost-min-bad-frames 8
--pose-human-model-ghost-no-points-frames 8
--pose-human-model-confirmed-grace-frames 30
--pose-human-model-bad-evidence-demote-frames 60
```

Bad evidence includes:

- `geom_pts == 0`
- `quality == NO_POINTS`
- `quality == TARGET_ONLY`
- `geom_quality == TARGET_ONLY`
- `assoc == auto_none`
- target-index association with zero geometry points
- invalid position
- invalid body geometry

Confirmed tracks are stable. Short point-association gaps do not hide a confirmed person. A confirmed active track is demoted only after the long bad-evidence threshold.

Stillness is not used as ghost evidence.

## 8. Posture display quality gate

Only `CONFIRMED` tracks render as full posture human models with strong posture labels such as `STANDING`, `SITTING`, `LYING`, or `FALLING`.

By default:

- `PROVISIONAL` tracks do not render as full human models.
- `SUSPECT_GHOST` tracks do not render as full human models.
- non-confirmed 3D text/table display uses `PROVISIONAL` or `SUSPECT_GHOST` instead of a confident posture label.

Debug flags can show non-confirmed tracks without strong posture labels:

```text
--pose-human-model-show-provisional
--pose-human-model-show-suspect
```

## 9. Debug output examples

Summary:

```text
[HUMAN_UI] frame=151 active_tracks=2 confirmed=1 provisional=0 suspect=1 rendered=1 stale=0
```

Per TID:

```text
[HUMAN_UI] tid=1 state=SUSPECT_GHOST geom_pts=0 quality=NO_POINTS geom_quality=TARGET_ONLY assoc=auto_none good_frames=0 bad_frames=8 no_points_frames=8 stale_age=0 rendered=false reason=no_points_bad_evidence
```

Transitions:

```text
[HUMAN_UI] tid=0 transition=NEW_TO_PROVISIONAL reason=new_tid
[HUMAN_UI] tid=0 transition=PROVISIONAL_TO_CONFIRMED reason=good_evidence
[HUMAN_UI] tid=1 transition=PROVISIONAL_TO_SUSPECT_GHOST reason=no_points_bad_evidence
[HUMAN_UI] tid=1 transition=CONFIRMED_TO_SUSPECT_GHOST reason=long_bad_evidence
[HUMAN_UI] tid=1 removed_stale_not_active age=5
```

Status panel text:

```text
Human UI: active=<n>, confirmed=<n>, provisional=<n>, suspect=<n>, rendered=<n>, stale=<n>
```

## 10. Validation status

Passed:

```powershell
python -m py_compile human_model_renderer.py ti_style_pose_overlay.py run_ti_style_visualizer.py
python -m py_compile ti_style_vendor\common\Demo_Classes\people_tracking.py
python run_ti_style_visualizer.py --help
```

The help output includes:

```text
--pose-human-model-confirm-frames
--pose-human-model-confirm-min-geom-pts
--pose-human-model-confirm-min-quality-frames
--pose-human-model-confirmed-grace-frames
--pose-human-model-bad-evidence-demote-frames
--pose-human-model-ghost-min-bad-frames
--pose-human-model-ghost-no-points-frames
--pose-human-model-show-provisional
--pose-human-model-show-suspect
```

Live COM6/COM7 radar validation was not run.

## 11. Final git status

`git status --short` after implementation and validation:

```text
 M __pycache__/human_model_renderer.cpython-311.pyc
 M __pycache__/rgb_camera_panel.cpython-311.pyc
 M __pycache__/run_ti_style_visualizer.cpython-311.pyc
 M __pycache__/ti_style_pose_overlay.cpython-311.pyc
 M human_model_renderer.py
 D logs/combined_mmw_rgb_human_models_logging_test/pose_predictions_ui.csv
 D logs/combined_mmw_rgb_human_models_logging_test/pose_ui_metadata.json
 D logs/combined_mmw_rgb_human_models_stable16/pose_predictions_ui.csv
 D logs/combined_mmw_rgb_human_models_stable16/pose_ui_metadata.json
 D logs/combined_mmw_rgb_human_models_stable16_noaction/pose_predictions_ui.csv
 D logs/combined_mmw_rgb_human_models_stable16_noaction/pose_ui_metadata.json
 D logs/fall_test1/fall_events.csv
 D logs/fall_test1/frames_summary.csv
 D logs/fall_test1/heights.csv
 D logs/fall_test1/points.csv
 D logs/fall_test1/targets.csv
 D logs/final_combined_mmw_rgb_human_models/pose_predictions_ui.csv
 D logs/final_combined_mmw_rgb_human_models/pose_ui_metadata.json
 D logs/mount170_flat_geometry_test/pose_predictions_ui.csv
 D logs/mount170_flat_geometry_test/pose_ui_metadata.json
 D logs/original_cfg_legacy_like_pose_test/pose_predictions_ui.csv
 D logs/original_cfg_legacy_like_pose_test/pose_ui_metadata.json
 D logs/original_cfg_sitting_less_sensitive_test/pose_predictions_ui.csv
 D logs/original_cfg_sitting_less_sensitive_test/pose_ui_metadata.json
 D logs/original_cfg_stable_test/pose_predictions_ui.csv
 D logs/original_cfg_stable_test/pose_ui_metadata.json
 D logs/pose_calibration_test_stability_confidence/pose_predictions_ui.csv
 D logs/pose_calibration_test_stability_confidence/pose_ui_metadata.json
 D logs/rollback_original_cfg_no_geometry/pose_predictions_ui.csv
 D logs/rollback_original_cfg_no_geometry/pose_ui_metadata.json
 D logs/stable_visual_mmw_rgb_pose/pose_predictions_ui.csv
 D logs/stable_visual_mmw_rgb_pose/pose_ui_metadata.json
 D logs/stand_sit_range_hysteresis_test/pose_predictions_ui.csv
 D logs/stand_sit_range_hysteresis_test/pose_ui_metadata.json
 D logs/stand_to_sit_gate_test/pose_predictions_ui.csv
 D logs/stand_to_sit_gate_test/pose_ui_metadata.json
 D logs/test_mmw_human_models_raw_rgb/pose_predictions_ui.csv
 D logs/test_mmw_human_models_raw_rgb/pose_ui_metadata.json
 D logs/test_mmw_human_models_rgb_posture_no_combined/pose_predictions_ui.csv
 D logs/test_mmw_human_models_rgb_posture_no_combined/pose_ui_metadata.json
 D logs/test_mmw_human_models_rgb_posture_no_combined_detected/pose_predictions_ui.csv
 D logs/test_mmw_human_models_rgb_posture_no_combined_detected/pose_ui_metadata.json
 D logs/ti_pose_ui_4class/pose_predictions_ui.csv
 D logs/ti_pose_ui_4class/pose_ui_metadata.json
 D logs/ti_pose_ui_4class_confidence_labels/pose_predictions_ui.csv
 D logs/ti_pose_ui_4class_confidence_labels/pose_ui_metadata.json
 D logs/ti_pose_ui_4class_labels/pose_predictions_ui.csv
 D logs/ti_pose_ui_4class_labels/pose_ui_metadata.json
 D logs/ti_pose_ui_human_models/pose_predictions_ui.csv
 D logs/ti_pose_ui_human_models/pose_ui_metadata.json
 D logs/ti_pose_ui_human_models_plane_stable16/pose_predictions_ui.csv
 D logs/ti_pose_ui_human_models_plane_stable16/pose_ui_metadata.json
 D logs/ti_pose_ui_human_models_plane_stable40/pose_predictions_ui.csv
 D logs/ti_pose_ui_human_models_plane_stable40/pose_ui_metadata.json
 D logs/ti_pose_ui_test1/pose_predictions_ui.csv
 D logs/ti_pose_ui_test1/pose_ui_metadata.json
 D logs/ti_pose_ui_warmup_debug/pose_predictions_ui.csv
 D logs/ti_pose_ui_warmup_debug/pose_ui_metadata.json
 D logs/ui_test1/fall_events.csv
 D logs/ui_test1/frames_summary.csv
 D logs/ui_test1/heights.csv
 D logs/ui_test1/points.csv
 D logs/ui_test1/targets.csv
 M rgb_camera_panel.py
 M run_ti_style_visualizer.py
 M ti_style_pose_overlay.py
 M ti_style_vendor/common/Common_Tabs/__pycache__/plot_3d.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/plot_3d.py
 M ti_style_vendor/common/Demo_Classes/__pycache__/people_tracking.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/people_tracking.py
?? HUMAN_MODEL_MULTIPERSON_TRACK_VALIDATION.md
?? HUMAN_MODEL_UI_STALE_GHOST_FIX.md
?? RGB_ANNOTATED_VIDEO_RECORDING.md
?? logs/human_ui_stale_cleanup_test/
?? logs/stand_to_sit_gate_video_test/
?? ti_style_vendor/Industrial_Visualizer/binData/human_ui_stale_cleanup_test/
?? ti_style_vendor/Industrial_Visualizer/binData/stand_to_sit_gate_video_test/
```

Intentional source files for this task are `human_model_renderer.py`, `ti_style_pose_overlay.py`, `run_ti_style_visualizer.py`, `ti_style_vendor/common/Demo_Classes/people_tracking.py`, and this report. The worktree also contains unrelated pre-existing log/cache/RGB changes.

## 12. Exact multi-person-safe command

Do not include `--pose-human-model-single-person-mode`.

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\multiperson_track_validation_test `
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
  --pose-human-model-debug `
  --pose-human-model-stale-frames 5 `
  --pose-human-model-ghost-distance-m 1.25 `
  --pose-human-model-confirm-frames 5 `
  --pose-human-model-confirm-min-geom-pts 3 `
  --pose-human-model-confirm-min-quality-frames 3 `
  --pose-human-model-confirmed-grace-frames 30 `
  --pose-human-model-bad-evidence-demote-frames 60 `
  --pose-human-model-ghost-min-bad-frames 8 `
  --pose-human-model-ghost-no-points-frames 8 `
  --pose-ground-plane `
  --pose-ground-z 0.0 `
  --pose-display-stability-frames 16 `
  --pose-display-stability-ratio 0.80 `
  --pose-sitting-stability-frames 16 `
  --pose-sitting-min-confidence 0.60 `
  --pose-stand-to-sit-min-confidence 0.65 `
  --pose-stand-to-sit-margin 0.15 `
  --pose-stand-to-sit-frames 12 `
  --pose-sit-to-stand-recovery-margin 0.10 `
  --pose-sit-to-stand-recovery-frames 6 `
  --enable-rgb-panel `
  --enable-rgb-posture `
  --rgb-repo "C:\Users\UBESC\Desktop\Combined MMwave and RGB\RGB Posture Estmation\Human-Falling-Detect-Tracks" `
  --rgb-source 0 `
  --rgb-camera-backend auto `
  --rgb-device cpu `
  --rgb-no-action `
  --rgb-show-skeleton `
  --rgb-show-detected `
  --rgb-record-video `
  --rgb-video-fps 20 `
  --rgb-video-codec mp4v `
  --combined-log `
  --combined-status-panel `
  --log-root "..\logs" `
  --rgb-log-keypoints
```

## 13. Limitations and next steps

- This does not remove radar ghosts from raw data.
- This does not tune the TI tracker or cfg.
- This prevents low-evidence ghosts from appearing as full posture humans in the UI.
- Real second people may appear with a short confirmation delay.
- Field validation with COM6/COM7 should verify the expected cases: one person plus ghost, two real people, new person entering, all tracks disappearing, and a stationary confirmed person.
