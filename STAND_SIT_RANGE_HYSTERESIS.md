# Stand/Sit Range Hysteresis

## Files Modified

- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `STAND_SIT_RANGE_HYSTERESIS.md`

No RGB repo files, RGB posture pipeline files, radar startup code, human model rendering code, point association logic, ONNX model files, or fall detection redesign were changed.

## What Changed

- Added a standing-vs-sitting resolver that compares smoothed `STANDING` and `SITTING` probabilities before generic display calibration.
- Added range-aware standing/sitting margins using target range from `sqrt(x^2 + y^2)`.
- Added range-aware hysteresis frames for `STANDING -> SITTING`, `SITTING -> STANDING`, and `MOVING/UNKNOWN -> STANDING/SITTING`.
- Added sustained MOVING override logic so `speed_high` does not immediately replace a stable standing/sitting display.
- Extended pose debug/CSV logging with resolver, range, hysteresis, and MOVING override fields.

## Generic Calibration Handling

The previous generic `STANDING` minimum confidence is bypassed for decisions produced by the standing/sitting resolver. A resolved stand/sit candidate uses probability margin plus hysteresis instead of the absolute standing threshold.

The existing generic pose-specific confidence logic remains for non-stand/sit decisions.

## New CLI Flags And Defaults

- `--pose-range-near-max 2.0`
- `--pose-range-mid-max 4.0`
- `--pose-stand-sit-near-margin 0.06`
- `--pose-stand-sit-mid-margin 0.10`
- `--pose-stand-sit-far-margin 0.15`
- `--pose-stand-to-sit-near-frames 6`
- `--pose-stand-to-sit-mid-frames 8`
- `--pose-stand-to-sit-far-frames 12`
- `--pose-sit-to-stand-near-frames 8`
- `--pose-sit-to-stand-mid-frames 10`
- `--pose-sit-to-stand-far-frames 14`
- `--pose-moving-override-near-frames 3`
- `--pose-moving-override-mid-frames 4`
- `--pose-moving-override-far-frames 5`

Existing flags remain backward-compatible, including `--pose-display-stability-frames`, `--pose-display-stability-ratio`, `--pose-sitting-stability-frames`, `--pose-sitting-min-confidence`, and the generic per-pose confidence/stability flags.

## Range Zones

- `near`: `range_m <= 2.0`
- `mid`: `2.0 < range_m <= 4.0`
- `far`: `range_m > 4.0`
- `unknown`: used only if range cannot be computed, with mid defaults used internally.

## Stand/Sit Margin Logic

When the smoothed pose is `STANDING` or `SITTING`:

- If `standing_prob - sitting_prob >= margin_for_range`, candidate becomes `STANDING`.
- If `sitting_prob - standing_prob >= margin_for_range`, candidate becomes `SITTING`.
- If the margin is ambiguous and previous display is `STANDING` or `SITTING`, hold the previous display pose.
- If the margin is ambiguous and previous display is `MOVING` or `UNKNOWN`, keep the current display until enough margin appears.

Reason strings include `stand_sit_margin_standing`, `stand_sit_margin_sitting`, and `stand_sit_hold_previous_ambiguous`.

## Hysteresis Logic

Range-aware frames replace generic pose-specific frames for stand/sit transitions:

- `STANDING -> SITTING`: near 6, mid 8, far 12
- `SITTING -> STANDING`: near 8, mid 10, far 14
- `MOVING/UNKNOWN -> STANDING/SITTING`: near 6, mid 8, far 12

Reason strings include `stand_sit_waiting_hysteresis` and `stand_sit_hysteresis_update`.

## Sustained MOVING Override

When current display or resolver candidate is `STANDING` or `SITTING`, MOVING requires consecutive motion evidence before overriding:

- near: 3 frames
- mid: 4 frames
- far: 5 frames

Before that count is met, the display holds the previous stand/sit pose or the resolver result with reason `moving_override_waiting`. Once met, MOVING can update with reason `moving_override_sustained`.

## Example Debug Output

```text
[pose] tid=2 raw=STANDING 0.58 smooth=STANDING 0.57 cand=STANDING display=SITTING candidate_conf=0.57 candidate_stable=5/10 pose_min_conf=0.00 pose_required_frames=10 stand_prob=0.57 sit_prob=0.48 stand_sit_margin=0.09 stand_sit_zone=mid stand_sit_decision=HOLD stand_sit_required_frames=10 stand_sit_stable=5/10 moving_override_stable=1/4 range_m=2.65 range_zone=mid reason=stand_sit_waiting_hysteresis quality=NO_POINTS
```

## Validation

Compile status:

```powershell
python -m py_compile run_ti_style_visualizer.py ti_style_pose_overlay.py pose_model_runtime.py pose_feature_extractor.py
```

Result: PASS.

Help status:

```powershell
python run_ti_style_visualizer.py --help
```

Result: PASS. The new range-aware stand/sit flags appear in help.

Live radar validation: not run. No COM6/COM7 visual validation was performed.

## Final Git Status

Full `rtk git status --short` output at completion:

```text
 M __pycache__/human_model_renderer.cpython-311.pyc
 M __pycache__/pose_feature_extractor.cpython-311.pyc
 M __pycache__/pose_model_runtime.cpython-311.pyc
 M __pycache__/run_ti_style_visualizer.cpython-311.pyc
 M __pycache__/ti_style_pose_overlay.cpython-311.pyc
 M logs/ti_pose_ui_human_models_plane_stable16/pose_predictions_ui.csv
 M logs/ti_pose_ui_human_models_plane_stable16/pose_ui_metadata.json
 M run_ti_style_visualizer.py
 M ti_style_pose_overlay.py
 M ti_style_vendor/Industrial_Visualizer/cache/cachedData.txt
 M ti_style_vendor/PySide2/__pycache__/QtCore.cpython-311.pyc
 M ti_style_vendor/PySide2/__pycache__/QtGui.cpython-311.pyc
 M ti_style_vendor/PySide2/__pycache__/QtOpenGL.cpython-311.pyc
 M ti_style_vendor/PySide2/__pycache__/QtSerialPort.cpython-311.pyc
 M ti_style_vendor/PySide2/__pycache__/QtWidgets.cpython-311.pyc
 M ti_style_vendor/PySide2/__pycache__/__init__.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/adc_plot.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/false_alarm_test.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/fft_plot.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/plot_1d.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/plot_2d.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/plot_3d.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/power_consumption_report.cpython-311.pyc
 M ti_style_vendor/common/Common_Tabs/__pycache__/range_snr_plot.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/Helper_Classes/__pycache__/classification.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/Helper_Classes/__pycache__/fall_detection.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/ChildPresenceDetection.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/LifePresenceDetection.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/SeatBeltReminder.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/calibration.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/dashcam.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/debug_plots.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/ebikes_x432.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/gesture_recognition.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/intruder_detection.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/kick_to_open.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/level_sensing.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/long_range_pd.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/mobile_tracker.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/out_of_box_x432.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/out_of_box_x843.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/out_of_box_x844.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/people_tracking.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/people_tracking_6844.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/point_cloud_classification.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/range_sensing.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/sleep_monitoring.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/small_obstacle.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/smart_toilet.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/surface_classification.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/true_ground_speed.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/video_doorbell.cpython-311.pyc
 M ti_style_vendor/common/Demo_Classes/__pycache__/vital_signs.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/Baud_Rates_Manager.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/cached_data.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/demo_defines.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/gl_text.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/graph_utilities.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/gui_common.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/gui_core.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/gui_parser.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/gui_threads.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/json_fix.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/parseFrame.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/parseTLVs.cpython-311.pyc
 M ti_style_vendor/common/__pycache__/tlv_defines.cpython-311.pyc
 M ti_style_vendor/common/gui_parser.py
?? DEBUG_COMBINED_LOG_FAILURE.md
?? POSE_STABILITY_CONFIDENCE_TUNING.md
?? RUN_COMBINED_MMWAVE_RGB.md
?? STAND_SIT_RANGE_HYSTERESIS.md
?? __pycache__/combined_session_logger.cpython-311.pyc
?? __pycache__/rgb_camera_panel.cpython-311.pyc
?? combined_session_logger.py
?? logs/combined_mmw_rgb_human_models_logging_test/
?? logs/combined_mmw_rgb_human_models_stable16/
?? logs/combined_mmw_rgb_human_models_stable16_noaction/
?? logs/final_combined_mmw_rgb_human_models/
?? logs/pose_calibration_test_stability_confidence/
?? logs/stable_visual_mmw_rgb_pose/
?? logs/test_mmw_human_models_raw_rgb/
?? logs/test_mmw_human_models_rgb_posture_no_combined/
?? logs/test_mmw_human_models_rgb_posture_no_combined_detected/
?? rgb_camera_panel.py
?? ti_style_vendor/Industrial_Visualizer/binData/
```

Scoped files for this task were `run_ti_style_visualizer.py`, `ti_style_pose_overlay.py`, and `STAND_SIT_RANGE_HYSTERESIS.md`. The other entries were not part of this scoped change.

## Command To Run Next

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\stand_sit_range_hysteresis_test `
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
  --pose-stand-sit-near-margin 0.06 `
  --pose-stand-sit-mid-margin 0.10 `
  --pose-stand-sit-far-margin 0.15 `
  --pose-stand-to-sit-near-frames 6 `
  --pose-stand-to-sit-mid-frames 8 `
  --pose-stand-to-sit-far-frames 12 `
  --pose-sit-to-stand-near-frames 8 `
  --pose-sit-to-stand-mid-frames 10 `
  --pose-sit-to-stand-far-frames 14 `
  --pose-moving-override-near-frames 3 `
  --pose-moving-override-mid-frames 4 `
  --pose-moving-override-far-frames 5 `
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
