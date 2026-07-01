# Pose Stability and Confidence Tuning

## Files modified

- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `POSE_STABILITY_CONFIDENCE_TUNING.md`

## New CLI flags added

- `--pose-standing-min-confidence`
- `--pose-sitting-min-confidence`
- `--pose-lying-min-confidence`
- `--pose-falling-min-confidence`
- `--pose-moving-min-confidence`
- `--pose-standing-stability-frames`
- `--pose-sitting-stability-frames`
- `--pose-lying-stability-frames`
- `--pose-falling-stability-frames`
- `--pose-moving-stability-frames`
- `--pose-unknown-stability-frames`

## Default confidence thresholds

- `STANDING`: `0.70`
- `SITTING`: `0.45`
- `LYING`: `0.60`
- `FALLING`: `0.70`
- `MOVING`: `0.35`
- `UNKNOWN`: `0.00`

## Default stability windows

- `STANDING`: `12` frames
- `SITTING`: `8` frames
- `LYING`: `14` frames
- `FALLING`: `4` frames
- `MOVING`: `4` frames
- `UNKNOWN`: `6` frames

## Compatibility notes

- `--pose-sitting-stability-frames` still controls the internal SITTING stability window.
- `--pose-sitting-min-confidence` still controls the internal SITTING confidence gate.
- `--pose-fall-stability-frames` and `--pose-falling-stability-frames` remain aliases for the same FALLING stability setting.
- `--pose-display-stability-frames` remains available as the fallback stability window for labels without a pose-specific value.
- `--pose-display-min-confidence` remains available as the fallback confidence gate for labels without a pose-specific value.
- `--pose-display-stability-ratio` remains accepted and logged for compatibility, but display updates now require the pose-specific consecutive candidate frame count.

## Example debug output

```text
[pose] tid=2 raw=STANDING 0.91 smooth=STANDING 0.88 cand=STANDING display=STANDING candidate_conf=0.88 candidate_stable=12/12 pose_min_conf=0.70 pose_required_frames=12 reason=pose_specific_stable_update
```

## Validation status

- Compile: passed
- Help: passed; new CLI flags appear in `python run_ti_style_visualizer.py --help`
- Live radar validation: not run

## Final git status

At report time, the relevant modified files are:

```text
 M run_ti_style_visualizer.py
 M ti_style_pose_overlay.py
?? POSE_STABILITY_CONFIDENCE_TUNING.md
```

The broader working tree already contained unrelated modified caches, logs, RGB/combined files, and vendor files before this change.

## Command to run next

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\pose_calibration_test_stability_confidence `
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
  --pose-standing-min-confidence 0.70 `
  --pose-sitting-min-confidence 0.45 `
  --pose-lying-min-confidence 0.60 `
  --pose-falling-min-confidence 0.70 `
  --pose-moving-min-confidence 0.35 `
  --pose-standing-stability-frames 12 `
  --pose-sitting-stability-frames 8 `
  --pose-lying-stability-frames 14 `
  --pose-falling-stability-frames 4 `
  --pose-moving-stability-frames 4 `
  --pose-unknown-stability-frames 6 `
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
