# Sitting A/B Test Commands

Purpose: run a controlled sitting-only A/B test where the only intended experiment variable is the TI cfg file.

Protocol for both tests:

- sit at 2m for 60 sec
- sit at 3m for 60 sec
- sit at 4m for 60 sec

Do not add posture threshold experiments to these commands. Keep both runs identical except for `--cfg`, `--out`, and `--session-id`.

## Test A: default cfg

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\sitting_ab_default_cfg `
  --session-id sitting_ab_default_cfg `
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
  --rgb-source 1 `
  --rgb-camera-backend dshow `
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

## Test B: static-retention cfg

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg" `
  --out logs\sitting_ab_static_retention_cfg `
  --session-id sitting_ab_static_retention_cfg `
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
  --rgb-source 1 `
  --rgb-camera-backend dshow `
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

## After recording

Fill the manual segment CSV for each run after checking the RGB video and range plot:

- `analysis_inputs\sitting_ab_default_segments.csv`
- `analysis_inputs\sitting_ab_static_retention_segments.csv`

Then analyze each session with the filled segment file before running the comparison helper.
