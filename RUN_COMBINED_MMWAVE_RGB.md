# Run Combined mmWave + RGB

## 1. Recommended full live command

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-human-models `
  --pose-ground-plane `
  --combined-session `
  --log-root "..\logs" `
  --rgb-repo "C:\Users\UBESC\Desktop\Combined MMwave and RGB\RGB Posture Estmation\Human-Falling-Detect-Tracks" `
  --rgb-source 0 `
  --rgb-camera-backend auto `
  --rgb-device auto `
  --rgb-show-skeleton `
  --rgb-log-keypoints
```

## 2. Safer CPU/no-action debug command

```powershell
python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-human-models `
  --pose-ground-plane `
  --combined-session `
  --log-root "..\logs" `
  --rgb-repo "C:\Users\UBESC\Desktop\Combined MMwave and RGB\RGB Posture Estmation\Human-Falling-Detect-Tracks" `
  --rgb-source 0 `
  --rgb-camera-backend auto `
  --rgb-device cpu `
  --rgb-no-action `
  --rgb-show-skeleton `
  --rgb-log-keypoints
```

## 3. Camera backend troubleshooting

- Start with `--rgb-camera-backend auto`.
- If the camera does not open, try `--rgb-camera-backend msmf`.
- If MSMF fails or is unstable, try `--rgb-camera-backend dshow`.
- If source `0` is unavailable, try `--rgb-source 1`.
- Close other camera applications before launching the combined UI.

## 4. Expected logs

With `--combined-session`, logs are written under the selected `--log-root` in a session folder named like `session_YYYYMMDD_HHMMSS`.

Expected files:

- `session_metadata.json`
- `events.jsonl`
- `rgb_frames.csv`
- `rgb_tracks.csv`
- `rgb_actions.csv`
- `rgb_keypoints.csv` when `--rgb-log-keypoints` is used
- `mmwave_frames.csv`
- `mmwave_tracks.csv`
- `mmwave_pose.csv`
- `mmwave_points.csv` when `--mmwave-log-points` is used
- `sync_index.csv`

In `--demo` mode, mmWave CSVs may be header-only because the TI-style demo path may not emit live UART frames. Live hardware runs should populate mmWave rows when frames arrive.
