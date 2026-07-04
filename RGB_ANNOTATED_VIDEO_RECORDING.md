# RGB Annotated Video Recording

## 1. Files Modified

- `run_ti_style_visualizer.py`
- `rgb_camera_panel.py`

`combined_session_logger.py` was inspected/compiled but not modified.

## 2. New CLI Flags

- `--rgb-record-video`
- `--rgb-video-output <path>`
- `--rgb-video-fps <float>`
- `--rgb-video-codec <fourcc>`
- `--rgb-video-max-queue <int>`

Defaults:

- `--rgb-record-video`: disabled unless passed
- `--rgb-video-output`: empty string
- `--rgb-video-fps`: `0`
- `--rgb-video-codec`: `mp4v`
- `--rgb-video-max-queue`: `120`

## 3. Where Video Is Saved

When `--rgb-record-video` is not passed, no video writer is created.

When `--rgb-record-video` is passed:

- If `--rgb-video-output` is provided, that path is used.
- If combined logging is active and a combined session folder exists, the default path is:

```text
<session_folder>\videos\rgb_annotated.mp4
```

- Otherwise, the default path is:

```text
logs\rgb_annotated_<timestamp>.mp4
```

## 4. Annotated Frames

Recording uses the already-annotated RGB frame after detection/tracking/skeleton/action overlays are drawn and immediately before the same frame is emitted to the RGB panel UI.

The posture worker flow is:

```text
camera frame -> RGB detector/pose/tracker overlays -> enqueue annotated frame for video -> emit same annotated frame to UI
```

No extra inference is run for recording.

## 5. Fail-Safe and Passive Recording

Recording is handled by a dedicated `AsyncRgbVideoWriter` thread with a bounded queue.

- RGB inference/display never waits on disk writes.
- If the queue is full, video frames are dropped and `frames_dropped` is incremented.
- OpenCV writer errors are caught, a warning is emitted, and recording is disabled without crashing the app.
- Writer failure events are best-effort and do not stop the UI.

## 6. Shutdown Behavior

On normal RGB worker shutdown:

- the video writer is stopped
- queued frames are allowed to drain within the writer join timeout
- the OpenCV video writer is released
- a final summary is emitted:

```text
[RGB_VIDEO] saved=<path> frames=<n> dropped=<n>
```

If combined logging is available, these best-effort events are connected to `events.jsonl`:

- `rgb_video_recording_started`
- `rgb_video_recording_stopped`
- `rgb_video_recording_failed`

Event metadata includes path, fps, codec, width, height, frames written, frames dropped, and failure error when applicable.

## 7. Compile Status

Passed:

```powershell
python -m py_compile run_ti_style_visualizer.py rgb_camera_panel.py combined_session_logger.py
```

## 8. Help Status

Passed:

```powershell
python run_ti_style_visualizer.py --help
```

The help output includes:

- `--rgb-record-video`
- `--rgb-video-output RGB_VIDEO_OUTPUT`
- `--rgb-video-fps RGB_VIDEO_FPS`
- `--rgb-video-codec RGB_VIDEO_CODEC`
- `--rgb-video-max-queue RGB_VIDEO_MAX_QUEUE`

## 9. Final Git Status

Expected final status after this change:

```text
 M rgb_camera_panel.py
 M run_ti_style_visualizer.py
?? RGB_ANNOTATED_VIDEO_RECORDING.md
```

Generated `__pycache__` changes from compile validation were restored.

## 10. Exact Command To Run

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\stand_to_sit_gate_video_test `
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

After the test, check:

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs"
dir
```

Then inside the newest session folder:

```powershell
dir .\videos
```

Expected file:

```text
rgb_annotated.mp4
```

Also verify:

- the video opens in Windows Media Player or VLC
- skeleton and boxes are visible
- CSV logs still exist
- app posture behavior did not change

## 11. Limitations

- Live camera/radar recording was not validated here.
- Video codec availability depends on the local OpenCV/Windows codec support.
- If the writer cannot open the selected path/codec, the app continues without recording and emits a warning/event.
- This records annotated RGB panel frames only; there is no raw-video recording flag.
