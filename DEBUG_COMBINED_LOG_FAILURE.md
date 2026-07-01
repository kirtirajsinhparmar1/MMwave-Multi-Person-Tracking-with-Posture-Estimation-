# Combined Logging/Status Failure Debug

## 1. Root cause

The combined path was doing synchronous CSV logging and status rendering inside high-frequency UI callbacks. The biggest issue was `CombinedSessionLogger._flush_writer()`: `csv.DictWriter` does not expose the file handle through `writer.f`, so every row write flushed every open log file. With RGB keypoint logging enabled, this made the RGB result path especially expensive and could stall the UI.

The RGB combined hook also wrote frames/tracks/actions/keypoints directly from the `resultReady` slot. That meant keypoint CSV work could compete with frame display. Status rendering also ran on every combined update.

## 2. Files modified

- `run_ti_style_visualizer.py`
- `combined_session_logger.py`
- `DEBUG_COMBINED_LOG_FAILURE.md`

No RGB repository files were modified.

## 3. What was fixed

- Logger creation is wrapped in `try/except`; failure prints `[COMBINED][ERROR] ...`, disables combined logging, and allows the visual app to continue.
- Added concise `[COMBINED]` diagnostics for enabled flags, log root/session id, logger creation, status panel creation, and hook attachment.
- mmWave logging hook renders first, logs second, catches logging exceptions, and returns the original `updateGraph()` result.
- mmWave point logging no longer loops over points unless `--mmwave-log-points` is enabled.
- RGB logging now uses a bounded `deque` plus a `QTimer` drain, so the `resultReady` slot only enqueues and returns.
- RGB logging drains at most two results per timer tick and records dropped queued results as events if logging falls behind.
- Status panel creation is fail-safe.
- Status panel rendering is throttled to about 4 Hz and catches formatting/widget update failures.
- CSV flushing is periodic instead of every row/every file.
- `--combined-session` still only expands to RGB panel, RGB posture, combined logging, and combined status panel.

## 4. Known-good visual command

Not visually retested from Codex because it requires the live COM6/COM7 radar, camera, and GUI confirmation. The known-good command path was preserved: no changes were made to radar startup, pose inference, RGB posture inference, or human model rendering logic.

## 5. Combined-log only

Not visually retested from Codex. Expected to work with less UI blocking because logging is fail-safe and CSV flushing is throttled.

## 6. Status panel

Not visually retested from Codex. Creation and updates are now wrapped and throttled; failure should no longer stop the UI.

## 7. Keypoint logging

Not visually retested from Codex. Keypoint logging is still potentially high volume, but it now runs from a timer-drained bounded queue instead of directly in the RGB result signal path.

## 8. Compile status

Passed:

```powershell
rtk python -m py_compile run_ti_style_visualizer.py rgb_camera_panel.py combined_session_logger.py human_model_renderer.py ti_style_pose_overlay.py
```

## 9. Help status

Passed:

```powershell
rtk python run_ti_style_visualizer.py --help
```

## 10. Final git status

The worktree is dirty. Relevant modified/untracked files include:

- `run_ti_style_visualizer.py`
- `combined_session_logger.py`
- `DEBUG_COMBINED_LOG_FAILURE.md`
- existing generated `__pycache__` and `logs/` entries
- existing untracked `rgb_camera_panel.py` and `RUN_COMBINED_MMWAVE_RGB.md`

## 11. Exact command to run next

Run combined-log only first:

```powershell
python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\combined_log_only_test `
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
  --pose-display-stability-ratio 0.70 `
  --pose-sitting-stability-frames 8 `
  --pose-sitting-min-confidence 0.40 `
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
  --log-root "..\logs"
```

If that works, add `--combined-status-panel`. If that works, add `--rgb-log-keypoints`.

## 12. Warnings or limitations

- Visual B/C/D validation still needs to be done with the physical radar/camera.
- `--rgb-log-keypoints` can still generate many rows; the queue prevents direct signal-path blocking, but if disk cannot keep up, old RGB log results can be dropped and an event will be written.
- Log verification should be run after a successful live combined-log session under `C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs`.
