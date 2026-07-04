# RGB External Camera Selection

## Problem

The RGB panel already supported OpenCV sources through `--rgb-source`, but there was no camera discovery or convenience path for choosing the externally attached USB camera instead of the default built-in camera. The selected camera also needed to be visible in terminal/UI status and combined-session metadata/events.

## Files Modified

- `run_ti_style_visualizer.py`
- `rgb_camera_panel.py`
- `RGB_EXTERNAL_CAMERA_SELECTION.md`

`combined_session_logger.py` was inspected. Its CSV schemas were left unchanged; selected-camera metadata is written through session metadata and events.

## Existing `--rgb-source` Behavior

Existing numeric source behavior is preserved:

```powershell
--rgb-source 0
--rgb-source 1
--rgb-source 2
```

Numeric values are still treated as OpenCV camera indices. Non-numeric values are still passed through as paths/URLs.

## New Camera Listing Flag

Added:

```powershell
--rgb-list-cameras
--rgb-camera-probe-max-index 10
```

List-only command:

```powershell
python run_ti_style_visualizer.py --rgb-list-cameras --rgb-camera-backend auto --rgb-camera-probe-max-index 10
```

Observed result on this machine:

```text
[RGB_CAMERA] index=0 opened=True width=640 height=480 fps=30.0 backend=auto name=UNKNOWN
[RGB_CAMERA] index=1 opened=True width=640 height=480 fps=30.0 backend=auto name=UNKNOWN
[RGB_CAMERA] recommended external candidates: [1]
```

OpenCV also printed camera-index-out-of-range diagnostics for unavailable indices; that comes from probing closed indices.

## New External Camera Selection Flag

Added:

```powershell
--rgb-prefer-external
```

Priority:

1. With `--rgb-prefer-external`, probe indices and choose the first opened index greater than `0`.
2. If none is found, fall back to `--rgb-source`.
3. Without `--rgb-prefer-external`, use `--rgb-source` exactly as provided.

## Selected Camera Metadata And Status

Startup now prints:

```text
[RGB_CAMERA] selected_source=<source>
[RGB_CAMERA] backend=<backend>
[RGB_CAMERA] opened=<true/false>
[RGB_CAMERA] width=<w> height=<h> fps=<fps>
```

The RGB panel title includes the selected source, and startup status shows:

```text
RGB camera: source=1 640x480 @ 30.0 fps
```

Combined logging receives `rgb_camera_selected` metadata/event fields:

```text
source, resolved_source, backend, opened, width, height, fps, prefer_external
```

## Video Recording

`--rgb-record-video` continues to record the already annotated RGB frame from the existing worker pipeline:

```text
selected camera source -> frame capture -> RGB posture overlay -> UI display -> same annotated frame to video writer
```

No second camera is opened for video. Video start/stop/failure events include selected camera source metadata.

## Validation Status

Passed:

```powershell
python -m py_compile run_ti_style_visualizer.py rgb_camera_panel.py combined_session_logger.py
python run_ti_style_visualizer.py --help
python run_ti_style_visualizer.py --rgb-list-cameras
python run_ti_style_visualizer.py --rgb-list-cameras --rgb-camera-backend auto --rgb-camera-probe-max-index 10
```

Help includes:

```text
--rgb-list-cameras
--rgb-camera-probe-max-index
--rgb-prefer-external
```

Live external-camera display was not claimed beyond OpenCV probe success.

## Final Git Status

Camera-selection files changed by this task:

```text
M rgb_camera_panel.py
M run_ti_style_visualizer.py
?? RGB_EXTERNAL_CAMERA_SELECTION.md
```

The worktree also contained pre-existing modified/deleted log files, pyc files, human-model files, and other reports before/during this task. Those were not intentionally reverted or normalized.

## Camera Listing Command

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --rgb-list-cameras `
  --rgb-camera-backend auto `
  --rgb-camera-probe-max-index 10
```

## Full External Camera Test Command

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\external_usb_camera_test `
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

Auto external selection variant: replace `--rgb-source 1` with:

```powershell
--rgb-prefer-external
```

## Limitations

OpenCV on Windows may not expose friendly camera names, so output uses index, open state, resolution, FPS, and backend. Index `1` is the recommended external candidate from the current probe.
