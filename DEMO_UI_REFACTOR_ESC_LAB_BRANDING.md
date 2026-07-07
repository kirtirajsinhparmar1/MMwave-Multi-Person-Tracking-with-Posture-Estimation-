# ESC LAB Demo UI Refactor

## 1. Problem / Motivation

The existing TI-style visualizer was functional but looked like an engineering/debug UI. For paper recordings and live research demos, the UI needed a cleaner product-style dashboard while preserving the existing radar, RGB, posture, logging, and recording behavior.

## 2. Files Modified

- `run_ti_style_visualizer.py`
- `rgb_camera_panel.py`
- `ui_assets/embedded_sensing_and_computing_lab_logo.jpg`
- `DEMO_UI_REFACTOR_ESC_LAB_BRANDING.md`

No radar parser, COM startup, TI cfg, ONNX model, posture inference, smoothing/gating, RGB detector/tracker, logging schema, or model file behavior was changed.

## 3. New Demo Dashboard Layout

The new layout is enabled with `--ui-demo-layout`. It wraps the existing TI widgets into:

- ESC LAB branded header/title bar
- collapsible left controls sidebar
- center `mmWave 3D View` card containing the existing TI tab widget
- right dashboard column with `RGB Camera` on top and `Analysis` below

The previous layout remains available by running without `--ui-demo-layout`.

## 4. Collapsible Sidebar Implementation

The existing TI control widgets are re-parented into a scrollable sidebar. They are not recreated, so their existing callbacks remain intact.

- Expanded width: approximately 280 to 420 px
- Collapsed width: 34 px
- Toggle button: `<` / `>`
- Shortcut: `Ctrl+B`
- Initial collapsed state: `--ui-sidebar-collapsed`

## 5. ESC LAB Branding / Logo Handling

The logo was copied from:

```text
C:\Users\UBESC\Downloads\embedded_sensing_and_computing_lab_logo.jpg
```

to:

```text
ui_assets/embedded_sensing_and_computing_lab_logo.jpg
```

The window title and demo header now use:

```text
ESC LAB Multi-Person mmWave + RGB Posture Tracking
```

If the source logo is unavailable in a future checkout, the UI falls back to text branding.

## 6. RGB Top-Right Panel Cleanup

The RGB panel can now run in compact display mode. By default, it hides noisy UI status such as frame count, FPS spam, and recording-frame counters while preserving:

- RGB result signals
- RGB keypoint logging
- terminal/debug prints
- annotated video recording
- camera selection behavior

Use `--ui-show-debug-status` to show detailed status text in the UI again.

## 7. Analysis Placeholder Panel

The right column now includes a large `Analysis` card below the RGB camera with placeholder text:

```text
Distance / posture / tracking diagnostics will appear here.
```

No fake metrics or plots were added.

## 8. New CLI Flags

```text
--ui-demo-layout
--ui-sidebar-collapsed
--ui-show-debug-status
```

## 9. Validation Status

Passed:

```powershell
python -m py_compile run_ti_style_visualizer.py rgb_camera_panel.py human_model_renderer.py ti_style_pose_overlay.py
python -m py_compile ti_style_vendor\common\gui_core.py
python -m py_compile ti_style_vendor\common\Demo_Classes\people_tracking.py
python run_ti_style_visualizer.py --help
```

Confirmed `--help` includes:

```text
--ui-demo-layout
--ui-sidebar-collapsed
--ui-show-debug-status
```

Live COM6/COM7 hardware validation was not run as part of this static pass.

## 10. Final Git Status

Observed pending changes after this refactor:

```text
 M rgb_camera_panel.py
 M run_ti_style_visualizer.py
?? DEMO_UI_REFACTOR_ESC_LAB_BRANDING.md
?? ui_assets/
```

Generated `.pyc` files from validation were reverted.

## 11. Exact Command To Run

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --ui-demo-layout `
  --ui-sidebar-collapsed `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\demo_ui_refactor_test `
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

## 12. Known Limitations / Rollback

- The `Analysis` panel is intentionally empty for now.
- The old demo/device selector behavior remains primarily a TI legacy path; the dashboard is intended for the configured 3D People Tracking demo.
- Run without `--ui-demo-layout` to use the previous layout if the demo dashboard has an issue.

## 13. Screenshots

No screenshots were captured during this pass.
