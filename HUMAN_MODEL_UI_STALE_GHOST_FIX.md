# Human Model UI Stale Ghost Fix

## 1. Problem Description

The 3D human model overlay could show duplicate or frozen posture figures even when radar/posture behavior was otherwise acceptable. The likely UI failure mode was stale GL objects remaining in the scene after a radar target ID disappeared or switched, while new target IDs continued rendering correctly.

This fix is limited to 3D UI human model lifecycle management. It does not change radar detection, posture classification, ONNX inference, confidence thresholds, smoothing, stand/sit gate logic, RGB processing, cfg startup, or logging architecture.

## 2. Files Modified

- `human_model_renderer.py`
- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `ti_style_vendor/common/Common_Tabs/plot_3d.py`
- `ti_style_vendor/common/Demo_Classes/people_tracking.py`

## 3. Renderer Lifecycle Fix

`HumanPoseModelRenderer.update_models()` now performs strict per-frame reconciliation:

- Builds the active target ID set from the current radar tracks / pose records.
- Keeps at most one GL human model item per target ID.
- Creates an item when an active target ID first appears.
- Updates the item transform and position every frame, even when the pose label is unchanged.
- Updates/recreates the mesh only when the model/pose changes.
- Ensures active items are visible.
- Removes invalid or non-renderable cached items safely.

This prevents a valid target from leaving an old human model fixed at a previous `x/y/z` while the person moves.

## 4. Stale-TID Cleanup Logic

The renderer now tracks `last_seen_frame` and last known position for every rendered target ID.

For every rendered target ID that is no longer present in the active target set:

- The item is treated as stale.
- Its age is computed from the current frame minus `last_seen_frame`.
- Once age exceeds `--pose-human-model-stale-frames`, the item is removed from the GL view and deleted from the renderer cache.

Default:

```powershell
--pose-human-model-stale-frames 10
```

## 5. Ghost-Distance Cleanup Logic

A stale renderer item is removed faster when it is close to an active current track, which handles target ID switches that otherwise look like duplicate shadows.

Default:

```powershell
--pose-human-model-ghost-distance-m 0.75
```

Only stale renderer items are removed by this rule. Active current tracks are never removed just because they are close to another active track, so real multi-person scenes remain visible.

## 6. Debug Fields And CLI Flags

New/extended CLI flags:

```powershell
--pose-human-model-debug
--pose-human-model-stale-frames
--pose-human-model-ghost-distance-m
```

The renderer prints throttled summary diagnostics every 30 frames:

```text
[HUMAN_UI] frame=<frame> active_tracks=<count> active_tids=<...> renderer_items=<count> renderer_tids=<...> stale_tids=<...>
```

When `--pose-human-model-debug` is enabled, each item also prints detail:

```text
[HUMAN_UI] tid=<tid> pos=(x,y,z) pose=<pose> last_seen=<frame> age=<n> visible=<true/false> item_id=<...>
```

The existing pose status text also includes:

```text
Human UI: active_tracks=<n>, rendered=<n>, stale=<n>
```

## 7. Shutdown Cleanup

Added safe renderer cleanup:

```python
clear_all_human_models()
```

It removes all human model GL items from the view, clears renderer caches, and resets stale tracking maps. It is called during shutdown and registered through `atexit` when the human model renderer is created.

Cleanup is fail-safe: removal exceptions are reported as `[human-model-warning]` messages and do not crash the UI.

## 8. Compile Status

Passed:

```powershell
rtk python -m py_compile human_model_renderer.py ti_style_pose_overlay.py run_ti_style_visualizer.py
```

## 9. Help Status

Passed:

```powershell
rtk python run_ti_style_visualizer.py --help
```

Confirmed the help output includes:

```text
--pose-human-model-debug
--pose-human-model-stale-frames
--pose-human-model-ghost-distance-m
```

## 10. Final Git Status

Final git status still includes pre-existing unrelated dirty files such as deleted log CSV/JSON files, RGB/video-related edits, and generated `__pycache__` files. The human-model UI source files changed by this task are:

```text
human_model_renderer.py
ti_style_pose_overlay.py
run_ti_style_visualizer.py
ti_style_vendor/common/Common_Tabs/plot_3d.py
ti_style_vendor/common/Demo_Classes/people_tracking.py
HUMAN_MODEL_UI_STALE_GHOST_FIX.md
```

## 11. Exact Command To Run

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\human_ui_stale_cleanup_test `
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
  --pose-human-model-stale-frames 10 `
  --pose-human-model-ghost-distance-m 0.75 `
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

## 12. UI Ghost Vs Radar Ghost

Use the `[HUMAN_UI]` lines to identify the source:

```text
active_tracks=1 renderer_items=2 stale_tids=[...]
```

This indicates a UI stale renderer item. After the fix, that stale item should disappear after `--pose-human-model-stale-frames`, or sooner if it is within `--pose-human-model-ghost-distance-m` of an active track.

```text
active_tracks=2 renderer_items=2
```

This indicates radar/tracker is currently reporting two active tracks. The UI keeps both visible because both are active, so that is not treated as a stale UI ghost.

For a frozen model, look for a renderer target ID that is no longer in `active_tids` and whose `age` keeps increasing. It should be removed once stale cleanup triggers.
