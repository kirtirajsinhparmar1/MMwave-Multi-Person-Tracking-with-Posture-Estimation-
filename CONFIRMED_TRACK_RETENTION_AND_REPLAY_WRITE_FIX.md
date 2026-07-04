# Confirmed Track Retention And Replay Write Fix

## 1. Problem Summary

Two failures were addressed:

- TI replay JSON writing could crash the GUI thread with `FileNotFoundError` when `binData/<session_name>` was missing.
- Human-model validation could demote an already confirmed, stationary person to `SUSPECT_GHOST` after repeated `NO_POINTS` / `TARGET_ONLY` / `auto_none` evidence, causing the model to disappear even though the radar TID was still active.

## 2. Files Modified

- `ti_style_vendor/common/gui_parser.py`
- `ti_style_pose_overlay.py`
- `human_model_renderer.py`
- `run_ti_style_visualizer.py`
- `CONFIRMED_TRACK_RETENTION_AND_REPLAY_WRITE_FIX.md`

## 3. Replay JSON Directory/Write Fix

`gui_parser.py` now uses `pathlib.Path` and a shared `_write_replay_json()` helper for both active replay JSON write paths.

The writer creates:

```text
binData/<session_name>
```

before writing:

```text
replay_<chunk>.json
```

Replay writes are wrapped in `try/except`. If directory creation, JSON serialization, or file writing fails, the GUI thread prints:

```text
[replay-warning] failed to write ...
```

and continues parsing live UART frames.

## 4. Why Stationary Confirmed People Are Not Ghosts

Confirmed active tracks are no longer demoted to `SUSPECT_GHOST` because of weak current evidence alone. A confirmed TID remains `CONFIRMED` while it is still active, even when the current frame has `NO_POINTS`, `TARGET_ONLY`, or `auto_none`.

The debug reason for this case is:

```text
confirmed_retained_despite_low_evidence
```

## 5. Updated Validation Behavior

- New TIDs still start as `PROVISIONAL`.
- `PROVISIONAL` tracks still need enough good evidence before becoming `CONFIRMED`.
- Weak provisional tracks can still become `SUSPECT_GHOST`.
- `CONFIRMED` tracks remain visible while active.
- Lost tracks are removed through stale cleanup after the configured stale TTL.

## 6. Provisional NO_POINTS Tracks

Repeated `NO_POINTS` evidence on an unconfirmed/provisional track can still produce:

```text
state=SUSPECT_GHOST
rendered=false
reason=provisional_no_points_ghost
```

This preserves the ghost/shadow protection for new weak tracks.

## 7. Confirmed NO_POINTS Tracks

An already confirmed active person with weak current geometry should now remain:

```text
state=CONFIRMED
quality=NO_POINTS
rendered=true
reason=confirmed_retained_despite_low_evidence
```

The full human model remains eligible to render using the latest available display pose.

## 8. Stale Cleanup Behavior

Renderer stale cleanup continues to run from `update_models()` even when the current render record list is empty. The stale removal debug line now reports:

```text
[HUMAN_UI] removed stale tid=<tid> age=<age> reason=removed_stale_not_active
```

Expected after stale TTL:

```text
active_tracks=0
renderer_items=0
rendered=0
```

## 9. Validation Status

Passed:

```powershell
python -m py_compile ti_style_vendor\common\gui_parser.py
python -m py_compile human_model_renderer.py ti_style_pose_overlay.py run_ti_style_visualizer.py
python run_ti_style_visualizer.py --help
python run_ti_style_visualizer.py --rgb-list-cameras --rgb-camera-backend auto --rgb-camera-probe-max-index 10
```

Camera probe result in this shell:

```text
indices 0..10 opened=False
recommended external candidates: []
```

OpenCV also printed camera-index-out-of-range messages during probing. The command exited successfully.

Live COM6/COM7 validation was not run.

## 10. Final Git Status

Intentional source/report files for this task:

```text
M human_model_renderer.py
M run_ti_style_visualizer.py
M ti_style_pose_overlay.py
M ti_style_vendor/common/gui_parser.py
?? CONFIRMED_TRACK_RETENTION_AND_REPLAY_WRITE_FIX.md
```

The worktree also contains pre-existing unrelated modified/deleted log files, cache files, RGB files, vendored plot/people-tracking files, previous markdown reports, and captured replay/log folders. Those were not reverted.

## 11. Exact Command To Run

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\external_camera_confirmed_retention_test `
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

## 12. Limitations

- This pass does not change RGB camera selection, RGB tracking/posture logic, RGB recording, parser packet format, TI cfg, ONNX/posture inference, stand/sit gate logic, raw CSV logging, or combined logging architecture.
- Live radar/camera validation with the full GUI command still needs to be run on the hardware session.
- The legacy `--pose-human-model-bad-evidence-demote-frames` option is retained for compatibility/debug metadata, but confirmed active tracks are not demoted to ghosts by low evidence in this pass.
