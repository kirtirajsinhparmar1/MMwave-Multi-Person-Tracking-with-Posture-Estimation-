# STANDING to SITTING Gate Fix

## Problem observed

The current best live command can initially show `STANDING`, then drift into `SITTING` after enough weak sitting frames accumulate. Logs often show `pts=0`, `quality=NO_POINTS`, and target-only fallback behavior, so weak or ambiguous SITTING evidence should not be allowed to replace a stable displayed STANDING pose.

## Files modified

- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `STAND_TO_SIT_GATE_FIX.md`

No RGB repo files, radar startup code, cfg files, human model rendering, or ONNX model files were modified.

## Standing-to-sitting gate logic

The strict gate applies only when the currently displayed pose is `STANDING` and the new display candidate is `SITTING`.

`STANDING -> SITTING` now requires all of:

- `sitting_confidence >= --pose-stand-to-sit-min-confidence`
- `sitting_prob - standing_prob >= --pose-stand-to-sit-margin`
- evidence persists for `--pose-stand-to-sit-frames`
- quality is not `NO_POINTS`, `TARGET_ONLY`, or `NO_ASSOC_POINTS`, unless `--pose-stand-to-sit-allow-target-only` is explicitly enabled

Blocked or waiting gate frames keep the previous displayed `STANDING` pose and do not feed SITTING into the display-history stability window.

## Sitting lock recovery logic

When the displayed pose is already `SITTING`, the overlay now counts frames where standing evidence beats sitting evidence.

Default recovery:

```text
display_pose == SITTING
AND standing_prob - sitting_prob >= 0.10
for 6 frames
=> display STANDING
```

If stand/sit probabilities are unavailable, recovery can use raw/smoothed `STANDING` confidence as a conservative fallback.

## New CLI flags and defaults

```text
--pose-stand-to-sit-min-confidence 0.65
--pose-stand-to-sit-margin 0.15
--pose-stand-to-sit-frames 12
--pose-stand-to-sit-allow-target-only false

--pose-sit-to-stand-recovery-margin 0.10
--pose-sit-to-stand-recovery-frames 6
```

Do not pass `--pose-stand-to-sit-allow-target-only` for the first test. The point of this test is to block target-only SITTING from replacing stable STANDING.

## Debug fields added

Pose debug and `pose_predictions_ui.csv` now include:

```text
stand_to_sit_gate=PASS/BLOCK/WAIT/NA
stand_to_sit_conf=<value>
stand_to_sit_margin=<value>
stand_to_sit_stable=<count>/<required>
stand_to_sit_quality_ok=<true/false>
sit_to_stand_recovery=<count>/<required>
reason=<specific reason>
```

Expected block reasons:

```text
stand_to_sit_blocked_target_only
stand_to_sit_blocked_margin
stand_to_sit_blocked_confidence
stand_to_sit_waiting_gate
stand_to_sit_gate_passed
sit_to_stand_recovery
```

## Compile status

Passed:

```powershell
python -m py_compile run_ti_style_visualizer.py ti_style_pose_overlay.py
```

## Help status

Passed. The new flags appeared in:

```powershell
python run_ti_style_visualizer.py --help
```

Confirmed flags:

```text
--pose-stand-to-sit-min-confidence
--pose-stand-to-sit-margin
--pose-stand-to-sit-frames
--pose-stand-to-sit-allow-target-only
--pose-sit-to-stand-recovery-margin
--pose-sit-to-stand-recovery-frames
```

No live COM6/COM7 validation has been claimed.

## Final git status

Task-relevant changes:

```text
M run_ti_style_visualizer.py
M ti_style_pose_overlay.py
?? STAND_TO_SIT_GATE_FIX.md
```

The worktree also contains pre-existing unrelated modified/generated files and logs.

## Exact command to run next

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\stand_to_sit_gate_test `
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
  --combined-log `
  --combined-status-panel `
  --log-root "..\logs" `
  --rgb-log-keypoints
```

## What to inspect

While standing still, false sitting should show one of:

```text
stand_to_sit_gate=BLOCK reason=stand_to_sit_blocked_target_only
stand_to_sit_gate=BLOCK reason=stand_to_sit_blocked_margin
stand_to_sit_gate=BLOCK reason=stand_to_sit_blocked_confidence
stand_to_sit_gate=WAIT reason=stand_to_sit_waiting_gate
```

If the display ever gets stuck as `SITTING`, check whether:

```text
sit_to_stand_recovery=6/6 reason=sit_to_stand_recovery
```
