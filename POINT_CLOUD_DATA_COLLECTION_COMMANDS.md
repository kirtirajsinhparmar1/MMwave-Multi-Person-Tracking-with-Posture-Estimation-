# Point-Cloud Data Collection Commands

Use these commands after the associated point logger has been validated with a short smoke test.

## 30-Second Smoke Test

Stand at 2m for about 30 seconds and run:

```powershell
python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out "..\logs\pc_smoke_standing_2m_01" `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-log `
  --pose-log-associated-points `
  --pose-associated-points-max-per-tid 64 `
  --pose-associated-points-format csv
```

Then validate:

```powershell
python analysis\validate_associated_point_log.py `
  --session "..\logs\pc_smoke_standing_2m_01" `
  --out analysis_outputs\associated_point_log_validation
```

## Full Collection Command Template

Change only the `--out` session name for each posture protocol:

```powershell
python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out "..\logs\<SESSION_NAME>" `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-log `
  --pose-log-associated-points `
  --pose-associated-points-max-per-tid 64 `
  --pose-associated-points-format csv
```

## Recommended Session Names

```text
pc_standing_center_1to5_01
pc_sitting_leanback_center_1to5_01
pc_sitting_upright_center_1to5_01
pc_sitting_leanforward_center_1to5_01
pc_two_person_standing_lr_1to5_01
pc_two_person_sitting_leanback_lr_1to5_01
pc_two_person_sitting_upright_lr_1to5_01
pc_two_person_sitting_leanforward_lr_1to5_01
```

Keep the same user-provided segment protocol style as the previous collection pass. The associated point file supplies geometry; segment labels still come from protocol timing, not from the old displayed pose.
