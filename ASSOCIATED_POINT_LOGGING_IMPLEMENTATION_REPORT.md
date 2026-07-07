# Associated Point Logging Implementation Report

## 1. Files Changed

- `run_ti_style_visualizer.py`
- `ti_style_pose_overlay.py`
- `associated_point_logger.py`
- `analysis\validate_associated_point_log.py`
- `ASSOCIATED_POINT_LOGGING_SCHEMA.md`
- `POINT_CLOUD_DATA_COLLECTION_COMMANDS.md`
- `ASSOCIATED_POINT_LOGGING_IMPLEMENTATION_REPORT.md`

No TI cfg files, ONNX model files, tracking logic, RGB code, posture decision logic, smoothing/gating logic, or human model rendering behavior were changed.

## 2. CLI Flags Added

```powershell
--pose-log-associated-points
--pose-associated-points-max-per-tid 64
--pose-associated-points-format csv
```

Defaults remain disabled, max 64 points per TID/frame, CSV format.

## 3. Output File Created

When enabled, the logger writes:

```text
<out>\mmwave_associated_points.csv
```

No heavy associated point file is created when `--pose-log-associated-points` is omitted.

## 4. Schema Summary

The CSV contains one row per associated radar point per TID per frame, plus one optional no-points summary row for a TID/frame when no associated points are available. The schema includes frame/TID identifiers, association source, point coordinates, target coordinates, target-centered relative coordinates, quality labels, and old pose model probabilities when available.

The full schema is documented in:

```text
ASSOCIATED_POINT_LOGGING_SCHEMA.md
```

## 5. Where Association Is Captured

Association is captured in `TiStylePoseManager.process_output_dict()` after the existing `_associate_points()` call. That location already has:

- `pointCloud`
- `trackData`
- `trackIndexes`
- `track_index_to_tid`
- associated points per TID
- existing pose probabilities
- displayed pose
- quality and geometry point counts

The existing association result is stored and passed to `AssociatedPointCloudLogger` only after the old pose result dictionary is finalized.

## 6. How Target-Centered Coordinates Are Computed

`associated_point_logger.py` computes:

```text
relative_x_m = point_x_m - target_x_m
relative_y_m = point_y_m - target_y_m
relative_z_m = point_z_m - target_z_m
relative_range_m = sqrt(relative_x_m^2 + relative_y_m^2 + relative_z_m^2)
height_above_ground_m = point_z_m - ground_z
```

The current coordinate convention follows the existing TI people-tracking path: `x` lateral, `y` forward/range, `z` vertical. `relative_radial_m` is logged as `relative_y_m`, and `relative_lateral_m` is logged as `relative_x_m`.

## 7. Performance Safeguards

- Logging is disabled by default.
- CSV rows are buffered and flushed periodically.
- The per-TID point cap defaults to 64.
- If a TID has more than the cap, selection keeps high-SNR points, high/low z points, and spatially spread points.
- The logger does not use the old top-5 highest-z feature selection.
- Row write failures are caught and do not crash the UI.
- Missing fields are written as blank values instead of causing exceptions.
- No-points frames use a single summary row per TID/frame instead of many empty rows.

## 8. Validation Commands Run

All requested non-live validation commands passed:

```powershell
python -m py_compile run_ti_style_visualizer.py
python -m py_compile ti_style_pose_overlay.py
python -m py_compile pose_data_logger.py
python -m py_compile associated_point_logger.py
python -m py_compile analysis\validate_associated_point_log.py
python run_ti_style_visualizer.py --help
```

No long live collection was run.

## 9. How To Run A 30-Second Smoke Test

Stand at 2m for about 30 seconds and run the smoke-test command in:

```text
POINT_CLOUD_DATA_COLLECTION_COMMANDS.md
```

Then validate the session:

```powershell
python analysis\validate_associated_point_log.py `
  --session "..\logs\pc_smoke_standing_2m_01" `
  --out analysis_outputs\associated_point_log_validation
```

## 10. How To Collect The Full New Dataset

Use the full command template in:

```text
POINT_CLOUD_DATA_COLLECTION_COMMANDS.md
```

Recommended session names:

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

## 11. Limitations

- Associated point logging currently requires `--enable-pose` because the association data is computed inside the existing pose manager.
- CSV is the only supported associated-point format.
- `point_noise` is blank unless the parsed point-cloud row supplies a usable noise field.
- Old model probability columns are auxiliary analysis fields only and must not be used as labels.
- The implementation has been compile/help validated but not live hardware validated in this pass.
