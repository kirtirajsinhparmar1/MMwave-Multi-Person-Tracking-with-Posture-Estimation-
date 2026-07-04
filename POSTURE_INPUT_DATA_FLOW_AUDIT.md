# Posture Input Data Flow Audit

This audit is based only on the current code paths in `ti_style_pose_overlay.py`, `pose_feature_extractor.py`, `pose_model_runtime.py`, `pose_data_logger.py`, and `analysis/analyze_distance_posture_session.py`.

## 1. What inputs go into posture inference?

`ti_style_pose_overlay.TIStylePoseOverlay.process_output_dict()` reads `trackData`, `pointCloud`, and `trackIndexes` from the mmWave output dictionary. For each track, `_track_to_target()` builds a target object, `_associate_points()` selects point evidence for that target, and `pose_feature_extractor.build_22_feature_vector()` builds the per-frame posture features. `pose_feature_extractor.update_8_frame_window()` stores those features per TID, `build_176_feature_vector()` flattens the 8-frame window, and `pose_model_runtime.PoseModelRuntime.predict()` runs the ONNX posture model.

## 2. Which features come from target-level tracking?

`pose_feature_extractor.FEATURE_NAMES_22` starts with target-level fields:

- `posz`
- `velx`
- `vely`
- `velz`
- `accx`
- `accy`
- `accz`

`build_22_feature_vector()` fills those from target fields `pos_z`, `vel_x`, `vel_y`, `vel_z`, `acc_x`, `acc_y`, and `acc_z`. In the live overlay, `_track_to_target()` maps these from `trackData` columns: TID, position, velocity, acceleration, and confidence.

## 3. Which features come from associated point cloud / geometry?

The remaining 15 fields in each 22-feature frame are five point slots:

- `y0`, `z0`, `snr0`
- `y1`, `z1`, `snr1`
- `y2`, `z2`, `snr2`
- `y3`, `z3`, `snr3`
- `y4`, `z4`, `snr4`

`build_22_feature_vector()` computes each point slot from associated points as relative Y (`point.y - target.y`), point Z, and SNR. It sorts candidate points by Z descending and uses the first five.

Separate geometry diagnostics are computed in `ti_style_pose_overlay._point_geometry()`, including `geom_pts`, centroids, top/bottom Z, height, spread, and range. Those geometry diagnostics are logged for analysis, but the posture ONNX model input is the 176-feature vector described above.

## 4. Does the posture model use point cloud points directly?

It does not pass the raw point cloud directly to the model. The model receives a numeric 176-feature vector. Point evidence can affect that vector through the selected associated point slots (`relative_y`, `z`, `snr`) for up to five points per frame.

## 5. If yes, how are points selected/associated to a TID?

The live selection is in `ti_style_pose_overlay._associate_points()`.

`_points_by_target_index()` checks the per-point `trackIndexes` array and point column 6 when present. A point matches if the candidate value is the TID or maps through `track_index_to_tid` to the TID.

`_points_by_nearest()` is the fallback geometry association. It selects points within the configured nearest-neighbor radius and Z limits around the target.

`_associate_points()` then chooses the final association set based on `assoc_method`:

- `target_index`: use target-index matches only.
- `nearest`: use nearest-neighbor matches only.
- `auto`/`hybrid`: prefer target-index matches, otherwise use nearest-neighbor matches when available.

## 6. What does quality=NO_POINTS mean in the code?

`ti_style_pose_overlay._quality_label()` returns `NO_POINTS` when the current target has zero associated points (`num_points <= 0`) after association. It is a quality label for the current frame's point evidence, not a tracking-loss label.

## 7. What does geom_pts mean?

In `ti_style_pose_overlay._point_geometry()`, `geom_pts` is `len(associated_points)`: the count of points associated to that target for the current frame. If there are no associated points, `geom_quality` remains `TARGET_ONLY`.

In `analysis/analyze_distance_posture_session.py`, normalized posture tables map `geom_pts` from `geom_pts`, `num_points`, or `num_associated_points`, depending on which column exists in the log.

## 8. What does points_total mean?

In `_associate_points()`, `points_total` is `len(points)`, the total number of point-cloud rows received for the frame before target association. It is not the number of points assigned to a specific TID.

## 9. What does assoc=index mean?

The current live `_associate_points()` final association labels are `target_index`, `nearest`, `auto_none`, `hybrid_target_index`, and `hybrid_nearest`. The current code does not emit a final `assoc=index` label in this function.

If an analysis table contains a normalized or older `assoc=index` value, the closest current-code behavior is target-index association: points are matched through `trackIndexes` or point column 6, with `track_index_to_tid` converting tracker indexes to TIDs.

## 10. What does assoc=target_index mean?

`assoc=target_index` means `_associate_points()` used `_points_by_target_index()` as the final point source. Points were accepted because their track-index candidate matched the target TID directly or mapped to that TID through `track_index_to_tid`.

## 11. What does assoc=auto_none mean?

`assoc=auto_none` means `assoc_method` was `auto`, target-index association found no points, nearest-neighbor fallback found no points, and the final associated point list is empty.

## 12. What happens to the 176-feature vector when no associated points exist?

`build_22_feature_vector()` still fills the seven target-level fields. The point fields are zero-padded until the 22-feature frame is complete. If `_can_use_frame_for_inference()` allows the frame, `update_8_frame_window()` appends that 22-feature frame to the TID window. `build_176_feature_vector()` then flattens the 8-frame window channel-major into 176 floats.

If the TID has fewer than eight stored frames, `get_8_frame_window()` pads missing history frames with all-zero 22-feature rows before flattening.

## 13. Are missing point features zero-filled, carried from previous frames, replaced by target-only features, or marked invalid?

Missing point slots are zero-filled by `build_22_feature_vector()`. They are not carried forward from previous frames and are not replaced by additional target-only fields. The quality metadata records low point count through `FeatureQuality`, including `num_points`, `low_quality`, and a reason such as `only_0_associated_points`.

## 14. Is posture being inferred from only target data when point geometry is missing?

Conditionally, yes. `ti_style_pose_overlay._can_use_frame_for_inference()` returns `self.allow_target_only` when `num_points <= 0`; otherwise it requires `num_points >= self.min_associated_points_for_inference`. If `allow_target_only` is enabled, a zero-associated-point frame can enter the 8-frame model window as target-level fields plus zero-filled point slots. If `allow_target_only` is disabled, that current frame is not used for inference.

The analysis logs also show `quality=NO_POINTS` frames. Those frames mean the current posture evidence lacks associated point geometry, but tracking can still be present and target-level fields can still exist.

## Logger and analyzer behavior

`pose_data_logger.PoseDataLogger.write_feature22()` and `write_feature176()` log feature vectors and quality metadata that were already computed; they do not change inference behavior.

`analysis/analyze_distance_posture_session.py` normalizes log aliases, computes segment metrics, computes stand/sit probability summaries, and buckets NO_POINTS/LOW_POINTS/HAS_POINTS for analysis. It does not run the model or alter live posture decisions.
