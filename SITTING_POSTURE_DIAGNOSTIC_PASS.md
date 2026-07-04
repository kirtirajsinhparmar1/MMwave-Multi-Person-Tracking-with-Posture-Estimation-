# Sitting Posture Diagnostic Pass

## 1. Files inspected

- `ti_style_pose_overlay.py`
- `pose_feature_extractor.py`
- `pose_model_runtime.py`
- `pose_data_logger.py`
- `analysis/analyze_distance_posture_session.py`
- `analysis_outputs/latest_distance_posture_analysis_v2/posture_verdict_by_segment.csv`
- `analysis_outputs/latest_distance_posture_analysis_v2/stand_sit_probability_by_segment.csv`
- `analysis_outputs/latest_distance_posture_analysis_v2/no_points_effect_by_pose.csv`
- `analysis_outputs/latest_distance_posture_analysis_v2/combined_diagnostics_by_segment.csv`
- `analysis_outputs/latest_distance_posture_analysis_v2/segments_auto.csv`
- `analysis_outputs/latest_distance_posture_analysis_v2/candidate_sessions.csv`
- `C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540\mmwave_pose.csv`
- `C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540\session_metadata.json`
- `CFG_POSTURE_AUDIT.md`
- `cfg/ODS_6m_posture_tuned.cfg`
- `C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg`

## 2. Exact posture input data flow

`TIStylePoseOverlay.process_output_dict()` reads `trackData`, `pointCloud`, and `trackIndexes`. `_track_to_target()` converts each track row into target-level position, velocity, and acceleration fields. `_associate_points()` selects point evidence for that TID. `pose_feature_extractor.build_22_feature_vector()` builds seven target fields plus up to five associated point triples. `update_8_frame_window()` stores per-TID 22-feature frames. `build_176_feature_vector()` flattens the 8-frame window. `PoseModelRuntime.predict()` runs ONNX inference, and `PoseSmoother.update()` smooths probabilities before display/gating logic.

The model does not receive the raw point cloud directly. It receives point-derived `relative_y`, `z`, and `snr` slots after association and zero padding.

## 3. Meaning of NO_POINTS

`quality=NO_POINTS` means the current target has zero associated points after point association. It is not tracking loss. In this code path, target-level fields can still exist while point-geometry fields are absent. Missing point slots in the 22-feature frame are zero-filled, not carried forward.

## 4. Probability diagnosis result

`sitting_4m` is the clearest model-probability failure: mean stand probability is 0.559, mean sit probability is 0.334, and stand_prob > sit_prob in 733 frames.

`sitting_3m` is not a mean model-favors-standing case: mean stand probability is 0.361 and mean sit probability is 0.498. It is still frame-mixed, with 340 frames where stand_prob > sit_prob and frequent STANDING display.

`sitting_1m` and `sitting_2m` have better model probabilities in favor of SITTING, but display STANDING remains non-trivial at 0.287 and 0.183.

## 5. Geometry diagnosis result

Sitting failures occur under sparse geometry: NO_POINTS rates are 0.732 at 1m, 0.819 at 2m, 0.814 at 3m, and 0.755 at 4m, with mean geom_pts below 1.0 for all sitting segments.

Standing remains correct even under NO_POINTS: standing_3m has NO_POINTS rate 0.919 and accuracy 1.000; standing_4m has NO_POINTS rate 0.963 and accuracy 1.000. Therefore NO_POINTS alone is not sufficient to explain posture failure; it hurts sitting discrimination more than standing.

## 6. Sensor/cfg diagnosis result

The latest benchmark metadata points to:

`C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg`

The cfg contains:

- `sensorPosition 2 0 15`
- `staticRangeAngleCfg -1 0 8 8`

`CFG_POSTURE_AUDIT.md` identifies that static range-angle processing is disabled in this cfg. An ODS static-retention cfg is available for later testing:

`C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg`

## 7. Next A/B test plan

Run the plan in `NEXT_SITTING_AB_TEST_PLAN.md`: current/original cfg versus the already available static-retention/fine-motion cfg, sitting at 2m, 3m, and 4m for 60 seconds each.

Compare posture accuracy, display STANDING/SITTING rates, mean stand/sit probabilities, stand-minus-sit margin, NO_POINTS rate, mean geom_pts, geom_pts_ge_3_rate, range MAE, range jitter, and time to stable sitting.

## 8. Validation commands run

- `python -m py_compile analysis\analyze_distance_posture_session.py`
- `python -m py_compile analysis\diagnose_sitting_failure.py`
- `python analysis\diagnose_sitting_failure.py analysis_outputs\latest_distance_posture_analysis_v2`

These are static/script validations only. No live radar validation was run.

## 9. What should not be changed yet

We are not applying random threshold changes.
We are not adding target-only posture rules yet.
We are first determining whether the failure is model probability, decision gating, or geometry/feature availability.

Do not change runtime posture thresholds, cfg, model files, target-only posture rules, hold-previous-posture logic, or target-only posture suppression as part of this diagnostic pass.

## 10. What single engineering path should be tested next

Run the static-retention A/B test. If static retention increases seated `mean_geom_pts` and `geom_pts_ge_3_rate` and improves sitting accuracy, cfg/static seated point extraction is the likely fix path. If geometry improves but stand probability still dominates, the model/features need improvement. If sit probability dominates but display remains STANDING, decision/gating logic needs improvement.
