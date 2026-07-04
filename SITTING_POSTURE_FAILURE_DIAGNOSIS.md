# Sitting Posture Failure Diagnosis

## 1. Executive conclusion

Tracking is not the bottleneck in the latest distance posture benchmark. The failure is in sit-vs-stand posture discrimination and becomes severe at 3m and 4m.

We are not applying random threshold changes.
We are not adding target-only posture rules yet.
We are first determining whether the failure is model probability, decision gating, or geometry/feature availability.

## 2. Tracking is not the bottleneck

The current segment diagnostics show tracking presence at 100%, ID switches at 0, and extra track rate at 0 across the benchmark segments. Standing posture accuracy is about 99-100%, including far segments with high NO_POINTS rates.

## 3. Sitting posture failure summary

| Segment | Accuracy | Display STANDING | Display SITTING | Mean stand prob | Mean sit prob | Failure type |
|---|---:|---:|---:|---:|---:|---|
| sitting_1m | 0.674 | 0.287 | 0.674 | 0.329 | 0.585 | MIXED |
| sitting_2m | 0.747 | 0.183 | 0.747 | 0.338 | 0.566 | MIXED |
| sitting_3m | 0.473 | 0.485 | 0.473 | 0.361 | 0.498 | MIXED |
| sitting_4m | 0.000 | 0.950 | 0.000 | 0.559 | 0.334 | MIXED |

## 4. Posture input data flow

The posture model receives a 176-float vector built from an 8-frame window of 22-float per-frame posture features. Each 22-float frame contains target-level kinematics plus up to five associated point entries. Missing point entries are zero-padded.

## 5. Meaning of NO_POINTS / geom_pts / assoc

NO_POINTS is the overlay quality label emitted when the current frame has zero associated points for the target. `geom_pts` is the associated point count used for point geometry. Association modes describe whether point evidence came from target-index matching, nearest-neighbor fallback, or no association.

## 6. Probability-level diagnosis

During sitting_4m, stand_prob is actually higher than sit_prob: mean stand=0.559, mean sit=0.334, mean margin=0.226. That segment has 733 frames where stand_prob > sit_prob and 305 frames where sit_prob > stand_prob.

During sitting_3m, the mean probabilities do not favor STANDING: mean stand=0.361, mean sit=0.498. The mean gap is larger than the 0.10 ambiguity rule in favor of SITTING, but the segment is frame-mixed: 340 frames still have stand_prob > sit_prob, and display STANDING remains common.

During sitting_1m/2m, the model is better: sitting_1m mean sit=0.585 vs stand=0.329; sitting_2m mean sit=0.566 vs stand=0.338. Display still shows residual STANDING at 0.287 and 0.183.

## 7. Geometry / point-evidence diagnosis

NO_POINTS and low geometry are much more damaging to sitting than to standing. Standing remains correct even with high NO_POINTS rates at 3m and 4m, while sitting fails under similar or lower geometry availability. The current data proves sparse geometry is part of the sitting failure context; it does not prove that a runtime NO_POINTS rule would be correct.

## 8. Sensor/cfg/static seated target diagnosis

Latest benchmark cfg: `C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg`
Sensor position line: `sensorPosition 2 0 15`
Static range-angle line: `staticRangeAngleCfg -1 0 8 8`
Static-retention/fine-motion cfg available for later testing: `C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg`

The metric that would prove static retention helps is an A/B increase in seated `mean_geom_pts` and `geom_pts_ge_3_rate` that also increases sitting accuracy or shifts mean sit probability above mean stand probability at 3m/4m.

## 9. What is proven

Tracking is strong. Standing is nearly perfect. Sitting_4m is a model-probability failure under sparse geometry because mean stand probability exceeds mean sit probability. Sitting_3m is not a mean model-favors-standing case; it is mixed frame-level probability plus display/gating under sparse geometry. Sitting_1m/2m have better model probabilities but still show residual STANDING display.

## 10. What is not proven

The data does not prove that changing thresholds, holding previous sitting posture, suppressing target-only posture, changing the model, or changing cfg will fix the issue. It also does not prove that NO_POINTS alone causes failure, because standing succeeds with NO_POINTS.

## 11. Recommended next experiment

Run the current/original cfg versus the already available static-retention cfg on sitting at 2m, 3m, and 4m for 60 seconds each. Compare posture accuracy, stand/sit probabilities, NO_POINTS rate, mean_geom_pts, geom_pts_ge_3_rate, range error, range jitter, and time to stable sitting.

## 12. What not to change yet

Do not tune random posture thresholds, add target-only posture rules, add hold-previous-posture logic, suppress target-only posture, change the model, or modify cfg until the A/B test separates model probability, display/gating, and geometry availability.
