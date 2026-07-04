# Static-Retention Per-TID Diagnosis Report

## 1. Executive summary
Static retention failed primarily because it introduced persistent extra tracks at 3m and 4m while the real primary TID did not produce stable displayed SITTING. Recommendation: **Fix track validation / point association / primary target selection before posture tuning.** Static retention produced persistent extra TIDs during the failing 3m/4m segments, so posture tuning would be premature.

## 2. Why this diagnosis was needed
The A/B summary showed lower NO_POINTS in some static-retention segments but worse displayed sitting posture and 100% extra-track rate at 3m/4m. This report checks whether those failures come from extra TIDs, primary TID selection, point evidence assignment, probability/display mismatch, or model probabilities.

## 3. Sessions and segments inspected
| cfg_name | session_path | cfg_path | segment_file |
| --- | --- | --- | --- |
| default_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg | analysis_inputs/sitting_ab_default_segments.csv |
| static_retention_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_static_retention_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg | analysis_inputs/sitting_ab_static_retention_segments.csv |

Default segments:
| segment_id | expected_pose | expected_distance_m | start_time_s | end_time_s | duration_s | segmentation_method | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | SITTING | 2.000 | 106.578 | 176.719 | 70.141 | auto_range_plateau_trimmed | 1.000 |
| sitting_3m | SITTING | 3.000 | 188.734 | 267.453 | 78.719 | auto_range_plateau_trimmed | 1.000 |
| sitting_4m | SITTING | 4.000 | 279.453 | 340.750 | 61.297 | auto_range_plateau_trimmed | 0.736 |

Static-retention segments:
| segment_id | expected_pose | expected_distance_m | start_time_s | end_time_s | duration_s | segmentation_method | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | SITTING | 2.000 | 101.640 | 216.828 | 115.188 | auto_range_plateau_trimmed | 1.000 |
| sitting_3m | SITTING | 3.000 | 228.859 | 355.453 | 126.594 | auto_range_plateau_trimmed | 1.000 |
| sitting_4m | SITTING | 4.000 | 367.453 | 441.797 | 74.344 | auto_range_plateau_trimmed | 1.000 |

## 4. Per-TID metrics summary
| cfg_name | segment_id | tid | presence_rate_within_segment | mean_range_m | range_mae_vs_expected_m | mean_geom_pts | NO_POINTS_rate | display_standing_rate | display_sitting_rate | mean_stand_prob | mean_sit_prob | mean_stand_minus_sit_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_cfg | sitting_2m | 0 | 1.000 | 1.952 | 0.053 | 0.646 | 0.880 | 0.076 | 0.924 | 0.332 | 0.567 | -0.235 |
| default_cfg | sitting_3m | 0 | 1.000 | 2.804 | 0.203 | 1.806 | 0.593 | 0.484 | 0.491 | 0.369 | 0.544 | -0.175 |
| default_cfg | sitting_4m | 0 | 1.000 | 3.442 | 0.558 | 0.261 | 0.884 | 0.388 | 0.595 | 0.227 | 0.522 | -0.295 |
| static_retention_cfg | sitting_2m | 0 | 1.000 | 1.936 | 0.064 | 2.133 | 0.707 | 0.681 | 0.040 | 0.472 | 0.262 | 0.210 |
| static_retention_cfg | sitting_3m | 0 | 1.000 | 3.004 | 0.039 | 0.782 | 0.833 | 0.926 | 0.003 | 0.596 | 0.286 | 0.311 |
| static_retention_cfg | sitting_3m | 5 | 1.000 | 5.143 | 2.143 | 1.173 | 0.286 | 0.004 | 0.886 | 0.210 | 0.580 | -0.369 |
| static_retention_cfg | sitting_4m | 0 | 1.000 | 3.902 | 0.099 | 1.656 | 0.785 | 0.649 | 0.000 | 0.488 | 0.394 | 0.094 |
| static_retention_cfg | sitting_4m | 5 | 1.000 | 5.123 | 1.123 | 1.246 | 0.245 | 0.033 | 0.846 | 0.201 | 0.599 | -0.398 |

## 5. Real primary vs extra target classification
| cfg_name | segment_id | tid | classification | primary_tid_for_segment | evidence |
| --- | --- | --- | --- | --- | --- |
| default_cfg | sitting_2m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.053m |
| default_cfg | sitting_3m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.203m |
| default_cfg | sitting_4m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.558m |
| static_retention_cfg | sitting_2m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.064m |
| static_retention_cfg | sitting_3m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.039m |
| static_retention_cfg | sitting_3m | 5 | LIKELY_EXTRA_STATIC | 0 | persistent extra TID offset from expected range; presence=1.000, range_mae=2.143m |
| static_retention_cfg | sitting_4m | 0 | REAL_PRIMARY | 0 | closest persistent TID; presence=1.000, range_mae=0.099m |
| static_retention_cfg | sitting_4m | 5 | LIKELY_EXTRA_STATIC | 0 | persistent extra TID offset from expected range; presence=1.000, range_mae=1.123m |

## 6. Static-retention extra-track regression
Did static retention fail because it created extra tracks? **Yes.**
TID 5=LIKELY_EXTRA_STATIC (persistent extra TID offset from expected range; presence=1.000, range_mae=2.143m)
TID 5=LIKELY_EXTRA_STATIC (persistent extra TID offset from expected range; presence=1.000, range_mae=1.123m)

## 7. Probability/display mismatch analysis
| cfg_name | segment_id | tid | sit_prob_gt_stand_prob_rate | display_sitting_rate | display_standing_rate | mismatch_rate | mean_stand_prob | mean_sit_prob | mean_stand_minus_sit_margin | frames_stand_prob_gt_sit_prob | frames_sit_prob_gt_stand_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static_retention_cfg | sitting_3m | 0 | 0.086 | 0.003 | 0.926 | 0.084 | 0.596 | 0.286 | 0.311 | 2103 | 199 |
| static_retention_cfg | sitting_3m | 5 | 0.983 | 0.886 | 0.004 | 0.108 | 0.210 | 0.580 | -0.369 | 39 | 2263 |
| static_retention_cfg | sitting_4m | 0 | 0.283 | 0.000 | 0.649 | 0.283 | 0.488 | 0.394 | 0.094 | 969 | 383 |
| static_retention_cfg | sitting_4m | 5 | 0.982 | 0.846 | 0.033 | 0.146 | 0.201 | 0.599 | -0.398 | 25 | 1327 |

At static sitting_4m, the per-TID table above identifies which TID has sit_prob > stand_prob. A mismatch exists when that same TID has high sit_prob_gt_stand_prob_rate but display_sitting_rate remains near zero.
Static sitting_3m: TIDs with sit_prob > stand_prob: TID 5 (sit=0.580, stand=0.210, display_sitting=0.886, display_standing=0.004). TIDs with stand_prob > sit_prob: TID 0 (sit=0.286, stand=0.596, display_sitting=0.003, display_standing=0.926).
Static sitting_4m: TIDs with sit_prob > stand_prob: TID 5 (sit=0.599, stand=0.201, display_sitting=0.846, display_standing=0.033). TIDs with stand_prob > sit_prob: TID 0 (sit=0.394, stand=0.488, display_sitting=0.000, display_standing=0.649).
Interpretation: the segment-level sit probability can be pulled upward by extra TID 5, while the real primary TID 0 still favors STANDING and is the correct target by range.
Does display/gating fail despite sit_prob being higher? **Not as the only cause.**

## 8. Point association analysis
Raw point clouds were not logged, and the current raw track table has `num_associated_points=0` for every static-retention TID. Therefore this report uses posture `num_points` as `geom_pts` and frame `num_points` as total point evidence. Association mode fields are `NA` when no `assoc` column is present.
| cfg_name | segment_id | tid | mean_geom_pts | NO_POINTS_rate | LOW_POINTS_rate | OK_rate | assoc_target_index_rate | assoc_nearest_rate | assoc_auto_none_rate | mean_points_total | mean_geom_to_total_ratio | presence_rate_within_segment | mean_range_m | range_mae_vs_expected_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_cfg | sitting_2m | 0 | 0.646 | 0.880 | 0.070 | 0.049 | NA | NA | NA | 1.150 | 0.447 | 1.000 | 1.952 | 0.053 |
| default_cfg | sitting_3m | 0 | 1.806 | 0.593 | 0.284 | 0.113 | NA | NA | NA | 2.702 | 0.694 | 1.000 | 2.804 | 0.203 |
| default_cfg | sitting_4m | 0 | 0.261 | 0.884 | 0.110 | 0.002 | NA | NA | NA | 0.604 | 0.362 | 1.000 | 3.442 | 0.558 |
| static_retention_cfg | sitting_2m | 0 | 2.133 | 0.707 | 0.209 | 0.036 | NA | NA | NA | 5.669 | 0.194 | 1.000 | 1.936 | 0.064 |
| static_retention_cfg | sitting_3m | 0 | 0.782 | 0.833 | 0.130 | 0.017 | NA | NA | NA | 2.720 | 0.149 | 1.000 | 3.004 | 0.039 |
| static_retention_cfg | sitting_3m | 5 | 1.173 | 0.286 | 0.714 | 0.000 | NA | NA | NA | 2.720 | 0.704 | 1.000 | 5.143 | 2.143 |
| static_retention_cfg | sitting_4m | 0 | 1.656 | 0.785 | 0.141 | 0.025 | NA | NA | NA | 3.579 | 0.202 | 1.000 | 3.902 | 0.099 |
| static_retention_cfg | sitting_4m | 5 | 1.246 | 0.245 | 0.755 | 0.000 | NA | NA | NA | 3.579 | 0.668 | 1.000 | 5.123 | 1.123 |
Static retention gives the extra TID its own seated-looking evidence at 3m/4m: TID 5 has high SITTING probability and much lower NO_POINTS than TID 0. The real primary TID 0 gets better range stability, but at 3m it has less geom_pts than the default primary and at 4m it still favors STANDING despite more geom_pts than default.
Did it attach useful sitting points to the wrong TID? **Partly supported but not fully proven: extra TID 5 carries seated-looking probabilities and lower NO_POINTS at 3m/4m, while exact raw point-to-TID assignment cannot be reconstructed because point coordinates and assoc modes were not logged.**

## 9. Per-distance diagnosis: 2m
TID 0 is the real primary by range/persistence: range_mae=0.064m, sit_prob=0.262, stand_prob=0.472, display_sitting=0.040, display_standing=0.681.

## 10. Per-distance diagnosis: 3m
TID 0 is the real primary by range/persistence: range_mae=0.039m, sit_prob=0.286, stand_prob=0.596, display_sitting=0.003, display_standing=0.926.
TID 5=LIKELY_EXTRA_STATIC (persistent extra TID offset from expected range; presence=1.000, range_mae=2.143m)

## 11. Per-distance diagnosis: 4m
TID 0 is the real primary by range/persistence: range_mae=0.099m, sit_prob=0.394, stand_prob=0.488, display_sitting=0.000, display_standing=0.649.
TID 5=LIKELY_EXTRA_STATIC (persistent extra TID offset from expected range; presence=1.000, range_mae=1.123m)

## 12. What is proven
- Extra/duplicate static-retention tracks are present in the failing 3m/4m segments: Yes.
- The per-TID probabilities and display labels can be evaluated from the same `mmwave_pose.csv` row keyed by frame/TID.
- Static-retention posture failure is not explained by tracking dropout; the real primary TID remains present by range/persistence.

## 13. What is not proven
- Raw radar point coordinates were not logged, so exact point-to-TID spatial assignment cannot be reconstructed.
- Renderer confirmation state is not available as a separate CSV in these sessions.
- This is offline log analysis only, not live radar validation.

## 14. Recommended next engineering path
**Fix track validation / point association / primary target selection before posture tuning.**
Static retention produced persistent extra TIDs during the failing 3m/4m segments, so posture tuning would be premature.
Do not tune posture thresholds or retrain the model until the extra-track/association behavior is isolated with offline replay or a narrower cfg experiment.

## 15. Plots and generated files
- `per_tid_segment_metrics.csv`: per cfg/segment/TID metrics.
- `tid_classification_by_segment.csv`: real primary vs extra/ghost/wrong-association labels.
- `probability_display_mismatch.csv`: sit probability vs displayed pose mismatch rates.
- `point_association_by_tid.csv`: geometry and point evidence by TID.
- `plots/static_sitting_3m_tid_range_timeline.png`: TID range separation at static 3m.
- `plots/static_sitting_4m_tid_range_timeline.png`: TID range separation at static 4m.
- `plots/static_sitting_3m_tid_pose_timeline.png`: display pose by TID at static 3m.
- `plots/static_sitting_4m_tid_pose_timeline.png`: display pose by TID at static 4m.
- `plots/static_sitting_3m_tid_stand_sit_probs.png`: stand/sit probabilities by TID at static 3m.
- `plots/static_sitting_4m_tid_stand_sit_probs.png`: stand/sit probabilities by TID at static 4m.
- `plots/static_sitting_3m_tid_geom_pts.png`: geometry evidence by TID at static 3m.
- `plots/static_sitting_4m_tid_geom_pts.png`: geometry evidence by TID at static 4m.
- `plots/default_vs_static_tid_count_by_time.png`: active TID count comparison.

## Final answer
- Did static retention fail because it created extra tracks? **Yes.**
- Did it attach useful sitting points to the wrong TID? **Partly supported but not fully proven: extra TID 5 carries seated-looking probabilities and lower NO_POINTS at 3m/4m, while exact raw point-to-TID assignment cannot be reconstructed because point coordinates and assoc modes were not logged.**
- Did the model probability fail on the real TID? **Yes, on the real primary TID in the inspected failing distances.**
- Did display/gating fail despite sit_prob being higher? **Not as the only cause.**
- What should we fix next? **Fix track validation / point association / primary target selection before posture tuning.**
