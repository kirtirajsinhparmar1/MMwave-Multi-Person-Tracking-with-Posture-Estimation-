# Posture Filter Model Report

This is an offline grouped-validation report. It does not claim runtime improvement.

Acceptance criteria passed: no
Decision: No candidate passed all offline acceptance criteria; no runtime model should be exported.

Feature count: 32

## Model Comparison

| accuracy | standing_accuracy | sitting_accuracy | false_SITTING_on_STANDING | false_STANDING_on_SITTING | false_SITTING_on_STANDING_3m | upright_sitting_accuracy | model | validation | heldout_groups | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.511 | 0.964 | 0.358 | 0.034 | 0.551 | 0.101 | 0.390 | baseline_current_display | all_labeled_examples | not_applicable |  |
| 0.516 | 0.868 | 0.397 | 0.132 | 0.578 | 0.145 | 0.460 | baseline_raw_max_probability | all_labeled_examples | not_applicable |  |
| 0.511 | 0.964 | 0.358 | 0.034 | 0.551 | 0.101 | 0.390 | baseline_refined_gate_display_if_available | all_labeled_examples | not_applicable |  |
| 0.749 | 0.396 | 0.868 | 0.604 | 0.132 | 0.591 | 1.000 | LogisticRegression | leave_one_session_out | session_20260703_205540,sitting_ab_default_cfg,sitting_ab_static_retention_cfg,sitting_relative_gate_refined_live_test |  |
| 0.688 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | LogisticRegression | older_sessions_to_live_session | sitting_relative_gate_refined_live_test |  |
| 0.792 | 0.396 | 0.926 | 0.604 | 0.074 | 0.591 | 1.000 | RandomForestClassifier | leave_one_session_out | session_20260703_205540,sitting_ab_default_cfg,sitting_ab_static_retention_cfg,sitting_relative_gate_refined_live_test |  |
| 0.688 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | RandomForestClassifier | older_sessions_to_live_session | sitting_relative_gate_refined_live_test |  |
| 0.788 | 0.396 | 0.921 | 0.604 | 0.079 | 0.591 | 1.000 | HistGradientBoostingClassifier | leave_one_session_out | session_20260703_205540,sitting_ab_default_cfg,sitting_ab_static_retention_cfg,sitting_relative_gate_refined_live_test |  |
| 0.688 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | HistGradientBoostingClassifier | older_sessions_to_live_session | sitting_relative_gate_refined_live_test |  |

## Best Candidate

| accuracy | standing_accuracy | sitting_accuracy | false_SITTING_on_STANDING | false_STANDING_on_SITTING | false_SITTING_on_STANDING_3m | upright_sitting_accuracy | model | validation | heldout_groups | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.688 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | LogisticRegression | older_sessions_to_live_session | sitting_relative_gate_refined_live_test |  |

## Feature Importance

| model | validation | feature | importance |
| --- | --- | --- | --- |
| LogisticRegression | older_sessions_to_live_session | z_mean | 18.766 |
| LogisticRegression | older_sessions_to_live_session | relative_gate_passed_rate_if_available | 5.624 |
| LogisticRegression | older_sessions_to_live_session | lie_prob_mean_if_available | 3.081 |
| LogisticRegression | older_sessions_to_live_session | range_m_mean | 3.028 |
| LogisticRegression | older_sessions_to_live_session | range_error_m_mean | 1.323 |
| LogisticRegression | older_sessions_to_live_session | range_m_std | 1.079 |
| LogisticRegression | older_sessions_to_live_session | z_std | 1.030 |
| LogisticRegression | older_sessions_to_live_session | speed_mean | 0.987 |
| LogisticRegression | older_sessions_to_live_session | display_moving_rate | 0.953 |
| LogisticRegression | older_sessions_to_live_session | stand_prob_mean | 0.924 |
| LogisticRegression | older_sessions_to_live_session | fall_prob_mean_if_available | 0.907 |
| LogisticRegression | older_sessions_to_live_session | display_unknown_rate | 0.773 |
| LogisticRegression | older_sessions_to_live_session | display_standing_rate | 0.772 |
| LogisticRegression | older_sessions_to_live_session | LOW_POINTS_rate | 0.764 |
| LogisticRegression | older_sessions_to_live_session | sit_prob_std | 0.717 |
| LogisticRegression | older_sessions_to_live_session | stand_prob_std | 0.453 |
| LogisticRegression | older_sessions_to_live_session | geom_pts_mean | 0.407 |
| LogisticRegression | older_sessions_to_live_session | sit_minus_stand_mean | 0.369 |
| LogisticRegression | older_sessions_to_live_session | geom_pts_ge_3_rate | 0.312 |
| RandomForestClassifier | older_sessions_to_live_session | z_mean | 0.275 |
| LogisticRegression | older_sessions_to_live_session | geom_pts_ge_1_rate | 0.258 |
| LogisticRegression | older_sessions_to_live_session | NO_POINTS_rate | 0.258 |
| LogisticRegression | older_sessions_to_live_session | sit_prob_mean | 0.231 |
| LogisticRegression | older_sessions_to_live_session | distance_m | 0.218 |
| RandomForestClassifier | older_sessions_to_live_session | lie_prob_mean_if_available | 0.201 |
| LogisticRegression | older_sessions_to_live_session | OK_rate | 0.194 |
| RandomForestClassifier | older_sessions_to_live_session | range_error_m_mean | 0.148 |
| LogisticRegression | older_sessions_to_live_session | pose_switch_count | 0.113 |
| RandomForestClassifier | older_sessions_to_live_session | stand_prob_mean | 0.093 |
| LogisticRegression | older_sessions_to_live_session | tracking_presence_rate | 0.093 |

## Limitations

- Validation is grouped by session, but the number of real sessions is small.
- Segment labels come from known protocols, not frame-by-frame human annotation.
- Display pose is used only as a baseline feature/evaluation reference, not as ground truth.