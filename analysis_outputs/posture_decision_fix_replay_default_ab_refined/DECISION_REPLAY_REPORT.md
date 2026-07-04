# Posture Decision Fix Replay

Session: `C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg`

Replay limitation: this is an offline approximation using logged probabilities, logged old display pose, TID, range/track position, and segment labels. It does not rerun the ONNX model or every live smoother state.

## Parameters

| relative_range_min_m | relative_min_prob | relative_margin | relative_frames | standing_veto_prob | standing_veto_margin | moving_guard |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 0.55 | 0.12 | 16 | 0.5 | 0.05 | True |

## Segments

| segment_id | expected_pose | expected_distance_m | start_time_s | end_time_s | duration_s | segmentation_method | confidence | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | SITTING | 2.0 | 106.578 | 176.719 | 70.14099999999999 | auto_range_plateau_trimmed | 1.0 | raw=101.58-181.72s; tol=0.45m; samples=1448 |
| sitting_3m | SITTING | 3.0 | 188.734 | 267.453 | 78.71899999999997 | auto_range_plateau_trimmed | 1.0 | raw=183.73-272.45s; tol=0.45m; samples=1610 |
| sitting_4m | SITTING | 4.0 | 279.453 | 340.75 | 61.297000000000025 | auto_range_plateau_trimmed | 0.7355640000000002 | raw=274.45-345.75s; tol=0.75m; samples=1293 |

## Segment Results

| segment_id | expected_pose | expected_distance_m | start_time_s | end_time_s | duration_s | segmentation_method | confidence | primary_tid | frames | old_accuracy | new_accuracy | old_display_sitting_rate | new_display_sitting_rate | old_display_standing_rate | new_display_standing_rate | old_display_moving_rate | new_display_moving_rate | pose_switch_count_old | pose_switch_count_new | standing_false_sitting_rate_old | standing_false_sitting_rate_new | sitting_false_standing_rate_old | sitting_false_standing_rate_new | mean_stand_prob | mean_sit_prob | mean_sit_minus_stand_margin | relative_gate_pass_rate | moving_translation_confirmed_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | SITTING | 2.0 | 106.578 | 176.719 | 70.14099999999999 | auto_range_plateau_trimmed | 1.0 | 0 | 1275 | 0.807843137254902 | 0.807843137254902 | 0.807843137254902 | 0.807843137254902 | 0.12392156862745098 | 0.12392156862745098 | 0.0596078431372549 | 0.0596078431372549 | 11 | 11 | nan | nan | 0.12392156862745098 | 0.12392156862745098 | 0.346487425882278 | 0.5487728415926297 | 0.20228541571035177 | 0.000784313725490196 | 0.04 | Closest/highest-presence TID selected for segment; replay uses logged probabilities and approximated translation evidence. |
| sitting_3m | SITTING | 3.0 | 188.734 | 267.453 | 78.71899999999997 | auto_range_plateau_trimmed | 1.0 | 0 | 1430 | 0.38741258741258744 | 0.3902097902097902 | 0.38741258741258744 | 0.3902097902097902 | 0.5328671328671328 | 0.5300699300699301 | 0.04195804195804196 | 0.04195804195804196 | 27 | 29 | nan | nan | 0.5328671328671328 | 0.5300699300699301 | 0.3802297447581769 | 0.5149136451089272 | 0.13468390035075037 | 0.04895104895104895 | 0.022377622377622378 | Closest/highest-presence TID selected for segment; replay uses logged probabilities and approximated translation evidence. |
| sitting_4m | SITTING | 4.0 | 279.453 | 340.75 | 61.297000000000025 | auto_range_plateau_trimmed | 0.7355640000000002 | 0 | 853 | 0.8839390386869871 | 0.9320046893317703 | 0.8839390386869871 | 0.9320046893317703 | 0.09378663540445487 | 0.04572098475967175 | 0.011723329425556858 | 0.011723329425556858 | 4 | 5 | nan | nan | 0.09378663540445487 | 0.04572098475967175 | 0.17017868050709278 | 0.5243235516590082 | 0.35414487115191545 | 0.06565064478311841 | 0.0035169988276670576 | Closest/highest-presence TID selected for segment; replay uses logged probabilities and approximated translation evidence. |
