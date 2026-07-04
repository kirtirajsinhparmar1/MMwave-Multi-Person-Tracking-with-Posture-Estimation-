# Sitting Relative Gate Live Validation Report

## 1. Executive summary
Standing 1m/2m remained protected, so the refined gate is safe but insufficient; sitting accuracy still depends on subtype and range.

## 2. Session analyzed
| session_path | cfg_path | session_id | rgb_video_present | segment_method | notes |
| --- | --- | --- | --- | --- | --- |
| C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg | sitting_relative_gate_refined_live_test | True | auto range plateau with time-order fallback | Selected folder contains 8 CSV files, mmwave_frames.csv, mmwave_tracks.csv, mmwave_pose.csv, rgb_frames.csv, rgb_tracks.csv, rgb_keypoints.csv, sync_index.csv, rgb_annotated.mp4. |

## 3. Protocol reconstructed
Fixed order: standing 1m-5m, lean-back sitting 1m-5m, upright sitting 1m-5m, lean-forward sitting 1m-5m. Boundaries were inferred from range plateaus and time order because no manual timestamps were provided.

## 4. Segment boundaries
| segment_id | expected_pose | expected_subpose | expected_distance_m | start_time_s | end_time_s | duration_s | confidence | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | STANDING | 1.000 | 25.506 | 78.687 | 53.181 | 0.950 | auto range plateau; raw=21.51-82.69s; tol=0.35m; samples=1113 |
| standing_2m | STANDING | STANDING | 2.000 | 88.768 | 149.030 | 60.262 | 0.950 | auto range plateau; raw=84.77-153.03s; tol=0.35m; samples=1243 |
| standing_3m | STANDING | STANDING | 3.000 | 168.960 | 231.854 | 62.894 | 0.950 | auto range plateau; raw=164.96-235.85s; tol=0.35m; samples=1289 |
| standing_4m | STANDING | STANDING | 4.000 | 241.856 | 268.823 | 26.967 | 0.490 | auto range plateau; raw=237.86-272.82s; tol=0.35m; samples=637 |
| standing_5m | STANDING | STANDING | 5.000 | 314.926 | 377.169 | 62.243 | 0.950 | auto range plateau; raw=310.93-381.17s; tol=0.35m; samples=1278 |
| leanback_1m | SITTING | SITTING_LEAN_BACK | 1.000 | 406.980 | 435.014 | 28.034 | 0.510 | auto range plateau; raw=406.98-435.01s; tol=0.35m; samples=502 |
| leanback_2m | SITTING | SITTING_LEAN_BACK | 2.000 | 471.363 | 497.291 | 25.928 | 0.471 | auto range plateau; raw=467.36-501.29s; tol=0.55m; samples=618 |
| leanback_3m | SITTING | SITTING_LEAN_BACK | 3.000 | 555.618 | 584.079 | 28.461 | 0.517 | auto range plateau; raw=551.62-588.08s; tol=0.35m; samples=664 |
| leanback_4m | SITTING | SITTING_LEAN_BACK | 4.000 | 637.990 | 674.837 | 36.847 | 0.670 | auto range plateau; raw=633.99-678.84s; tol=0.35m; samples=816 |
| leanback_5m | SITTING | SITTING_LEAN_BACK | 5.000 | 712.573 | 746.749 | 34.176 | 0.621 | auto range plateau; raw=708.57-750.75s; tol=0.35m; samples=767 |
| upright_1m | SITTING | SITTING_UPRIGHT | 1.000 | 789.929 | 855.717 | 65.788 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=785.93-859.72s; tol=0.35m; samples=1342 |
| upright_2m | SITTING | SITTING_UPRIGHT | 2.000 | 865.764 | 930.163 | 64.399 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=861.76-934.16s; tol=0.35m; samples=1318 |
| upright_3m | SITTING | SITTING_UPRIGHT | 3.000 | 940.195 | 1004.283 | 64.088 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=936.20-1008.28s; tol=0.35m; samples=1312 |
| upright_4m | SITTING | SITTING_UPRIGHT | 4.000 | 1016.474 | 1042.685 | 26.211 | 0.477 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1012.47-1046.68s; tol=0.35m; samples=623 |
| upright_5m | SITTING | SITTING_UPRIGHT | 5.000 | 1155.993 | 1205.993 | 50.000 | 0.250 | auto time-order fallback; range plateau not reliable; auto time-order fallback; range plateau not reliable |
| leanforward_1m | SITTING | SITTING_LEAN_FORWARD | 1.000 | 1207.996 | 1234.170 | 26.174 | 0.476 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1208.00-1234.17s; tol=0.35m; samples=475 |
| leanforward_2m | SITTING | SITTING_LEAN_FORWARD | 2.000 | 1240.200 | 1278.767 | 38.567 | 0.701 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1236.20-1282.77s; tol=0.35m; samples=847 |
| leanforward_3m | SITTING | SITTING_LEAN_FORWARD | 3.000 | 1318.334 | 1364.520 | 46.186 | 0.840 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1314.33-1368.52s; tol=0.35m; samples=986 |
| leanforward_4m | SITTING | SITTING_LEAN_FORWARD | 4.000 | 1403.470 | 1432.177 | 28.707 | 0.522 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1403.47-1432.18s; tol=0.35m; samples=523 |
| leanforward_5m | SITTING | SITTING_LEAN_FORWARD | 5.000 | 1480.633 | 1504.587 | 23.954 | 0.436 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1476.63-1508.59s; tol=0.35m; samples=582 |

## 5. Tracking/distance performance
| segment_id | expected_subpose | distance_m | tracking_presence_rate | pose_presence_rate | ui_visible_rate | disappearance_rate | num_disappearance_events | longest_disappearance_s | range_mae_m | tid_switch_count | extra_track_rate | tracking_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | 1.000 | 0.909 | 0.909 | 0.898 | 0.102 | 2 | 0.400 | 0.051 | 0 | 0.000 | OK |
| standing_2m | STANDING | 2.000 | 0.910 | 0.910 | 0.901 | 0.099 | 2 | 0.400 | 0.048 | 0 | 0.000 | OK |
| standing_3m | STANDING | 3.000 | 0.909 | 0.909 | 0.888 | 0.112 | 5 | 0.300 | 0.088 | 0 | 0.000 | OK |
| standing_4m | STANDING | 4.000 | 0.911 | 0.911 | 0.909 | 0.091 | 0 | 0.000 | 0.003 | 0 | 0.000 | OK |
| standing_5m | STANDING | 5.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.122 | 0 | 0.000 | OK |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.311 | 0 | 0.000 | OK |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.353 | 0 | 0.000 | OK |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.910 | 0.910 | 0.902 | 0.098 | 1 | 0.300 | 0.227 | 0 | 0.000 | OK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.279 | 0 | 0.000 | OK |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.908 | 0.908 | 0.908 | 0.092 | 0 | 0.000 | 0.061 | 0 | 0.000 | OK |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.908 | 0.908 | 0.902 | 0.098 | 1 | 0.350 | 0.081 | 0 | 0.000 | OK |
| upright_2m | SITTING_UPRIGHT | 2.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.133 | 0 | 0.000 | OK |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.909 | 0.909 | 0.883 | 0.117 | 6 | 0.350 | 0.091 | 0 | 0.000 | OK |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.910 | 0.910 | 0.908 | 0.092 | 0 | 0.000 | 0.091 | 0 | 0.000 | OK |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.909 | 0.896 | 0.852 | 0.148 | 9 | 0.700 | 4.055 | 0 | 0.000 | OK |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.908 | 0.908 | 0.899 | 0.101 | 1 | 0.300 | 0.226 | 0 | 0.000 | OK |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.911 | 0.911 | 0.901 | 0.099 | 1 | 0.350 | 0.240 | 0 | 0.000 | OK |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.908 | 0.908 | 0.903 | 0.097 | 1 | 0.300 | 0.278 | 0 | 0.000 | OK |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.911 | 0.899 | 0.882 | 0.118 | 2 | 0.550 | 0.234 | 0 | 0.000 | OK |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.910 | 0.910 | 0.852 | 0.148 | 1 | 1.550 | 0.170 | 0 | 0.000 | OK |

## 6. UI disappearance/dropout analysis
UI disappearance was observed with mean 5m disappearance 0.120 versus other distances 0.099. The dominant logged reason was RENDER_NOT_CONFIRMED, with NO_POINTS/low geometry rates used as supporting evidence.

Answering the explicit dropout questions: disappearances are listed in `disappearance_events.csv`; compare 5m against other distances in `disappearance_summary.csv`; sitting versus standing and subpose differences are in the subtype and distance tables below. The reason column separates track dropout, render confirmation, TID switch, range jump, and low-geometry hypotheses.

## 7. Overall posture accuracy
Mean posture accuracy across analyzed segments: 0.589.

## 8. Posture accuracy by distance
| distance_m | leanback_accuracy | leanforward_accuracy | upright_accuracy | standing_accuracy | tracking_presence_rate | disappearance_rate | distance_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 1.000 | 0.971 | 0.100 | 0.980 | 0.909 | 0.098 | OK |
| 2.000 | 1.000 | 0.417 | 1.000 | 0.984 | 0.910 | 0.095 | OK |
| 3.000 | 0.861 | 0.031 | 0.165 | 0.827 | 0.909 | 0.106 | OK |
| 4.000 | 0.000 | 0.966 | 0.000 | 1.000 | 0.910 | 0.098 | OK |
| 5.000 | 0.000 | 0.000 | 0.482 | 1.000 | 0.909 | 0.120 | OK |

## 9. Posture accuracy by subtype
| expected_subpose | mean_accuracy | mean_display_sitting_rate | mean_display_standing_rate | mean_disappearance_rate | worst_distance | best_distance | subtype_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SITTING_LEAN_BACK | 0.572 | 0.520 | 0.389 | 0.093 | 4.000 | 1.000 | WEAK |
| SITTING_LEAN_FORWARD | 0.477 | 0.434 | 0.452 | 0.113 | 5.000 | 1.000 | WEAK |
| SITTING_UPRIGHT | 0.349 | 0.318 | 0.574 | 0.109 | 4.000 | 2.000 | WEAK |
| STANDING | 0.958 | 0.037 | 0.872 | 0.099 | 3.000 | 5.000 | OK |

## 10. Standing protection result
Standing protection passed live.

## 11. Lean-back sitting result
| segment_id | expected_subpose | distance_m | posture_accuracy | display_standing_rate | display_sitting_rate | display_moving_rate | display_unknown_rate | mean_stand_prob | mean_sit_prob | mean_sit_minus_stand_margin | posture_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.255 | 0.608 | 0.354 | GOOD |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.461 | 0.453 | -0.008 | GOOD |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.861 | 0.127 | 0.784 | 0.000 | 0.000 | 0.375 | 0.528 | 0.153 | WEAK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | 0.687 | 0.252 | -0.435 | SIT_AS_STAND |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 | 0.629 | 0.302 | -0.327 | SIT_AS_STAND |

## 12. Upright sitting result
| segment_id | expected_subpose | distance_m | posture_accuracy | display_standing_rate | display_sitting_rate | display_moving_rate | display_unknown_rate | mean_stand_prob | mean_sit_prob | mean_sit_minus_stand_margin | posture_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.100 | 0.816 | 0.091 | 0.001 | 0.000 | 0.364 | 0.573 | 0.209 | SIT_AS_STAND |
| upright_2m | SITTING_UPRIGHT | 2.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.406 | 0.508 | 0.103 | GOOD |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.165 | 0.759 | 0.150 | 0.000 | 0.000 | 0.513 | 0.414 | -0.100 | SIT_AS_STAND |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.000 | 0.910 | 0.000 | 0.000 | 0.000 | 0.578 | 0.351 | -0.227 | SIT_AS_STAND |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.482 | 0.383 | 0.438 | 0.075 | 0.013 | 0.311 | 0.571 | 0.259 | FAILED |

## 13. Lean-forward sitting result
| segment_id | expected_subpose | distance_m | posture_accuracy | display_standing_rate | display_sitting_rate | display_moving_rate | display_unknown_rate | mean_stand_prob | mean_sit_prob | mean_sit_minus_stand_margin | posture_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.971 | 0.000 | 0.881 | 0.027 | 0.000 | 0.352 | 0.552 | 0.200 | GOOD |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.417 | 0.530 | 0.380 | 0.000 | 0.000 | 0.436 | 0.481 | 0.045 | SIT_AS_STAND |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.031 | 0.880 | 0.028 | 0.000 | 0.000 | 0.548 | 0.377 | -0.171 | SIT_AS_STAND |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.966 | 0.000 | 0.880 | 0.019 | 0.012 | 0.450 | 0.450 | -0.001 | GOOD |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.000 | 0.852 | 0.000 | 0.058 | 0.000 | 0.665 | 0.268 | -0.397 | SIT_AS_STAND |

## 14. 5m range result
| segment_id | expected_subpose | tracking_presence_rate | ui_visible_rate | disappearance_rate | range_mae_m | posture_accuracy | mean_geom_pts | NO_POINTS_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_5m | STANDING | 0.909 | 0.909 | 0.091 | 0.122 | 1.000 | 0.150 | 0.840 |
| leanback_5m | SITTING_LEAN_BACK | 0.908 | 0.908 | 0.092 | 0.061 | 0.000 | 0.003 | 0.906 |
| upright_5m | SITTING_UPRIGHT | 0.909 | 0.852 | 0.148 | 4.055 | 0.482 | 2.260 | 0.635 |
| leanforward_5m | SITTING_LEAN_FORWARD | 0.910 | 0.852 | 0.148 | 0.170 | 0.000 | 0.060 | 0.896 |

## 15. Refined gate live behavior
| segment_id | expected_subpose | distance_m | relative_gate_trigger_rate | relative_gate_passed_rate | blocked_range_rate | blocked_standing_veto_rate | false_sitting_if_standing | false_standing_if_sitting | moving_false_positive_rate | gate_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | 1.000 | 0.000 | 0.000 | 0.909 | 0.477 | 0.012 | 0.000 | 0.007 | OK |
| standing_2m | STANDING | 2.000 | 0.000 | 0.000 | 0.910 | 0.669 | 0.016 | 0.000 | 0.000 | OK |
| standing_3m | STANDING | 3.000 | 0.152 | 0.068 | 0.000 | 0.679 | 0.173 | 0.000 | 0.000 | UNSAFE_STANDING_FALSE_SIT |
| standing_4m | STANDING | 4.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | OK |
| standing_5m | STANDING | 5.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | OK |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | 0.000 | OK |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 0.000 | 0.000 | 0.909 | 0.829 | 0.000 | 0.000 | 0.000 | OK |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.090 | 0.037 | 0.000 | 0.065 | 0.000 | 0.139 | 0.000 | OK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.000 | 0.000 | 0.000 | 0.908 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.000 | 0.000 | 0.908 | 0.407 | 0.000 | 0.899 | 0.001 | INSUFFICIENT_SIT_AS_STAND |
| upright_2m | SITTING_UPRIGHT | 2.000 | 0.000 | 0.000 | 0.909 | 0.167 | 0.000 | 0.000 | 0.000 | OK |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.119 | 0.041 | 0.000 | 0.677 | 0.000 | 0.835 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.019 | 0.000 | 0.000 | 0.876 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.000 | 0.000 | 0.909 | 0.113 | 0.000 | 0.421 | 0.075 | OK |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.000 | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 | 0.027 | OK |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.000 | 0.000 | 0.911 | 0.541 | 0.000 | 0.583 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.000 | 0.000 | 0.908 | 0.815 | 0.000 | 0.969 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.005 | 0.000 | 0.012 | 0.012 | 0.000 | 0.000 | 0.019 | OK |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.013 | 0.000 | 0.000 | 0.877 | 0.000 | 0.936 | 0.058 | INSUFFICIENT_SIT_AS_STAND |

## 16. RGB data summary
| rgb_frames_rows | rgb_tracks_rows | rgb_keypoints_rows | sync_index_rows | rgb_actions_rows | rgb_annotated_mp4 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 14050 | 10354 | 134602 | 42475 | 0 | present | No RGB posture accuracy claimed unless rgb_actions.csv has meaningful labels. |

## 17. What is proven
- The selected folder was found and analyzed from local logs.
- The relative gate was enabled in metadata and per-frame debug fields.
- UI disappearance/dropout behavior is measurable from pose/display/geometry rows.

## 18. What is not proven
- RGB posture accuracy is not proven because `rgb_actions.csv` does not contain meaningful action labels for this report.
- Segment boundaries remain best-effort without manually entered timestamps or verified video.

## 19. Recommended next fix path
Do not tune one global threshold. The next fix path should separate geometry/track retention from posture subtype handling, then validate with RGB or manually checked video boundaries.

## 20. Appendix: generated files and plots
- `segment_metrics.csv`
- `disappearance_events.csv`
- `disappearance_summary.csv`
- `relative_gate_live_validation.csv`
- `rgb_summary.csv`
- `plots/*.png`

## Required Session Table
| session_path | cfg_path | session_id | rgb_video_present | segment_method | notes |
| --- | --- | --- | --- | --- | --- |
| C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg | sitting_relative_gate_refined_live_test | True | auto range plateau with time-order fallback | Selected folder contains 8 CSV files, mmwave_frames.csv, mmwave_tracks.csv, mmwave_pose.csv, rgb_frames.csv, rgb_tracks.csv, rgb_keypoints.csv, sync_index.csv, rgb_annotated.mp4. |

## Required Segment Table
| segment_id | expected_pose | expected_subpose | expected_distance_m | start_time_s | end_time_s | duration_s | confidence | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | STANDING | 1.000 | 25.506 | 78.687 | 53.181 | 0.950 | auto range plateau; raw=21.51-82.69s; tol=0.35m; samples=1113 |
| standing_2m | STANDING | STANDING | 2.000 | 88.768 | 149.030 | 60.262 | 0.950 | auto range plateau; raw=84.77-153.03s; tol=0.35m; samples=1243 |
| standing_3m | STANDING | STANDING | 3.000 | 168.960 | 231.854 | 62.894 | 0.950 | auto range plateau; raw=164.96-235.85s; tol=0.35m; samples=1289 |
| standing_4m | STANDING | STANDING | 4.000 | 241.856 | 268.823 | 26.967 | 0.490 | auto range plateau; raw=237.86-272.82s; tol=0.35m; samples=637 |
| standing_5m | STANDING | STANDING | 5.000 | 314.926 | 377.169 | 62.243 | 0.950 | auto range plateau; raw=310.93-381.17s; tol=0.35m; samples=1278 |
| leanback_1m | SITTING | SITTING_LEAN_BACK | 1.000 | 406.980 | 435.014 | 28.034 | 0.510 | auto range plateau; raw=406.98-435.01s; tol=0.35m; samples=502 |
| leanback_2m | SITTING | SITTING_LEAN_BACK | 2.000 | 471.363 | 497.291 | 25.928 | 0.471 | auto range plateau; raw=467.36-501.29s; tol=0.55m; samples=618 |
| leanback_3m | SITTING | SITTING_LEAN_BACK | 3.000 | 555.618 | 584.079 | 28.461 | 0.517 | auto range plateau; raw=551.62-588.08s; tol=0.35m; samples=664 |
| leanback_4m | SITTING | SITTING_LEAN_BACK | 4.000 | 637.990 | 674.837 | 36.847 | 0.670 | auto range plateau; raw=633.99-678.84s; tol=0.35m; samples=816 |
| leanback_5m | SITTING | SITTING_LEAN_BACK | 5.000 | 712.573 | 746.749 | 34.176 | 0.621 | auto range plateau; raw=708.57-750.75s; tol=0.35m; samples=767 |
| upright_1m | SITTING | SITTING_UPRIGHT | 1.000 | 789.929 | 855.717 | 65.788 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=785.93-859.72s; tol=0.35m; samples=1342 |
| upright_2m | SITTING | SITTING_UPRIGHT | 2.000 | 865.764 | 930.163 | 64.399 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=861.76-934.16s; tol=0.35m; samples=1318 |
| upright_3m | SITTING | SITTING_UPRIGHT | 3.000 | 940.195 | 1004.283 | 64.088 | 0.950 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=936.20-1008.28s; tol=0.35m; samples=1312 |
| upright_4m | SITTING | SITTING_UPRIGHT | 4.000 | 1016.474 | 1042.685 | 26.211 | 0.477 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1012.47-1046.68s; tol=0.35m; samples=623 |
| upright_5m | SITTING | SITTING_UPRIGHT | 5.000 | 1155.993 | 1205.993 | 50.000 | 0.250 | auto time-order fallback; range plateau not reliable; auto time-order fallback; range plateau not reliable |
| leanforward_1m | SITTING | SITTING_LEAN_FORWARD | 1.000 | 1207.996 | 1234.170 | 26.174 | 0.476 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1208.00-1234.17s; tol=0.35m; samples=475 |
| leanforward_2m | SITTING | SITTING_LEAN_FORWARD | 2.000 | 1240.200 | 1278.767 | 38.567 | 0.701 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1236.20-1282.77s; tol=0.35m; samples=847 |
| leanforward_3m | SITTING | SITTING_LEAN_FORWARD | 3.000 | 1318.334 | 1364.520 | 46.186 | 0.840 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1314.33-1368.52s; tol=0.35m; samples=986 |
| leanforward_4m | SITTING | SITTING_LEAN_FORWARD | 4.000 | 1403.470 | 1432.177 | 28.707 | 0.522 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1403.47-1432.18s; tol=0.35m; samples=523 |
| leanforward_5m | SITTING | SITTING_LEAN_FORWARD | 5.000 | 1480.633 | 1504.587 | 23.954 | 0.436 | auto time-order fallback; range plateau not reliable; auto range plateau; raw=1476.63-1508.59s; tol=0.35m; samples=582 |

## Required Tracking/Disappearance Table
| segment_id | expected_subpose | distance_m | tracking_presence_rate | pose_presence_rate | ui_visible_rate | disappearance_rate | num_disappearance_events | longest_disappearance_s | range_mae_m | tid_switch_count | extra_track_rate | tracking_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | 1.000 | 0.909 | 0.909 | 0.898 | 0.102 | 2 | 0.400 | 0.051 | 0 | 0.000 | OK |
| standing_2m | STANDING | 2.000 | 0.910 | 0.910 | 0.901 | 0.099 | 2 | 0.400 | 0.048 | 0 | 0.000 | OK |
| standing_3m | STANDING | 3.000 | 0.909 | 0.909 | 0.888 | 0.112 | 5 | 0.300 | 0.088 | 0 | 0.000 | OK |
| standing_4m | STANDING | 4.000 | 0.911 | 0.911 | 0.909 | 0.091 | 0 | 0.000 | 0.003 | 0 | 0.000 | OK |
| standing_5m | STANDING | 5.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.122 | 0 | 0.000 | OK |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.311 | 0 | 0.000 | OK |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.353 | 0 | 0.000 | OK |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.910 | 0.910 | 0.902 | 0.098 | 1 | 0.300 | 0.227 | 0 | 0.000 | OK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.279 | 0 | 0.000 | OK |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.908 | 0.908 | 0.908 | 0.092 | 0 | 0.000 | 0.061 | 0 | 0.000 | OK |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.908 | 0.908 | 0.902 | 0.098 | 1 | 0.350 | 0.081 | 0 | 0.000 | OK |
| upright_2m | SITTING_UPRIGHT | 2.000 | 0.909 | 0.909 | 0.909 | 0.091 | 0 | 0.000 | 0.133 | 0 | 0.000 | OK |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.909 | 0.909 | 0.883 | 0.117 | 6 | 0.350 | 0.091 | 0 | 0.000 | OK |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.910 | 0.910 | 0.908 | 0.092 | 0 | 0.000 | 0.091 | 0 | 0.000 | OK |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.909 | 0.896 | 0.852 | 0.148 | 9 | 0.700 | 4.055 | 0 | 0.000 | OK |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.908 | 0.908 | 0.899 | 0.101 | 1 | 0.300 | 0.226 | 0 | 0.000 | OK |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.911 | 0.911 | 0.901 | 0.099 | 1 | 0.350 | 0.240 | 0 | 0.000 | OK |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.908 | 0.908 | 0.903 | 0.097 | 1 | 0.300 | 0.278 | 0 | 0.000 | OK |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.911 | 0.899 | 0.882 | 0.118 | 2 | 0.550 | 0.234 | 0 | 0.000 | OK |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.910 | 0.910 | 0.852 | 0.148 | 1 | 1.550 | 0.170 | 0 | 0.000 | OK |

## Required Posture Table
| segment_id | expected_subpose | distance_m | posture_accuracy | display_standing_rate | display_sitting_rate | display_moving_rate | display_unknown_rate | mean_stand_prob | mean_sit_prob | mean_sit_minus_stand_margin | posture_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | 1.000 | 0.980 | 0.891 | 0.011 | 0.007 | 0.000 | 0.479 | 0.469 | -0.010 | GOOD |
| standing_2m | STANDING | 2.000 | 0.984 | 0.895 | 0.014 | 0.000 | 0.000 | 0.581 | 0.357 | -0.224 | GOOD |
| standing_3m | STANDING | 3.000 | 0.827 | 0.751 | 0.157 | 0.000 | 0.000 | 0.552 | 0.381 | -0.171 | STAND_AS_SIT |
| standing_4m | STANDING | 4.000 | 1.000 | 0.911 | 0.000 | 0.000 | 0.000 | 0.774 | 0.171 | -0.602 | GOOD |
| standing_5m | STANDING | 5.000 | 1.000 | 0.909 | 0.000 | 0.000 | 0.000 | 0.874 | 0.085 | -0.790 | GOOD |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.255 | 0.608 | 0.354 | GOOD |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.461 | 0.453 | -0.008 | GOOD |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.861 | 0.127 | 0.784 | 0.000 | 0.000 | 0.375 | 0.528 | 0.153 | WEAK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | 0.687 | 0.252 | -0.435 | SIT_AS_STAND |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 | 0.629 | 0.302 | -0.327 | SIT_AS_STAND |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.100 | 0.816 | 0.091 | 0.001 | 0.000 | 0.364 | 0.573 | 0.209 | SIT_AS_STAND |
| upright_2m | SITTING_UPRIGHT | 2.000 | 1.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.406 | 0.508 | 0.103 | GOOD |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.165 | 0.759 | 0.150 | 0.000 | 0.000 | 0.513 | 0.414 | -0.100 | SIT_AS_STAND |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.000 | 0.910 | 0.000 | 0.000 | 0.000 | 0.578 | 0.351 | -0.227 | SIT_AS_STAND |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.482 | 0.383 | 0.438 | 0.075 | 0.013 | 0.311 | 0.571 | 0.259 | FAILED |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.971 | 0.000 | 0.881 | 0.027 | 0.000 | 0.352 | 0.552 | 0.200 | GOOD |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.417 | 0.530 | 0.380 | 0.000 | 0.000 | 0.436 | 0.481 | 0.045 | SIT_AS_STAND |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.031 | 0.880 | 0.028 | 0.000 | 0.000 | 0.548 | 0.377 | -0.171 | SIT_AS_STAND |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.966 | 0.000 | 0.880 | 0.019 | 0.012 | 0.450 | 0.450 | -0.001 | GOOD |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.000 | 0.852 | 0.000 | 0.058 | 0.000 | 0.665 | 0.268 | -0.397 | SIT_AS_STAND |

## Required Gate Table
| segment_id | expected_subpose | distance_m | relative_gate_trigger_rate | relative_gate_passed_rate | blocked_range_rate | blocked_standing_veto_rate | false_sitting_if_standing | false_standing_if_sitting | moving_false_positive_rate | gate_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standing_1m | STANDING | 1.000 | 0.000 | 0.000 | 0.909 | 0.477 | 0.012 | 0.000 | 0.007 | OK |
| standing_2m | STANDING | 2.000 | 0.000 | 0.000 | 0.910 | 0.669 | 0.016 | 0.000 | 0.000 | OK |
| standing_3m | STANDING | 3.000 | 0.152 | 0.068 | 0.000 | 0.679 | 0.173 | 0.000 | 0.000 | UNSAFE_STANDING_FALSE_SIT |
| standing_4m | STANDING | 4.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | OK |
| standing_5m | STANDING | 5.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | OK |
| leanback_1m | SITTING_LEAN_BACK | 1.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 | 0.000 | 0.000 | OK |
| leanback_2m | SITTING_LEAN_BACK | 2.000 | 0.000 | 0.000 | 0.909 | 0.829 | 0.000 | 0.000 | 0.000 | OK |
| leanback_3m | SITTING_LEAN_BACK | 3.000 | 0.090 | 0.037 | 0.000 | 0.065 | 0.000 | 0.139 | 0.000 | OK |
| leanback_4m | SITTING_LEAN_BACK | 4.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanback_5m | SITTING_LEAN_BACK | 5.000 | 0.000 | 0.000 | 0.000 | 0.908 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_1m | SITTING_UPRIGHT | 1.000 | 0.000 | 0.000 | 0.908 | 0.407 | 0.000 | 0.899 | 0.001 | INSUFFICIENT_SIT_AS_STAND |
| upright_2m | SITTING_UPRIGHT | 2.000 | 0.000 | 0.000 | 0.909 | 0.167 | 0.000 | 0.000 | 0.000 | OK |
| upright_3m | SITTING_UPRIGHT | 3.000 | 0.119 | 0.041 | 0.000 | 0.677 | 0.000 | 0.835 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_4m | SITTING_UPRIGHT | 4.000 | 0.019 | 0.000 | 0.000 | 0.876 | 0.000 | 1.000 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| upright_5m | SITTING_UPRIGHT | 5.000 | 0.000 | 0.000 | 0.909 | 0.113 | 0.000 | 0.421 | 0.075 | OK |
| leanforward_1m | SITTING_LEAN_FORWARD | 1.000 | 0.000 | 0.000 | 0.908 | 0.000 | 0.000 | 0.000 | 0.027 | OK |
| leanforward_2m | SITTING_LEAN_FORWARD | 2.000 | 0.000 | 0.000 | 0.911 | 0.541 | 0.000 | 0.583 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanforward_3m | SITTING_LEAN_FORWARD | 3.000 | 0.000 | 0.000 | 0.908 | 0.815 | 0.000 | 0.969 | 0.000 | INSUFFICIENT_SIT_AS_STAND |
| leanforward_4m | SITTING_LEAN_FORWARD | 4.000 | 0.005 | 0.000 | 0.012 | 0.012 | 0.000 | 0.000 | 0.019 | OK |
| leanforward_5m | SITTING_LEAN_FORWARD | 5.000 | 0.013 | 0.000 | 0.000 | 0.877 | 0.000 | 0.936 | 0.058 | INSUFFICIENT_SIT_AS_STAND |

## Required Subtype Summary Table
| expected_subpose | mean_accuracy | mean_display_sitting_rate | mean_display_standing_rate | mean_disappearance_rate | worst_distance | best_distance | subtype_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SITTING_LEAN_BACK | 0.572 | 0.520 | 0.389 | 0.093 | 4.000 | 1.000 | WEAK |
| SITTING_LEAN_FORWARD | 0.477 | 0.434 | 0.452 | 0.113 | 5.000 | 1.000 | WEAK |
| SITTING_UPRIGHT | 0.349 | 0.318 | 0.574 | 0.109 | 4.000 | 2.000 | WEAK |
| STANDING | 0.958 | 0.037 | 0.872 | 0.099 | 3.000 | 5.000 | OK |

## Required Distance Summary Table
| distance_m | leanback_accuracy | leanforward_accuracy | upright_accuracy | standing_accuracy | tracking_presence_rate | disappearance_rate | distance_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 1.000 | 0.971 | 0.100 | 0.980 | 0.909 | 0.098 | OK |
| 2.000 | 1.000 | 0.417 | 1.000 | 0.984 | 0.910 | 0.095 | OK |
| 3.000 | 0.861 | 0.031 | 0.165 | 0.827 | 0.909 | 0.106 | OK |
| 4.000 | 0.000 | 0.966 | 0.000 | 1.000 | 0.910 | 0.098 | OK |
| 5.000 | 0.000 | 0.000 | 0.482 | 1.000 | 0.909 | 0.120 | OK |

## Required Final Decision Table
| question | answer | evidence |
| --- | --- | --- |
| Did the refined gate protect standing 1m/2m live? | Yes | standing_1m/standing_2m false SITTING rates in gate table |
| Did posture improve at 4m/5m? | Mixed/limited | far sitting mean accuracy 0.241 |
| Is 3m still weak? | Yes | 3m sitting mean accuracy 0.352 |
| Which sitting subtype performs best? | STANDING | highest mean posture accuracy by subtype |
| Which sitting subtype performs worst? | SITTING_UPRIGHT | lowest mean posture accuracy by subtype |
| Does upright sitting look like STANDING? | Check posture table | upright display STANDING rates |
| Does lean-forward cause MOVING/UNKNOWN? | Check posture table | lean-forward MOVING and UNKNOWN rates |
| Did the person disappear in the UI? | Yes | disappearance_events.csv and disappearance_summary.csv |
| Were disappearances tracking loss or render/pose/display loss? | RENDER_NOT_CONFIRMED | dominant disappearance reason hypothesis |
| Is the refined gate safe to keep enabled by default? | Safe but insufficient | Standing 1m/2m remained protected, so the refined gate is safe but insufficient; sitting accuracy still depends on subtype and range. |
| What should be fixed next? | Geometry/track retention plus subtype handling | Do not tune one global threshold. The next fix path should separate geometry/track retention from posture subtype handling, then validate with RGB or manually checked video boundaries. |
