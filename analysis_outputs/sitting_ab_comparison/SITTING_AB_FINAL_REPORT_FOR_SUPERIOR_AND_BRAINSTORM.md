# Sitting A/B Final Report for Superior and Brainstorm

## 1. Executive summary
Static retention may improve posture evidence but introduces tracking/range regression, so it is not deployable without further cfg tuning.

Top next engineering path: **cfg/static-retention tracking regression and point association cleanup before posture tuning**. This report does not modify runtime posture logic, thresholds, cfg files, model files, renderer code, or RGB code.

## 2. Why this A/B test was run
The prior benchmark showed strong tracking and nearly perfect standing posture, but sitting posture failed, especially at 3m and 4m. This A/B test isolates whether TI static-retention cfg improves seated point evidence and sitting posture detection.

## 3. Test protocol
- Test A: default cfg, sitting at 2m for 60 sec, 3m for 60 sec, and 4m for 60 sec.
- Test B: static-retention cfg, sitting at 2m for 60 sec, 3m for 60 sec, and 4m for 60 sec.
- Both sessions used the same runtime pose/RGB/combined logging setup; cfg was the intended experiment variable.

## 4. Sessions analyzed
| test_name | session_path | cfg_path | session_id | manual_or_auto_segments | rgb_video_present | notes |
| --- | --- | --- | --- | --- | --- | --- |
| default_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg | sitting_ab_default_cfg | auto range-plateau suggestions written to manual CSV | True | combined mmWave/RGB log folder selected |
| static_retention_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_static_retention_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg | sitting_ab_static_retention_cfg | auto range-plateau suggestions written to manual CSV | True | combined mmWave/RGB log folder selected |

Discovery CSV: `analysis_outputs/sitting_ab_session_discovery.csv`.

## 5. Segment boundaries used
The original manual segment templates were blank. Segment boundaries below were inferred from range plateaus near 2m, 3m, and 4m, trimmed away from transitions, written back to the manual segment CSVs, and then passed to the analyzer.

| test_name | segment_id | expected_pose | expected_distance_m | start_time_s | end_time_s | duration_s | segmentation_method | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_cfg | sitting_2m | SITTING | 2.0 | 106.58 | 176.72 | 70.14 | auto_range_plateau_trimmed | 1.00 |
| default_cfg | sitting_3m | SITTING | 3.0 | 188.73 | 267.45 | 78.72 | auto_range_plateau_trimmed | 1.00 |
| default_cfg | sitting_4m | SITTING | 4.0 | 279.45 | 340.75 | 61.30 | auto_range_plateau_trimmed | 0.74 |
| static_retention_cfg | sitting_2m | SITTING | 2.0 | 101.64 | 216.83 | 115.19 | auto_range_plateau_trimmed | 1.00 |
| static_retention_cfg | sitting_3m | SITTING | 3.0 | 228.86 | 355.45 | 126.59 | auto_range_plateau_trimmed | 1.00 |
| static_retention_cfg | sitting_4m | SITTING | 4.0 | 367.45 | 441.80 | 74.34 | auto_range_plateau_trimmed | 1.00 |

## 6. Tracking comparison
| segment_id | default_tracking_presence | static_tracking_presence | default_range_mae_m | static_range_mae_m | delta_range_mae_m | default_tid_switches | static_tid_switches | default_extra_track_rate | static_extra_track_rate | tracking_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | 100.0% | 100.0% | 0.053 | 0.064 | 0.011 | 0 | 0 | 0.0% | 0.0% | tracking stable |
| sitting_3m | 100.0% | 100.0% | 0.203 | 0.039 | -0.164 | 0 | 0 | 0.0% | 100.0% | extra-track regression |
| sitting_4m | 100.0% | 100.0% | 0.558 | 0.099 | -0.459 | 0 | 0 | 0.0% | 100.0% | extra-track regression |

## 7. Distance/range accuracy comparison
Range MAE is included in the tracking table. Negative delta_range_mae_m means static retention reduced range error; positive means range error increased.

## 8. Point geometry / NO_POINTS comparison
| segment_id | default_NO_POINTS_rate | static_NO_POINTS_rate | delta_NO_POINTS_rate | default_mean_geom_pts | static_mean_geom_pts | delta_mean_geom_pts | default_geom_pts_ge_3_rate | static_geom_pts_ge_3_rate | delta_geom_pts_ge_3_rate | geometry_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | 88.0% | 70.7% | -17.3% | 0.646 | 2.133 | 1.487 | NA | NA | NA | static improved seated point evidence |
| sitting_3m | 59.3% | 56.0% | -3.3% | 1.806 | 0.978 | -0.829 | NA | NA | NA | static worsened point evidence |
| sitting_4m | 88.4% | 51.5% | -37.0% | 0.261 | 1.451 | 1.190 | NA | NA | NA | static improved seated point evidence |

## 9. Stand-vs-sit probability comparison
| segment_id | default_mean_stand_prob | default_mean_sit_prob | default_margin_stand_minus_sit | static_mean_stand_prob | static_mean_sit_prob | static_margin_stand_minus_sit | probability_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | 0.332 | 0.567 | -0.235 | 0.472 | 0.262 | 0.210 | static still model-favors STANDING |
| sitting_3m | 0.369 | 0.544 | -0.175 | 0.403 | 0.433 | -0.029 | static probabilities are ambiguous |
| sitting_4m | 0.227 | 0.522 | -0.295 | 0.345 | 0.497 | -0.152 | static model-favors SITTING |

## 10. Sitting posture accuracy comparison
| segment_id | default_accuracy | static_accuracy | delta_accuracy | default_display_standing_rate | static_display_standing_rate | delta_display_standing_rate | default_display_sitting_rate | static_display_sitting_rate | delta_display_sitting_rate | posture_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_2m | 92.4% | 4.0% | -88.4% | 7.6% | 68.1% | 60.5% | 92.4% | 4.0% | -88.4% | static worsened sitting display accuracy |
| sitting_3m | 49.1% | 0.3% | -48.8% | 48.4% | 92.6% | 44.1% | 49.1% | 0.3% | -48.8% | static worsened sitting display accuracy |
| sitting_4m | 59.5% | 0.0% | -59.5% | 38.8% | 64.9% | 26.0% | 59.5% | 0.0% | -59.5% | static worsened sitting display accuracy |

## 11. RGB data summary
| session | rgb_frames_rows | rgb_tracks_rows | rgb_keypoints_rows | sync_index_rows | rgb_actions_rows | rgb_annotated_mp4 |
| --- | --- | --- | --- | --- | --- | --- |
| sitting_ab_default_cfg | 2263 | 2292 | 29796 | 8825 | 0 | present |
| sitting_ab_static_retention_cfg | 3245 | 3249 | 42237 | 11493 | 0 | present |

RGB was recorded as visual/synchronization reference. Do not claim RGB posture accuracy unless `rgb_actions.csv` contains meaningful action labels. If `rgb_actions.csv` is empty or contains only headers/default entries, RGB action classification was not available as quantitative ground truth.

## 12. Per-distance analysis: 2m
`sitting_2m`: accuracy 92.4% -> 4.0%; display SITTING 92.4% -> 4.0%; stand_prob 0.332 -> 0.472; sit_prob 0.567 -> 0.262; NO_POINTS 88.0% -> 70.7%; mean_geom_pts 0.646 -> 2.133; range MAE 0.053m -> 0.064m; verdict `GEOMETRY_IMPROVED_MODEL_STILL_WRONG`.

## 13. Per-distance analysis: 3m
`sitting_3m`: accuracy 49.1% -> 0.3%; display SITTING 49.1% -> 0.3%; stand_prob 0.369 -> 0.403; sit_prob 0.544 -> 0.433; NO_POINTS 59.3% -> 56.0%; mean_geom_pts 1.806 -> 0.978; range MAE 0.203m -> 0.039m; verdict `STATIC_RETENTION_TRACKING_REGRESSION`.

## 14. Per-distance analysis: 4m
`sitting_4m`: accuracy 59.5% -> 0.0%; display SITTING 59.5% -> 0.0%; stand_prob 0.227 -> 0.345; sit_prob 0.522 -> 0.497; NO_POINTS 88.4% -> 51.5%; mean_geom_pts 0.261 -> 1.451; range MAE 0.558m -> 0.099m; verdict `STATIC_RETENTION_TRACKING_REGRESSION`.

## 15. Final verdict
| question | answer | evidence |
| --- | --- | --- |
| Did static retention improve seated point geometry? | Mixed by distance | mean_geom_pts delta: min=-0.829, max=1.487, mean=0.616 |
| Did static retention reduce NO_POINTS? | Yes | NO_POINTS delta: min=-0.370, max=-0.033, mean=-0.192 |
| Did static retention increase sit_prob? | No; it worsened | sit_prob delta: min=-0.305, max=-0.025, mean=-0.147 |
| Did static retention reduce stand_prob during sitting? | No; it worsened | stand_prob delta: min=0.034, max=0.140, mean=0.097 |
| Did static retention improve sitting posture accuracy? | No; it worsened | accuracy delta: min=-0.884, max=-0.488, mean=-0.655 |
| Did static retention hurt tracking/range? | Yes | range MAE delta: min=-0.459, max=0.011, mean=-0.204; comparison verdict flagged tracking regression where extra tracks increased |
| Is sitting_4m still model-favoring STANDING? | No | static stand_prob=0.345, sit_prob=0.497, margin=-0.152 |
| Is sitting_3m still a gating/display issue? | Yes | static sit_prob=0.433, stand_prob=0.403, display STANDING=92.6%, display SITTING=0.3% |
| What should be fixed next? | cfg/static-retention tracking regression and point association cleanup before posture tuning | Static retention may improve posture evidence but introduces tracking/range regression, so it is not deployable without further cfg tuning. |

## 16. What the result means technically
Static retention may improve posture evidence but introduces tracking/range regression, so it is not deployable without further cfg tuning.

If sit_prob is higher but display remains STANDING, the model has enough sitting probability, but display/gating/hysteresis is preventing correct posture output. If stand_prob remains higher during sitting, the posture model/features are not separating sitting from standing under these conditions.

## 17. What is proven
- The two recorded sessions were found and analyzed from combined mmWave/RGB logs.
- Sitting-only segment boundaries were generated from range plateaus and used consistently for default and static-retention analyses.
- Tracking, range, geometry, probability, and displayed posture outputs were compared per distance.

## 18. What is not proven
- This does not prove RGB posture accuracy because RGB is used as visual/synchronization reference unless action labels are meaningful.
- This does not prove a deployable runtime fix because no runtime logic, thresholds, cfg contents, or model files were changed.
- This does not prove that random threshold tuning would help.

## 19. Recommended next engineering path
Recommended next path: **cfg/static-retention tracking regression and point association cleanup before posture tuning**.

Do not apply random threshold tuning. Use the A/B evidence above to choose one controlled offline replay or data-collection experiment.

## 20. Brainstorming section: possible fixes ranked by evidence
### Engineering brainstorm based on A/B result
| rank | fix_path | why_it_may_help | evidence_supports | evidence_against | next_test | instability_risk |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | cfg/static retention / fine motion tuning | Could preserve seated static/fine-motion point evidence that target-only features miss. | mean_geom_pts delta: min=-0.829, max=1.487, mean=0.616; NO_POINTS delta: min=-0.370, max=-0.033, mean=-0.192 | If geometry and NO_POINTS did not improve consistently, this cfg alone is not the fix. | Run a second static-retention cfg variant only if tracking/range remains stable and geometry improves. | Medium: static retention can alter point/range behavior and may hurt tracking if over-tuned. |
| 2 | point association radius / target-index association improvement | Sitting failure is strongly tied to missing or sparse point geometry. | NO_POINTS and mean_geom_pts deltas show whether the cfg supplied evidence but association still failed. | If model probabilities remain wrong even when geometry improves, association alone is insufficient. | Replay logs with diagnostic-only association variants and compare geom_pts without changing display behavior. | Medium: wider association can attach wrong points in multi-person scenes. |
| 3 | sensor mount calibration verification | Height and range geometry affect sitting-vs-standing features, especially farther out. | range MAE delta: min=-0.459, max=0.011, mean=-0.204 | Strong tracking and low range error would make calibration less likely as the primary cause. | Record calibration target/person at known distances and compare target z/range against expected mount geometry. | Low to medium: bad calibration changes can shift all posture features. |
| 4 | sitting-specific geometry feature engineering | Can add discriminative seated geometry when current 22-feature slots are sparse or zero-filled. | Use this if stand_prob remains high during sitting despite available geometry. | Will not help if the core issue is display gating after sit_prob already dominates. | Offline feature ablation on sitting 2m/3m/4m logs using existing probabilities and geometry fields. | Medium: new features can degrade standing unless validated against standing sessions. |
| 5 | stand-vs-sit decision/gating update | Needed when sit_prob is higher but displayed pose remains STANDING. | Priority rises if sitting_3m shows sit_prob > stand_prob while display remains STANDING. | Should not be used when stand_prob still dominates; that is a model/feature problem. | Offline replay of decision logic only, measuring display lag and false sitting on standing data. | Medium to high: random threshold tuning could destabilize standing, so use evidence-based replay only. |
| 6 | model retraining with sitting 2m/3m/4m data | Required if the model probabilities themselves favor STANDING under seated conditions. | Strongly supported when 4m sitting still has positive stand-minus-sit margin. | Retraining is premature if missing geometry or gating is the primary failure. | Build a labeled sitting-at-distance dataset and compare cross-distance stand/sit probability margins. | High: retraining can regress standing unless the dataset is balanced and held-out. |
| 7 | RGB-assisted ground truth/fusion | RGB can validate segment timing and later provide cross-modal posture evidence. | RGB frames/tracks/keypoints are present as visual reference. | Do not claim RGB posture accuracy unless rgb_actions.csv has meaningful labels. | Use RGB only to verify segment labels first, then evaluate fusion separately. | Medium: sensor sync/visibility issues can create false confidence. |

## 21. Appendix: generated files and plots
Generated files:
- `analysis_outputs/sitting_ab_session_discovery.csv`
- `analysis_inputs/sitting_ab_default_segments.csv`
- `analysis_inputs/sitting_ab_static_retention_segments.csv`
- `analysis_outputs/sitting_ab_default_analysis/`
- `analysis_outputs/sitting_ab_static_retention_analysis/`
- `analysis_outputs/sitting_ab_comparison/sitting_ab_summary.csv`
- `analysis_outputs/sitting_ab_comparison/sitting_ab_probability_comparison.csv`
- `analysis_outputs/sitting_ab_comparison/sitting_ab_geometry_comparison.csv`
- `analysis_outputs/sitting_ab_comparison/sitting_ab_tracking_comparison.csv`
- `analysis_outputs/sitting_ab_comparison/SITTING_AB_COMPARISON_REPORT.md`
- `analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`

Important plots:
### default cfg
- `analysis_outputs\sitting_ab_default_analysis\plots\timeline_range_by_track.png` - Check the distance plateaus and transition trims used for segmentation.
- `analysis_outputs\sitting_ab_default_analysis\plots\timeline_display_pose.png` - Check whether the UI/display output stayed STANDING during sitting.
- `analysis_outputs\sitting_ab_default_analysis\plots\timeline_quality_geom_pts.png` - Check NO_POINTS and associated geometry availability over time.
- `analysis_outputs\sitting_ab_default_analysis\plots\timeline_stand_sit_probs.png` - Check whether model probabilities favored STANDING or SITTING.
- `analysis_outputs\sitting_ab_default_analysis\plots\posture_accuracy_by_distance.png` - Compare sitting accuracy by distance.
- `analysis_outputs\sitting_ab_default_analysis\plots\pose_distribution_by_segment.png` - Check pose confusion distribution in each segment.
- `analysis_outputs\sitting_ab_default_analysis\plots\stand_vs_sit_probability_by_segment.png` - Compare average stand and sit probability by segment.
- `analysis_outputs\sitting_ab_default_analysis\plots\stand_minus_sit_margin_by_segment.png` - Positive margin means STANDING probability exceeded SITTING.
- `analysis_outputs\sitting_ab_default_analysis\plots\sitting_segments_stand_sit_prob_timeline.png` - Inspect stand/sit probability dynamics inside sitting segments.
- `analysis_outputs\sitting_ab_default_analysis\plots\tracking_vs_posture_summary.png` - Separate tracking quality from posture accuracy.
- `analysis_outputs\sitting_ab_default_analysis\plots\failure_mode_heatmap.png` - Locate dominant posture failure modes by distance.

### static-retention cfg
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\timeline_range_by_track.png` - Check the distance plateaus and transition trims used for segmentation.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\timeline_display_pose.png` - Check whether the UI/display output stayed STANDING during sitting.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\timeline_quality_geom_pts.png` - Check NO_POINTS and associated geometry availability over time.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\timeline_stand_sit_probs.png` - Check whether model probabilities favored STANDING or SITTING.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\posture_accuracy_by_distance.png` - Compare sitting accuracy by distance.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\pose_distribution_by_segment.png` - Check pose confusion distribution in each segment.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\stand_vs_sit_probability_by_segment.png` - Compare average stand and sit probability by segment.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\stand_minus_sit_margin_by_segment.png` - Positive margin means STANDING probability exceeded SITTING.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\sitting_segments_stand_sit_prob_timeline.png` - Inspect stand/sit probability dynamics inside sitting segments.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\tracking_vs_posture_summary.png` - Separate tracking quality from posture accuracy.
- `analysis_outputs\sitting_ab_static_retention_analysis\plots\failure_mode_heatmap.png` - Locate dominant posture failure modes by distance.
