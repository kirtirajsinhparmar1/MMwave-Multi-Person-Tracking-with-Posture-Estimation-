# Sitting A/B Test Setup

## 1. Files created

- `SITTING_AB_TEST_COMMANDS.md`
- `analysis_inputs\sitting_ab_default_segments.csv`
- `analysis_inputs\sitting_ab_static_retention_segments.csv`
- `analysis\compare_sitting_ab.py`
- `analysis_outputs\sitting_ab_comparison\SITTING_AB_COMPARISON_REPORT.md`
- `SITTING_AB_TEST_SETUP.md`

## 2. Commands created

`SITTING_AB_TEST_COMMANDS.md` contains two full runtime commands:

- Test A: `sitting_ab_default_cfg` using `ODS_6m_default.cfg`
- Test B: `sitting_ab_static_retention_cfg` using `ODS_6m_staticRetention.cfg`

Both commands keep the latest successful runtime posture/RGB/logging flags and differ only by cfg, output folder, and session id. No new posture threshold experiment is added.

## 3. Manual segment templates

Both segment templates contain the same sitting-only protocol with blank start/end times:

```csv
segment_id,expected_pose,expected_distance_m,start_time_s,end_time_s
sitting_2m,SITTING,2.0,,
sitting_3m,SITTING,3.0,,
sitting_4m,SITTING,4.0,,
```

Fill `start_time_s` and `end_time_s` after reviewing the RGB video and range plot for each recorded session.

## 4. Comparison script behavior

Run after both live sessions are analyzed:

```powershell
python analysis\compare_sitting_ab.py `
  --default analysis_outputs\sitting_ab_default_analysis `
  --static analysis_outputs\sitting_ab_static_retention_analysis `
  --out analysis_outputs\sitting_ab_comparison
```

The script reads these CSVs from both analysis folders:

- `posture_verdict_by_segment.csv`
- `stand_sit_probability_by_segment.csv`
- `no_points_effect_by_pose.csv`
- `combined_diagnostics_by_segment.csv`
- `tracking_metrics_by_segment.csv`

It writes:

- `sitting_ab_summary.csv`
- `sitting_ab_probability_comparison.csv`
- `sitting_ab_geometry_comparison.csv`
- `sitting_ab_tracking_comparison.csv`
- `SITTING_AB_COMPARISON_REPORT.md`

## 5. Metrics compared

- posture_accuracy
- display_standing_rate
- display_sitting_rate
- mean_stand_prob
- mean_sit_prob
- stand_minus_sit_margin
- NO_POINTS_rate
- mean_geom_pts
- geom_pts_ge_3_rate, if present in the analysis outputs
- range_mae
- range_jitter
- tracking_presence_rate
- extra_track_rate
- tid_switch_count

## 6. Verdict logic

- `STATIC_RETENTION_HELPED_GEOMETRY_AND_POSTURE`: static retention improves point geometry and sitting accuracy.
- `GEOMETRY_IMPROVED_MODEL_STILL_WRONG`: static retention improves geometry, but sitting accuracy does not improve and stand probability still dominates.
- `STATIC_RETENTION_DID_NOT_IMPROVE_GEOMETRY`: static retention does not improve seated point geometry.
- `GATING_DECISION_REMAINS_PROBLEM`: static retention improves sit probability, but display remains mostly STANDING.
- `STATIC_RETENTION_TRACKING_REGRESSION`: static retention heavily worsens range/tracking quality.

## 7. Validation commands run

```powershell
python -m py_compile analysis\compare_sitting_ab.py
```

## 8. Exact next steps for the physical test

1. Run Test A from `SITTING_AB_TEST_COMMANDS.md`.
2. Sit at 2m for 60 sec, 3m for 60 sec, and 4m for 60 sec.
3. Run Test B from `SITTING_AB_TEST_COMMANDS.md`.
4. Repeat the same 2m, 3m, and 4m sitting protocol.
5. Review each RGB video/range plot and fill the corresponding segment CSV.
6. Analyze the default session:

```powershell
python analysis\analyze_distance_posture_session.py `
  --session "..\logs\sitting_ab_default_cfg" `
  --out analysis_outputs\sitting_ab_default_analysis `
  --expected-distances "2,3,4" `
  --manual-segments analysis_inputs\sitting_ab_default_segments.csv `
  --make-plots
```

7. Analyze the static-retention session:

```powershell
python analysis\analyze_distance_posture_session.py `
  --session "..\logs\sitting_ab_static_retention_cfg" `
  --out analysis_outputs\sitting_ab_static_retention_analysis `
  --expected-distances "2,3,4" `
  --manual-segments analysis_inputs\sitting_ab_static_retention_segments.csv `
  --make-plots
```

8. Run the A/B comparison helper:

```powershell
python analysis\compare_sitting_ab.py `
  --default analysis_outputs\sitting_ab_default_analysis `
  --static analysis_outputs\sitting_ab_static_retention_analysis `
  --out analysis_outputs\sitting_ab_comparison
```

Do not claim A/B results until these live sessions are recorded, manually segmented, analyzed, and compared.
