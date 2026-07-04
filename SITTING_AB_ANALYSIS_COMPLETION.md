# Sitting A/B Analysis Completion

## 1. Sessions found
| test_name | session_path | cfg_path | session_id | manual_or_auto_segments | rgb_video_present | notes |
| --- | --- | --- | --- | --- | --- | --- |
| default_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg | sitting_ab_default_cfg | auto range-plateau suggestions written to manual CSV | True | combined mmWave/RGB log folder selected |
| static_retention_cfg | C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_static_retention_cfg | C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_staticRetention.cfg | sitting_ab_static_retention_cfg | auto range-plateau suggestions written to manual CSV | True | combined mmWave/RGB log folder selected |

## 2. Analysis commands run
```powershell
python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" --out analysis_outputs\sitting_ab_default_analysis --expected-distances "2,3,4" --manual-segments analysis_inputs\sitting_ab_default_segments.csv --make-plots
python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_static_retention_cfg" --out analysis_outputs\sitting_ab_static_retention_analysis --expected-distances "2,3,4" --manual-segments analysis_inputs\sitting_ab_static_retention_segments.csv --make-plots
```

## 3. Whether manual or auto segments were used
The original manual templates were blank. Auto range-plateau boundaries were generated, written to the manual segment CSVs, and then used by the analyzer.

## 4. Comparison command run
```powershell
python analysis\compare_sitting_ab.py --default analysis_outputs\sitting_ab_default_analysis --static analysis_outputs\sitting_ab_static_retention_analysis --out analysis_outputs\sitting_ab_comparison
```

## 5. Files created
- `analysis_outputs/sitting_ab_session_discovery.csv`
- `analysis_outputs/sitting_ab_default_analysis/`
- `analysis_outputs/sitting_ab_static_retention_analysis/`
- `analysis_outputs/sitting_ab_comparison/`
- `analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`
- `SITTING_AB_ANALYSIS_COMPLETION.md`

## 6. Final report path
`analysis_outputs/sitting_ab_comparison/SITTING_AB_FINAL_REPORT_FOR_SUPERIOR_AND_BRAINSTORM.md`

## 7. Main result
Static retention may improve posture evidence but introduces tracking/range regression, so it is not deployable without further cfg tuning.

## 8. Validation commands run
```powershell
python -m py_compile analysis\analyze_distance_posture_session.py
python -m py_compile analysis\compare_sitting_ab.py
python -m py_compile analysis\generate_sitting_ab_final_report.py
```

## 9. Any warnings or limitations
- Segments were inferred from range plateaus because the provided manual segment CSVs were blank.
- This is an offline analysis/reporting pass only.
- RGB was recorded as reference; RGB posture accuracy is not claimed unless action labels are meaningful.
