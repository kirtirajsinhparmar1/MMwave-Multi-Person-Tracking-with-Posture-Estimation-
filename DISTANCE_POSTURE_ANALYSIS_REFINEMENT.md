# Distance/Posture Analysis Refinement

## What was wrong/misleading in the first report
The first report ranked tracking segments by relative metric values and could label segments as "tracking failures" even when presence was 100%, dropouts were zero, TID switches were zero, and extra-track rate was zero. That made strong tracking look like a failure and obscured the real engineering issue: sitting posture detection.

## What interpretation logic changed
The refined script separates tracking status, posture status, range/calibration status, and point-density status. Tracking is only called a failure when presence, dropout, extra-track, ID-switch, range-MAE, or tracking-score thresholds are crossed. Otherwise smaller range/jitter deviations are reported as minor range/jitter issues.

## New CSVs added
- `tracking_verdict_by_segment.csv`
- `posture_verdict_by_segment.csv`
- `stand_sit_probability_by_segment.csv`
- `no_points_effect_by_pose.csv`
- `ghost_shadow_verdict_by_segment.csv`

## New plots added
- `plots/stand_vs_sit_probability_by_segment.png`
- `plots/stand_minus_sit_margin_by_segment.png`
- `plots/sitting_segments_stand_sit_prob_timeline.png`

## Updated executive summary rules
The executive summary now reports tracking as strong unless at least one segment crosses a defined failure threshold: tracking presence below 0.95, dropout rate above 0.05, extra-track rate above 0.05, any TID switch, range MAE above 0.30 m, or tracking score below 85. Posture is summarized separately by standing and sitting accuracy.

## Updated ranked recommendation rules
Recommendations are now generated around the highest-value engineering failures first: sitting_4m classified as STANDING, sitting_3m ambiguous/unstable, and near-range sitting partial failures. Duplicate generic point-association recommendations are avoided unless the data supports that as a distinct issue.

## Validation commands run

```powershell
python -m py_compile analysis\analyze_distance_posture_session.py
python analysis\analyze_distance_posture_session.py --log-root "..\logs" --latest --out analysis_outputs\latest_distance_posture_analysis_v2 --make-plots
```

## Exact command to regenerate the report

```powershell
python analysis\analyze_distance_posture_session.py --log-root "..\logs" --latest --out analysis_outputs\latest_distance_posture_analysis_v2 --make-plots
```
