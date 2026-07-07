# Posture Failure Map Report

This is an offline analysis of existing lite logs. Protocol segment labels are the ground truth; old displayed posture and raw old ONNX probabilities are measured outputs/features only.

## Overall Metrics

- Windows analyzed: 23272
- Old displayed posture accuracy: 59.54%
- Raw old ONNX probability posture accuracy: 61.74%
- Standing accuracy: 91.68%
- Sitting accuracy: 42.75%
- False SITTING on STANDING: 3.38%
- False SITTING on standing_3m: 3.97%
- False STANDING on SITTING: 48.79%

## Required Questions

1. Which pose/subpose is worst? SITTING_UPRIGHT has the lowest displayed accuracy at 39.63%.
2. Which distance is worst? 5.0m has the lowest displayed accuracy at 39.46%.
3. Which position is worst? CENTER has the lowest displayed accuracy at 54.08%.
4. Does two-person degrade posture? two-person accuracy 68.35% vs one-person 55.86%.
5. Does 3m standing become false sitting? Yes, measured false SITTING on standing_3m is 3.97%.
6. Does upright sitting become standing? Upright sitting is displayed as STANDING on 55.27% of upright windows.
7. Does lean-forward become standing or moving? Lean-forward is displayed as STANDING on 40.96% and MOVING on 4.56% of lean-forward windows.
8. Are failures correlated with disappearance/NO_POINTS/low geom? Wrong windows average disappearance=0.019, NO_POINTS=0.752, LOW_POINTS=0.899, geom_pts=1.80; correct windows average disappearance=0.014, NO_POINTS=0.762, LOW_POINTS=0.960, geom_pts=0.88.
9. Which data should be collected next once point logging is added? Prioritize the lowest-accuracy subpose/distance/position combinations in failure_map_by_distance_subpose_position.csv, with repeated standing_3m, upright sitting, lean-forward sitting, left/right, two-person, and 5m coverage using associated point-cloud logging.

## Output Tables

- failure_map_by_session.csv
- failure_map_by_distance.csv
- failure_map_by_subpose.csv
- failure_map_by_position.csv
- failure_map_by_people_count.csv
- failure_map_by_distance_subpose_position.csv
- standing_false_sitting_cases.csv
- sitting_false_standing_cases.csv
- disappearance_failure_cases.csv
