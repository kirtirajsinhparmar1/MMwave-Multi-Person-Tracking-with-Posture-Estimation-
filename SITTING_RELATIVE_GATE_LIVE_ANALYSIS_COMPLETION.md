# Sitting Relative Gate Live Analysis Completion

## 1. Selected session path
C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test

## 2. Files discovered
- events.jsonl
- mmwave_frames.csv
- mmwave_pose.csv
- mmwave_tracks.csv
- rgb_actions.csv
- rgb_frames.csv
- rgb_keypoints.csv
- rgb_tracks.csv
- session_metadata.json
- sync_index.csv

## 3. Segment method used
Auto range plateau with time-order fallback. Suggested times were written back to `analysis_inputs/sitting_relative_gate_live_segments.csv`.

## 4. Scripts created/updated
- Updated `analysis/analyze_distance_posture_session.py` for pose-log tracking fallback and arbitrary manual segment IDs.
- Created `analysis/analyze_relative_gate_live_session.py`.

## 5. Main outputs
- C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00\analysis_outputs\sitting_relative_gate_live_subtype_analysis\segment_metrics.csv
- C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00\analysis_outputs\sitting_relative_gate_live_subtype_analysis\disappearance_events.csv
- C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00\analysis_outputs\sitting_relative_gate_live_subtype_analysis\disappearance_summary.csv
- C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00\analysis_outputs\sitting_relative_gate_live_subtype_analysis\relative_gate_live_validation.csv

## 6. Final report path
C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00\analysis_outputs\sitting_relative_gate_live_subtype_analysis\SITTING_RELATIVE_GATE_LIVE_VALIDATION_REPORT.md

## 7. Main result
Standing 1m/2m remained protected, so the refined gate is safe but insufficient; sitting accuracy still depends on subtype and range.

## 8. Disappearance result
UI disappearance was observed with mean 5m disappearance 0.120 versus other distances 0.099. The dominant logged reason was RENDER_NOT_CONFIRMED, with NO_POINTS/low geometry rates used as supporting evidence.

## 9. Validation commands run
- `python -m py_compile analysis\analyze_distance_posture_session.py`
- `python -m py_compile analysis\analyze_relative_gate_live_session.py`
- `python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test" --out analysis_outputs\sitting_relative_gate_live_analysis --expected-distances "1,2,3,4,5" --manual-segments analysis_inputs\sitting_relative_gate_live_segments.csv --make-plots`
- `python analysis\analyze_relative_gate_live_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test" --segments analysis_inputs\sitting_relative_gate_live_segments.csv --base-analysis analysis_outputs\sitting_relative_gate_live_analysis --out analysis_outputs\sitting_relative_gate_live_subtype_analysis`

## 10. Limitations
- RGB CSVs/video were present, but RGB posture accuracy was not claimed because action labels were not meaningful for this report.
- Segment boundaries are best-effort inferred boundaries.
- No runtime thresholds, cfg files, model files, or RGB code were changed.
