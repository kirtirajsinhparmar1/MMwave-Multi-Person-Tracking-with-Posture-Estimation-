# Static-Retention TID Diagnosis Implementation

## 1. Files inspected
- `mmwave_tracks.csv`, `mmwave_pose.csv`, `mmwave_frames.csv`, `sync_index.csv`, `rgb_frames.csv`, `rgb_tracks.csv`, `session_metadata.json` from both A/B sessions.
- `analysis_inputs/sitting_ab_default_segments.csv`.
- `analysis_inputs/sitting_ab_static_retention_segments.csv`.
- Existing A/B analysis folders under `analysis_outputs/`.

## 2. Script created
- `analysis/diagnose_ab_tid_tracks.py`.

## 3. Metrics computed
- Per cfg/segment/TID range, position, presence, geometry, quality, display pose, stand/sit probabilities, probability/display mismatch, and point-total ratios.

## 4. Plots created
- Static 3m/4m TID range, pose, probability, and geom_pts timelines.
- Default-vs-static TID count timeline.

## 5. Final diagnosis report path
- `analysis_outputs\sitting_ab_tid_diagnosis\STATIC_RETENTION_TID_DIAGNOSIS_REPORT.md`.

## 6. Main conclusion
- Fix track validation / point association / primary target selection before posture tuning. Static retention produced persistent extra TIDs during the failing 3m/4m segments, so posture tuning would be premature.

## 7. Validation commands run
- `python -m py_compile analysis\diagnose_ab_tid_tracks.py`.
- `python analysis\diagnose_ab_tid_tracks.py --default-session ... --static-session ... --default-segments ... --static-segments ... --out analysis_outputs\sitting_ab_tid_diagnosis`.

## 8. Any limitations
- Raw point coordinates were not logged, so exact point-to-TID spatial attachment is not reconstructable.
- Renderer confirmed/rendered/suspect rates are `NA` because no renderer state CSV is present.
- This is offline analysis only and does not claim live radar validation.
