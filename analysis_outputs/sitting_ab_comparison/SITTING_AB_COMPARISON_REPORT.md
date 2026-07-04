# Sitting A/B Comparison Report

## 1. Purpose
Compare the current default cfg against TI static-retention cfg for sitting-only posture behavior.

## 2. Protocol
- Test A: sitting at 2m, 3m, and 4m for 60 seconds each with the default cfg.
- Test B: sitting at 2m, 3m, and 4m for 60 seconds each with the static-retention cfg.

## 3. Sessions compared
- Default analysis folder: `analysis_outputs\sitting_ab_default_analysis`
- Static-retention analysis folder: `analysis_outputs\sitting_ab_static_retention_analysis`

## 4. Tracking comparison
See `sitting_ab_tracking_comparison.csv` for range, jitter, extra-track, presence, and TID-switch deltas.

## 5. Geometry/point evidence comparison
See `sitting_ab_geometry_comparison.csv` for NO_POINTS, mean_geom_pts, and geom_pts_ge_3 deltas.

## 6. Stand-vs-sit probability comparison
See `sitting_ab_probability_comparison.csv` for mean stand/sit probabilities and margins.

## 7. Posture accuracy comparison
See `sitting_ab_summary.csv` for posture accuracy and display-rate deltas.

## 8. Per-distance result: 2m
sitting_2m: accuracy 0.924 -> 0.040, sit_prob 0.567 -> 0.262, stand_prob 0.332 -> 0.472, mean_geom_pts 0.646 -> 2.133, verdict GEOMETRY_IMPROVED_MODEL_STILL_WRONG.

## 9. Per-distance result: 3m
sitting_3m: accuracy 0.491 -> 0.003, sit_prob 0.544 -> 0.433, stand_prob 0.369 -> 0.403, mean_geom_pts 1.806 -> 0.978, verdict STATIC_RETENTION_TRACKING_REGRESSION.

## 10. Per-distance result: 4m
sitting_4m: accuracy 0.595 -> 0.000, sit_prob 0.522 -> 0.497, stand_prob 0.227 -> 0.345, mean_geom_pts 0.261 -> 1.451, verdict STATIC_RETENTION_TRACKING_REGRESSION.

## 11. Final verdict
- Did static retention improve seated point geometry? mixed by distance.
- Did static retention improve sitting posture accuracy? no.
- Did sitting_4m still favor STANDING? no.
- Did sitting_3m still suffer from gating/display mismatch? yes.
- What should we fix next? resolve tracking/range regression before posture-specific changes.

## 12. Recommended next engineering path
Use the final verdict above to choose the next engineering path. Do not change posture thresholds, renderer logic, RGB code, or model behavior based on this report until the live A/B sessions are recorded and analyzed.
