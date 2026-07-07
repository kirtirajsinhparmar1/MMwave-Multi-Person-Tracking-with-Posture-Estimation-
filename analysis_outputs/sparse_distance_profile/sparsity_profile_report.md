# Sparse Distance Profile Report

## Inputs

- Registry: `analysis_inputs\posture_session_registry_full.csv`
- Cleaned root: `analysis_outputs\posture_cleaning`
- Segments analyzed: 126

Optional inputs not found:
- `analysis_outputs/range_sparse_posture_audit`

## Distance Band Definitions

- NEAR: `<= 3m`
- FAR: `> 3m and <= 5m`
- EDGE: `> 5m`

## Band Summary

| distance_band | segment_count | pose_accuracy | standing_accuracy | sitting_accuracy | NO_POINTS_rate | LOW_POINTS_rate | OK_rate | mean_geom_pts | range_error_mean | range_jitter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FAR | 48 | 0.4427 | 0.9282 | 0.205 | 0.7314 | 0.8344 | 0.052 | 0.8806 | 0.5467 | 0.3086 |
| NEAR | 78 | 0.6595 | 0.9001 | 0.5395 | 0.7145 | 0.9203 | 0.0608 | 1.4021 | 0.3108 | 0.1833 |

## Required Questions

1. Severe sparsity/failure begins at: 4m.
2. 4m/5m/6m check: pose_accuracy_4m=0.496, pose_accuracy_5m=0.371, 6m unavailable.
3. Standing vs sitting degradation: standing NEAR=0.902, standing FAR=0.898; sitting NEAR=0.535, sitting FAR=0.185.
4. The most sensitive FAR sitting subtype is `SITTING_UPRIGHT` (sitting_accuracy=0.135, disappearance_rate=0.284).
5. Worst FAR position in this pass is `LEFT` (pose_accuracy=0.373, disappearance_rate=0.355, NO_POINTS_rate=0.688).
6. People-count comparison: 1 person(s): accuracy=0.549, disappearance=0.076, NO_POINTS=0.760; 2 person(s): accuracy=0.573, disappearance=0.285, NO_POINTS=0.669.
7. Sparse indicators most associated with failure: `ui_visible_rate` (corr=0.364), `LOW_POINTS_rate` (corr=0.174), `mean_geom_pts` (corr=-0.130)

## Interpretation Notes

- `range_error_mean` is the mean absolute error between measured target range and protocol distance.
- `range_jitter` is the within-segment standard deviation of target range.
- `point_count_if_available` and `mean_snr_if_available` are populated only for sessions that include `mmwave_associated_points.csv` aligned to cleaned segment frames.
- The current registry mostly predates full associated-point logging, so sparse architecture design should treat these aggregates as a failure profile, not as the final full point-cloud training set.
