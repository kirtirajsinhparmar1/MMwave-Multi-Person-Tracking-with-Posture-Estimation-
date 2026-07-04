# Next Sitting A/B Test Plan

## Protocol

### Test A: current/original cfg

- sitting at 2m for 60 sec
- sitting at 3m for 60 sec
- sitting at 4m for 60 sec

### Test B: static-retention/fine-motion cfg, only if already available

- sitting at 2m for 60 sec
- sitting at 3m for 60 sec
- sitting at 4m for 60 sec

## Metrics to compare

- posture_accuracy
- display_standing_rate
- display_sitting_rate
- mean_stand_prob
- mean_sit_prob
- stand_minus_sit_margin
- NO_POINTS_rate
- mean_geom_pts
- geom_pts_ge_3_rate
- range_mae_m
- range_jitter
- time_to_stable_sitting

## Decision rules

If static-retention increases geom_pts and sitting accuracy, cfg/static seated point extraction is likely the fix path.

If geom_pts increases but stand_prob still dominates, posture model/features need improvement.

If sit_prob dominates but display remains STANDING, decision/gating logic needs improvement.

If probabilities are close/ambiguous, range-aware sit-vs-stand margin or additional geometry features may be needed later.

No implementation changes are part of this plan.
