# RadarPostureNet-v2 Training Report

Timestamp: 2026-07-06T22:59:15+00:00

Main validation: leave-one-session-out grouped validation. No random frame/window split was used as the main result.

Dataset root: `analysis_outputs\posturenet_v2_dataset`

Windows evaluated: 23272

Full point-cloud data available: no

Lite dataset available: yes

Trained model candidates: LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier, MLPClassifier_lite

Best model: RandomForestClassifier

Acceptance passed: no

## Best Metrics

- overall_accuracy: 0.8198
- standing_accuracy: 0.7813
- sitting_accuracy: 0.8399
- upright_sitting_accuracy: 0.7747
- lean_forward_sitting_accuracy: 0.9880
- false_sitting_on_standing_3m: 0.2837
- false_standing_on_sitting: 0.1601

## Acceptance Criteria

| criterion | passed | evidence |
| --- | --- | --- |
| Standing accuracy >= 95% | False | standing_accuracy=0.7813 |
| False SITTING on standing_3m <= 5% | False | false_sitting_on_standing_3m=0.2837 |
| Sitting accuracy improves over old runtime | True | best=0.8399; baseline=0.4795 |
| Upright sitting improves over old runtime | True | best=0.7747; baseline=0.4377 |
| Lean-forward sitting improves over old runtime | True | best=0.9880; baseline=0.5809 |
| Left/right position gap is not severe | True | center=0.8423; side_min=0.7817; side_gap=0.0606 |
| Two-person accuracy does not collapse | True | single_person=0.8233; two_person=0.8114 |
| 5m is reported separately | True | metrics_by_distance.csv contains 5m rows |
| Validation is grouped, not random frame split | True | leave-one-session-out grouped validation used for every trained model |

No runtime model was exported or integrated. The v2 model remains offline-only until grouped validation protects standing, standing_3m, side positions, and two-person sessions simultaneously.
