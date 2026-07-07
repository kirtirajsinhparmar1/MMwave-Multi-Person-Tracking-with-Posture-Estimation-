# RadarPostureNet-v2 Runtime Integration Plan

Runtime replacement was not performed because the v2 candidate did not pass all grouped-validation acceptance criteria.

Required flags for any future integration:

```powershell
--pose-v2-enable
--pose-v2-model "<path>"
--pose-v2-mode shadow
--pose-v2-mode replace
--pose-v2-log
--pose-v2-debug
```

Default runtime posture behavior must remain old behavior. Shadow mode should log old pose, v2 pose, final pose, confidence, reliability, and reason while the UI continues using the old output. Replace mode should be enabled only after acceptance stays green on grouped validation and live smoke tests.

Best offline candidate: RandomForestClassifier

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
