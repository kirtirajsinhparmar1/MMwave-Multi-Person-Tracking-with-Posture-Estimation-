# Reliability Model Report

This offline model predicts whether the old/current posture output should be trusted. It does not predict a replacement posture class and no runtime model is exported by this task.

## Best Candidate

- Best model: RandomForestClassifier
- Trust coverage: 7.76%
- Trusted accuracy: 88.71%
- Baseline false SITTING on standing_3m: 3.97%
- Trusted false SITTING on standing_3m: 1.96%
- Correct standing preservation: 14.03%
- Correct sitting preservation: 8.81%
- Wrong rejection rate: 97.67%
- Acceptance passed: no

## Interpretation

The reliability gate did not meet all acceptance criteria. It should remain offline analysis only until point-cloud logging and stronger validation are available.
