# Posture Data Modality Audit

## 1. Do the logs contain raw pointCloud rows per frame?

no

## 2. Do the logs contain per-point x/y/z/snr/doppler?

no

## 3. Do the logs contain trackIndexes or point-to-TID association?

no

## 4. Do the logs contain associated points per TID?

no

## 5. Do the logs contain only mmwave_pose probabilities and track metadata?

yes

## 6. Do the logs contain old 176-feature vectors?

no

## 7. Can target-centered point-cloud tensors be reconstructed?

no

## 8. Is full RadarPostureNet-v2 trainable from current logs?

no

## 9. Is only RadarPostureNet-v2-lite trainable from current logs?

yes

## 10. What exact logging must be added for full point-cloud training?

Log per-frame per-point x/y/z/doppler/SNR plus target index/TID association, frame number, timestamp, target pose/track rows, and point quality for every TID.

## Evidence

- session_20260703_205540: session_metadata.json has mmwave_log_points=false
- sitting_ab_default_cfg: session_metadata.json has mmwave_log_points=false
- sitting_ab_static_retention_cfg: session_metadata.json has mmwave_log_points=false
- sitting_relative_gate_refined_live_test: session_metadata.json has mmwave_log_points=false
- session_20260704_145249: session_metadata.json has mmwave_log_points=false
- session_20260704_150636: session_metadata.json has mmwave_log_points=false
- session_20260704_152302: session_metadata.json has mmwave_log_points=false
- session_20260706_173741: session_metadata.json has mmwave_log_points=false
- session_20260706_175519: session_metadata.json has mmwave_log_points=false
- No mmWave point-cloud CSV with xyz/snr/doppler and TID/track-index association was found in the discovered session folders.

## Decision

Per-point associated point data is not available. Build the lite dataset and treat full point-cloud training as blocked until logging is added.
