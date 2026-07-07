# RadarPostureNet-v2 Dataset Build Report

Ground truth labels come from user-provided segment protocols and filled segment files. Displayed posture was not used as a target.

Sessions in registry: 9

Person-instances in cleaned segments: 126

Lite windows written: 23272

Full point-cloud tensors possible: no

Lite dataset available: yes

## Windows By Size

- 1.0s: 12769
- 2.0s: 6334
- 3.0s: 4169

## Windows By Coarse Label

- SITTING: 15289
- STANDING: 7983

## Session Manifest

| session_id | person_instances | segments | windows | skipped_no_tid | skipped_empty_windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| session_20260703_205540 | 8 | 8 | 1123 | 0 | 0 |
| sitting_ab_default_cfg | 4 | 4 | 649 | 0 | 0 |
| sitting_ab_static_retention_cfg | 4 | 4 | 1422 | 0 | 0 |
| sitting_relative_gate_refined_live_test | 20 | 20 | 3036 | 0 | 0 |
| session_20260704_145249 | 20 | 10 | 3946 | 1 | 804 |
| session_20260704_150636 | 20 | 10 | 2903 | 3 | 871 |
| session_20260704_152302 | 20 | 20 | 4111 | 0 | 809 |
| session_20260706_173741 | 15 | 15 | 3184 | 0 | 116 |
| session_20260706_175519 | 15 | 15 | 2898 | 0 | 567 |

## Data Modality Decision

No session contained the required per-point xyz/snr-or-doppler rows with point-to-TID association. The bounded pass therefore builds RadarPostureNet-v2-lite only. Full RadarPostureNet-v2 requires logging associated point-cloud rows per frame and TID.
