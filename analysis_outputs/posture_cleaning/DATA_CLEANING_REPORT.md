# Data Cleaning Report

Labels came from the user-provided segment protocols. Displayed posture was never used as ground truth.

Segment times were imported from prior segment files where available; otherwise they were inferred from protocol order and marked with fallback notes.

Sessions processed: 9

Person-instances labeled: 126

Disappearance/reliability events: 466

| session_id | segments | person_instances | assigned | low_quality | mean_tracking_presence | mean_pose_presence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| session_20260703_205540 | 8 | 8 | 8 | 2 | 1.0044 | 1.0044 |
| sitting_ab_default_cfg | 4 | 4 | 4 | 2 | 0.9985 | 0.9985 |
| sitting_ab_static_retention_cfg | 4 | 4 | 4 | 0 | 1.0006 | 1.0006 |
| sitting_relative_gate_refined_live_test | 20 | 20 | 20 | 9 | 1.0003 | 1.0003 |
| session_20260704_145249 | 10 | 20 | 19 | 5 | 0.7851 | 0.7851 |
| session_20260704_150636 | 10 | 20 | 17 | 9 | 0.6464 | 0.6464 |
| session_20260704_152302 | 20 | 20 | 20 | 2 | 0.829 | 0.829 |
| session_20260706_173741 | 15 | 15 | 15 | 0 | 0.9645 | 0.9645 |
| session_20260706_175519 | 15 | 15 | 15 | 0 | 0.8313 | 0.8313 |

Low-confidence segments were retained and marked; disappearance periods were retained as reliability evidence.
