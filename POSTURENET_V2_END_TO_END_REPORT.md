# RadarPostureNet-v2 End-to-End Report

## 1. Executive Summary

One bounded end-to-end pass was completed: old posture assets were preserved, all registered sessions were discovered, protocol segment templates were generated, cleaned segment assignments were produced, data modality was audited, a lite dataset was built, grouped validation was run, and runtime replacement was blocked unless acceptance passed.

## 2. Session Registry

| session_id | people_count | positions | distances_m | notes |
| --- | ---: | --- | --- | --- |
| session_20260703_205540 | 1 | CENTER | 1;2;3;4 | Distance: 1m, 2m, 3m, 4m. People: 1. Position: center/front. Duration approximately 40-60 seconds per segment. Sitting was leaned back. Discovery: metadata says mmwave_log_points=false |
| sitting_ab_default_cfg | 1 | CENTER | 1;2;3;4 | Default cfg. This was just sitting leaned back. Corrected protocol includes 1m. Duration approximately 40-60 seconds per segment. Discovery: metadata says mmwave_log_points=false |
| sitting_ab_static_retention_cfg | 1 | CENTER | 1;2;3;4 | Static-retention cfg. This was just sitting leaned back. Corrected protocol includes 1m. Duration approximately 40-60 seconds per segment. Discovery: metadata says mmwave_log_points=false |
| sitting_relative_gate_refined_live_test | 1 | CENTER | 1;2;3;4;5 | At least 40-45 seconds per segment. Occasional UI disappearance observed. Discovery: metadata says mmwave_log_points=false |
| session_20260704_145249 | 2 | LEFT;RIGHT | 1;2;3;4;5 | Two-person simultaneous posture recording. From each center distance mark, one person was placed 1m left and one person 1m right. Discovery: metadata says mmwave_log_points=false |
| session_20260704_150636 | 2 | LEFT;RIGHT | 1;2;3;4;5 | Two-person simultaneous sitting subtype recording. From each center distance mark, one person was placed 1m left and one person 1m right. Discovery: metadata says mmwave_log_points=false |
| session_20260704_152302 | 1 | CENTER | 1;2;3;4;5 | Single person straight/front to the sensor. Duration approximately 40-60 seconds per segment. Discovery: metadata says mmwave_log_points=false |
| session_20260706_173741 | 1 | CENTER;RIGHT;LEFT | 1;2;3;4;5 | Single-person standing at center/front, right side, and left side. Duration approximately 40-60 seconds per segment. Discovery: metadata says mmwave_log_points=false |
| session_20260706_175519 | 1 | CENTER;RIGHT;LEFT | 1;2;3;4;5 | Single-person sitting lean-back at center/front, right side, and left side. Duration approximately 40-60 seconds per segment. Discovery: metadata says mmwave_log_points=false |

## 3. Data Cleaning Result

Segments labeled: 106. Person-instances labeled: 126. See `analysis_outputs/posture_cleaning/DATA_CLEANING_REPORT.md`.

## 4. Disappearance/Dropout Summary

Disappearance and reliability evidence was retained in `analysis_outputs/posture_cleaning/disappearance_events.csv`; low-confidence segments were not silently discarded.

## 5. Old Architecture Snapshot Result

Old posture code and model artifacts were copied under `old_architecture`; the manifest and summary are in `old_architecture/manifests` and `old_architecture/reports`.

## 6. Data Modality Audit

Full point-cloud architecture possible: no. Lite architecture possible: yes. See `POSTURE_DATA_MODALITY_AUDIT.md` and `analysis_outputs/posturenet_v2_dataset/pointcloud_availability_report.csv`.

## 7. Architecture Design

The final full and lite architectures are specified in `RADAR_POSTURENET_V2_ARCHITECTURE.md`. The old ONNX output is treated as auxiliary input/teacher signal, not ground truth.

## 8. Dataset Build Summary

Lite dataset root: `analysis_outputs\posturenet_v2_dataset`. Lite dataset available: yes.

## 9. Training Result

Best model: RandomForestClassifier. See `analysis_outputs\posturenet_v2_model\POSTURENET_V2_TRAINING_REPORT.md`.

## 10. Validation Result

Validation used leave-one-session-out grouping, with separate metrics by session, distance, position, and subpose. See the CSV outputs in `analysis_outputs/posturenet_v2_model`.

## 11. Whether Model Passed Acceptance

Acceptance passed: no.

## 12. Whether Runtime Was Integrated

Runtime integrated: no.

## 13. UI/UX Changes If Any

No UI runtime changes were made unless acceptance passed; default old behavior remains unchanged.

## 14. What Is Production-Ready

The offline registry, cleaning, dataset, modality audit, architecture specification, grouped validation, and reports are production-usable as analysis artifacts.

## 15. What Is Not Production-Ready

Runtime replacement is not production-ready unless all acceptance criteria pass and a flagged shadow/replace integration is implemented.

## 16. Exact Next Step

Add non-invasive logging of per-frame associated point-cloud rows: frame, TID/track index, x/y/z, snr, doppler, point quality, and association source. Then repeat the same bounded pipeline for full RadarPostureNet-v2.

## 17. Limitations

The lite model depends on old model probabilities, track metadata, quality flags, and display stability features. It cannot learn target-centered point geometry without associated point-cloud logs.

## Final Decision Table

| question | answer | evidence |
| --- | --- | --- |
| Is the data clean enough to train? | yes | cleaned segment files and lite windows were produced |
| Is full point-cloud architecture possible? | no | pointcloud_availability_report.csv |
| Is lite architecture possible? | yes | posturenet_lite_windows.csv |
| Did the model improve sitting? | yes | best=0.8399; old=0.4795 |
| Did it protect standing? | no | standing_accuracy=0.7813 |
| Did it protect standing_3m? | no | false_sitting_on_standing_3m=0.2837 |
| Did it improve upright sitting? | yes | best=0.7747; old=0.4377 |
| Did it improve lean-forward sitting? | yes | best=0.9880; old=0.5809 |
| Did it handle left/right position? | see metrics | metrics_by_position.csv |
| Did it handle two-person sessions? | see metrics | model comparison and metrics_by_session.csv |
| Is 5m reliable? | see metrics | metrics_by_distance.csv reports 5m separately |
| Was runtime integration performed? | no | runtime only changes after acceptance; this pass did not modify UI runtime |
| Is the system production-ready? | no | all acceptance criteria must pass and runtime must be integrated safely |
| What is the next fix? | add associated point-cloud logging | full architecture requires per-frame per-point xyz/snr/doppler with target association |
