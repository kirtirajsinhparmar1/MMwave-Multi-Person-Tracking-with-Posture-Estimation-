# Posture Model Training Data Audit

Audited model:

`model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx`

Audited nearby artifacts:

- `model_metadata.json`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `feature_scaler.json`
- `feature_scaler.npz`
- `model_experiments\outputs\dataset_audit\audit_report.md`
- `model_experiments\outputs\prepared_ti_pose_4class_clean_recording_split\dataset_summary.json`

## Answers

1. What classes was the ONNX model trained on?

   `STANDING`, `SITTING`, `LYING`, and `FALLING`. `WALKING` was removed from ML training and live `MOVING` is handled by velocity/motion rules according to `model_metadata.json`.

2. What exact feature vector shape was used?

   `176` input features: an 8-frame window of 22 features per frame. The feature order is channel-major: `posz_f0..posz_f7`, `velx_f0..velx_f7`, through `snr4_f0..snr4_f7`.

3. What source recordings/datasets were used?

   The final metrics point to `outputs\prepared_ti_pose_4class_clean_recording_split`. The source recording IDs in `metrics.json` include TI-style recordings named such as `results_STOOD_*`, `results_SAT_*`, `results_LAY_*`, `results_FALL_*`, and `replay_2025-*`. The audit artifacts identify this as TI Pose/Fall data with synthetic sensor-domain augmentation.

4. What cfg/sensor setup was used for training recordings, if recoverable?

   Not recoverable from current artifacts.

5. What sensor orientation was used, if recoverable?

   Not recoverable from current artifacts.

6. Was the person facing the radar, side-facing, or mixed?

   Not recoverable from current artifacts.

7. What distances were included?

   Not recoverable from current artifacts.

8. Was sitting recorded as upright, lean-back, lean-forward, or unspecified?

   Not recoverable from current artifacts. The source labels indicate `SITTING` / `SAT`, but no subpose labels were found.

9. What chair height/posture variations were included, if any?

   Not recoverable from current artifacts.

10. Were hands still or moving?

   Not recoverable from current artifacts.

11. What train/val/test split was used?

   Recording-level split. `metrics.json` reports `split_mode=recording`, `num_train=6678`, `num_test=1297`, `train_recording_count=35`, and `test_recording_count=8`.

12. What metrics did the model achieve?

   `metrics.json` reports `accuracy_percent=97.76407093292214`, `macro_f1=0.9772607783618852`, and `weighted_f1=0.9776041002427925`. The classification report shows held-out support of 1297 windows with class F1 scores: STANDING 0.96, SITTING 0.96, LYING 1.00, FALLING 0.98.

13. What limitations are visible from the training data?

   The artifacts do not recover cfg, sensor orientation, body orientation, distances, sitting subtype, chair variation, or hand-motion conditions. The model is trained on TI data rather than the current live IWR6843ISK-ODS recording sessions. The metrics notes state that random window splits are optimistic and that live IWR6843 validation is still required. Sitting is a single coarse class, so upright, lean-back, and lean-forward behavior cannot be audited from the original training labels.

## Recoverable Training Summary

| Field | Value |
|---|---|
| model_type | MLP |
| input_size | 176 |
| window_size | 8 |
| per_frame_features | 22 |
| classes | STANDING, SITTING, LYING, FALLING |
| walking_removed | true |
| split_mode | recording |
| num_samples | 7975 |
| num_train | 6678 |
| num_test | 1297 |
| train_recording_count | 35 |
| test_recording_count | 8 |
| final_accuracy_percent | 97.76407093292214 |
| macro_f1 | 0.9772607783618852 |
| weighted_f1 | 0.9776041002427925 |

## Class Counts

| Class | Windows |
|---|---:|
| STANDING | 2736 |
| SITTING | 1242 |
| LYING | 1452 |
| FALLING | 2545 |

## Conclusion

The original trained ONNX model is a strong four-class TI-data model, but the current artifacts do not prove that it saw the same sensor setup, distances, orientations, chair posture variants, or sitting subtypes now being tested live. The missing sitting subtype and setup metadata are directly relevant to the observed live standing/sitting ambiguity.
