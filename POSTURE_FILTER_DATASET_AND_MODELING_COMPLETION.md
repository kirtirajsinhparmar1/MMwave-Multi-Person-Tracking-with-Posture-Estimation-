# Posture Filter Dataset And Modeling Completion

## 1. Training-data audit result

The original ONNX posture model at `model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx` was audited from nearby artifacts. It is a four-class model trained on `STANDING`, `SITTING`, `LYING`, and `FALLING`, with `MOVING` handled outside the model by runtime motion rules.

Recoverable training details:

- Input shape: 176 features.
- Window: 8 frames.
- Per-frame features: 22.
- Split: recording-level split.
- Training windows: 6678.
- Test windows: 1297.
- Reported original test accuracy: 97.764%.
- Reported macro F1: 0.977.

The audit could not recover cfg, live sensor orientation, subject facing direction, training distances, sitting subtype, chair variation, or hand-motion conditions.

## 2. Original model orientation conclusion

The current artifacts do not prove that the original training data matched this live IWR6843ISK-ODS setup. In particular, the original model has only a coarse `SITTING` class and no recoverable evidence for upright, lean-back, or lean-forward sitting coverage.

## 3. Sessions found and registry created

Created `analysis_inputs\posture_session_registry.csv` with four HIGH-trust user-provided sessions:

- `session_20260703_205540`
- `sitting_ab_default_cfg`
- `sitting_ab_static_retention_cfg`
- `sitting_relative_gate_refined_live_test`

Metadata was filled from available session files where recoverable, including cfg path, recording date, and RGB video presence.

## 4. Corrected session protocols used

The corrected protocols were applied exactly from the user notes:

- `session_20260703_205540`: standing 1-4m, then sitting lean-back 1-4m.
- `sitting_ab_default_cfg`: sitting lean-back 1-4m.
- `sitting_ab_static_retention_cfg`: sitting lean-back 1-4m.
- `sitting_relative_gate_refined_live_test`: standing 1-5m, lean-back 1-5m, upright 1-5m, lean-forward 1-5m.

This corrects the earlier A/B assumption that the default and static-retention sessions only covered 2-4m.

## 5. Segment labels created/reused

Created or rebuilt:

- `analysis_inputs\session_20260703_205540_segments.csv`
- `analysis_inputs\sitting_ab_default_segments_corrected_1to4.csv`
- `analysis_inputs\sitting_ab_static_retention_segments_corrected_1to4.csv`
- `analysis_inputs\sitting_relative_gate_live_segments.csv`

The live refined-gate segment file reused the already generated segment times where available.

## 6. Segment-filling confidence

Created `analysis\fill_segment_times_from_protocol.py` and generated:

- `analysis_outputs\posture_session_registry_segment_filling_report.md`

Most segments were filled from range plateaus. Weak or short segments were flagged, especially:

- `session_20260703_205540`: leanback 2m, 3m, and 4m.
- `sitting_ab_default_cfg`: leanback 3m and 4m.
- `sitting_relative_gate_refined_live_test`: several live segments kept earlier low-confidence timing estimates.

## 7. Corrected analyses completed

Corrected analyses were run for all registry sessions under:

- `analysis_outputs\registry_analysis\session_20260703_205540`
- `analysis_outputs\registry_analysis\sitting_ab_default_cfg`
- `analysis_outputs\registry_analysis\sitting_ab_static_retention_cfg`
- `analysis_outputs\registry_analysis\sitting_relative_gate_refined_live_test`

Summary outputs:

- `analysis_outputs\registry_analysis\REGISTRY_ANALYSIS_SUMMARY.md`
- `analysis_outputs\registry_analysis\registry_dataset_summary_metrics.csv`

## 8. Dataset generated

Created `analysis\build_posture_filter_dataset.py`.

Generated:

- `analysis_outputs\posture_filter_dataset\posture_filter_examples.csv`
- `analysis_outputs\posture_filter_dataset\dataset_summary.md`

Dataset size:

- Sessions: 4.
- Segments: 36.
- Examples: 45358.
- Classes: 33911 `SITTING`, 11447 `STANDING`.
- Subposes: 22614 `SITTING_LEAN_BACK`, 7049 `SITTING_UPRIGHT`, 4248 `SITTING_LEAN_FORWARD`, 11447 `STANDING`.

Warnings:

- Class imbalance ratio: 2.96.
- Subpose imbalance ratio: 5.32.

## 9. Offline models trained

Created `analysis\train_posture_filter_model.py`.

Trained and evaluated:

- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

Validation used grouped splits:

- Leave-one-session-out validation.
- Older-sessions-to-live-session validation, with `sitting_relative_gate_refined_live_test` held out.

Generated:

- `analysis_outputs\posture_filter_model\model_comparison.csv`
- `analysis_outputs\posture_filter_model\confusion_matrix.csv`
- `analysis_outputs\posture_filter_model\feature_importance.csv`
- `analysis_outputs\posture_filter_model\POSTURE_FILTER_MODEL_REPORT.md`

## 10. Best model result

No candidate passed the offline acceptance criteria.

The learned models improved sitting recall in grouped validation, but they regressed standing protection too severely. On older-sessions-to-live-session validation, the trained candidates classified all held-out standing examples as sitting:

- Standing accuracy: 0.000.
- False `SITTING` on standing: 1.000.
- Sitting accuracy: 1.000.

This is not acceptable for a runtime posture correction/filter.

The current display baseline remained much safer for standing:

- Baseline standing accuracy: 0.964.
- Baseline false `SITTING` on standing: 0.034.
- Baseline sitting accuracy: 0.358.

## 11. Whether acceptance criteria passed

Acceptance criteria did not pass.

Failed criteria:

- Standing accuracy regressed versus current display.
- False `SITTING` on standing_3m increased instead of decreasing or staying below current display.
- Held-out live-session validation failed for standing protection.

No runtime model was exported, and no runtime integration plan was created.

## 12. Recommended next step

The next step should be more balanced real data collection before another offline filter attempt:

- Add more standing examples at 1-5m under the same cfg families.
- Add sitting subtype repeats for upright, lean-back, and lean-forward at 1-5m.
- Add session-level variety: chair position, body orientation, hands still, hands moving, and different days.
- Keep grouped validation as the required gate.
- Prefer subtype-aware modeling or calibrated abstention rather than one global standing/sitting correction threshold.

## 13. Limitations

- Segment labels are protocol-level labels, not frame-by-frame human annotations.
- Some segment boundaries are inferred from range plateaus and include weak or short intervals.
- Only four real sessions are available, which is small for leave-one-session-out modeling.
- The dataset is imbalanced toward sitting and lean-back examples.
- RGB was not used as posture ground truth.
- Runtime logic, ONNX model, cfg files, and RGB code were intentionally not changed.

## Validation Commands Run

Compilation:

```powershell
rtk python -m py_compile analysis\fill_segment_times_from_protocol.py
rtk python -m py_compile analysis\build_posture_filter_dataset.py
rtk python -m py_compile analysis\train_posture_filter_model.py
rtk python -m py_compile analysis\analyze_distance_posture_session.py
```

Segment filling:

```powershell
rtk python analysis\fill_segment_times_from_protocol.py --registry analysis_inputs\posture_session_registry.csv --out-dir analysis_inputs
```

Corrected registry analyses:

```powershell
rtk python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" --out analysis_outputs\registry_analysis\session_20260703_205540 --expected-distances "1,2,3,4" --manual-segments analysis_inputs\session_20260703_205540_segments.csv --make-plots
rtk python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" --out analysis_outputs\registry_analysis\sitting_ab_default_cfg --expected-distances "1,2,3,4" --manual-segments analysis_inputs\sitting_ab_default_segments_corrected_1to4.csv --make-plots
rtk python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_static_retention_cfg" --out analysis_outputs\registry_analysis\sitting_ab_static_retention_cfg --expected-distances "1,2,3,4" --manual-segments analysis_inputs\sitting_ab_static_retention_segments_corrected_1to4.csv --make-plots
rtk python analysis\analyze_distance_posture_session.py --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test" --out analysis_outputs\registry_analysis\sitting_relative_gate_refined_live_test --expected-distances "1,2,3,4,5" --manual-segments analysis_inputs\sitting_relative_gate_live_segments.csv --make-plots
```

Dataset and model:

```powershell
rtk python analysis\build_posture_filter_dataset.py --registry analysis_inputs\posture_session_registry.csv --analysis-root analysis_outputs\registry_analysis --out analysis_outputs\posture_filter_dataset
rtk python analysis\train_posture_filter_model.py --dataset analysis_outputs\posture_filter_dataset\posture_filter_examples.csv --out analysis_outputs\posture_filter_model
```
