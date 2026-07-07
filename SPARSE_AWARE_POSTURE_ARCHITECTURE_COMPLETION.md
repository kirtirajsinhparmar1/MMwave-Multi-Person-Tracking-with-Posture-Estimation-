# Sparse-Aware Posture Architecture Completion

## 1. Summary Of User Observation

The current posture path works well up to about 3m. Beyond 3m, especially at 4m/5m/6m, associated point evidence becomes sparse. Tracking can still remain usable, but posture classification becomes unreliable because the point cloud no longer provides the same dense body-shape evidence seen at near range.

The requested direction is not a runtime rule that simply marks far targets low confidence. The design target is a model architecture that explicitly learns how range-driven sparsity changes the posture evidence, including how standing and sitting subtypes look when only weak far-range evidence remains.

## 2. Sparse-Distance Profile Result

The sparse profile script was created at:

```text
analysis\summarize_distance_sparsity_profile.py
```

It used:

```text
analysis_inputs\posture_session_registry_full.csv
analysis_outputs\posture_cleaning
```

and wrote:

```text
analysis_outputs\sparse_distance_profile\sparsity_by_distance.csv
analysis_outputs\sparse_distance_profile\sparsity_by_band.csv
analysis_outputs\sparse_distance_profile\sparsity_by_subpose.csv
analysis_outputs\sparse_distance_profile\sparsity_profile_report.md
```

The current run analyzed 126 cleaned segments. `analysis_outputs\range_sparse_posture_audit` was not present, so the report states that optional input is missing.

Key findings:

- Severe degradation begins at 4m.
- Pose accuracy is about 0.496 at 4m and 0.371 at 5m. No 6m cleaned segments were available in this registry pass.
- Standing is comparatively stable: NEAR standing accuracy 0.902, FAR standing accuracy 0.898.
- Sitting degrades strongly: NEAR sitting accuracy 0.535, FAR sitting accuracy 0.185.
- The most sensitive FAR sitting subtype is `SITTING_UPRIGHT`, with sitting accuracy 0.135 and disappearance rate 0.284.
- FAR `LEFT` is the weakest lateral position in this pass, with pose accuracy 0.373 and disappearance rate 0.355.
- Two-person segments show higher disappearance pressure than one-person segments, even when aggregate posture accuracy is not worse in this small cleaned set.
- Current registered segments mostly predate full associated-point logging, so `point_count_if_available` and `mean_snr_if_available` are only available where matching full point logs exist.

## 3. Proposed Architecture

The architecture report was created at:

```text
RADAR_POSTURENET_V2_SPARSE_MOE_ARCHITECTURE.md
```

It defines `RadarPostureNet-v2 Sparse-MoE` with:

- Target-centered point normalization.
- Point-cloud encoder.
- Track/quality encoder.
- Sparsity/range encoder.
- Near-range posture expert.
- Far-sparse posture expert.
- Gating network.
- Temporal encoder.
- Coarse posture head.
- Sitting subtype head.
- Reliability/confidence head.
- Range-evidence-mode head.

The model uses per-TID associated point sequences, track metadata, old ONNX probabilities, and explicit sparsity/range features. The Sparse-MoE split lets the near expert specialize in dense 1-3m evidence while the far expert learns sparse 3-6m evidence instead of forcing one global representation to cover both regimes.

## 4. How Sparsity Is Represented

Sparsity is represented as both raw evidence and summary context:

- Point tensor: `[num_windows, T, N, F]`, default `T=32/48/64`, `N=64`.
- Track tensor: range, height, velocity, speed, geometric point count, status flags, track age, UI visibility when available.
- Sparsity tensor: range band, point-count mean/std, `NO_POINTS` rate, `LOW_POINTS` rate, SNR mean/std, valid frame rate.
- Old model tensor: legacy posture probabilities used as weak teacher/context, not as the final decision source.

The planned range bands are:

```text
NEAR: <= 3m
FAR: > 3m and <= 5m
EDGE: > 5m
```

The planned sparsity labels are:

```text
DENSE
MODERATE
SPARSE
EXTREME_SPARSE
```

## 5. How Sparsity Is Taught To The Model

The architecture report includes a dedicated `How sparsity is taught to the model` section.

Training windows receive real sparsity labels from range, associated point count, `NO_POINTS_rate`, `LOW_POINTS_rate`, valid frame rate, and SNR distribution.

Near-range dense examples are also converted into synthetic sparse examples using:

- Random point dropout at 25%, 50%, and 75%.
- Low-SNR point dropout.
- Occasional high-SNR point dropout.
- Occasional high-z or low-z point dropout.
- Random frame masking.
- Simulated `NO_POINTS` frames.
- Max-point limits of 1, 2, 4, and 8.
- SNR and Doppler jitter.

The posture label remains the same when evidence is still sufficient, while the reliability target is lowered when evidence becomes ambiguous. This creates a teacher-student path where dense near-range posture evidence teaches the far-sparse expert how the same posture degrades under weak evidence.

## 6. Dataset Schema

The full dataset schema was created at:

```text
POSTURENET_V2_FULL_DATASET_SCHEMA.md
```

It defines required inputs:

```text
mmwave_associated_points.csv
mmwave_pose.csv
mmwave_tracks.csv
cleaned segment labels
session registry
```

It defines output tensors:

```text
point_tensor: [num_windows, T, N, F]
track_tensor: [num_windows, T, K]
sparsity_tensor: [num_windows, S]
old_model_tensor: [num_windows, T, P]
```

and labels:

```text
coarse_pose
subtype_pose
reliability
range_band
sparsity_level
```

Fallback rules are included for missing associated points, missing track fields, missing old ONNX probabilities, missing SNR, missing labels, and missing UI visibility.

## 7. Training Plan

The training plan was created at:

```text
RADAR_POSTURENET_V2_SPARSE_MOE_TRAINING_PLAN.md
```

It defines:

- Phase 1: pretrain shared encoders on all valid sessions.
- Phase 2: train the near expert on `<=3m` dense windows.
- Phase 3: train the far expert on `>3m` real sparse windows plus synthetic sparse near windows.
- Phase 4: train the gating network and reliability head.
- Phase 5: fine-tune end-to-end with grouped validation.

It also defines losses, validation splits, metrics, and acceptance criteria for NEAR, FAR, and EDGE behavior.

## 8. Whether Point Logs Exist Yet

Full associated-point logs exist for at least one smoke session:

```text
logs\pc_smoke_standing_2m_01\mmwave_associated_points.csv
```

This is enough to verify tensor construction, but it is not enough to train the final far-range sparse-aware model.

## 9. Whether Tensor Preview Was Built

The offline tensor preview script was created at:

```text
analysis\build_sparse_moe_dataset_preview.py
```

It wrote:

```text
analysis_outputs\sparse_moe_dataset_preview\tensor_shape_report.md
analysis_outputs\sparse_moe_dataset_preview\sample_window_manifest.csv
```

Current preview result:

- Associated point logs found: 1.
- Windows constructed: 200.
- Sessions represented: 1.
- TIDs represented: 2.
- `point_tensor`: `[num_windows, 32, 64, 8]`.
- `track_tensor`: `[num_windows, 32, 12]`.
- `sparsity_tensor`: `[num_windows, 10]`.
- `old_model_tensor`: `[num_windows, 32, 5]`.

No model was trained.

## 10. Far-Range Collection Protocol

The far-range protocol was created at:

```text
SPARSE_FAR_RANGE_DATA_COLLECTION_PROTOCOL.md
```

It specifies the requested sessions:

```text
pc_far_standing_center_3to6_01
pc_far_sitting_leanback_center_3to6_01
pc_far_sitting_upright_center_3to6_01
pc_far_sitting_leanforward_center_3to6_01
pc_far_standing_left_3to6_01
pc_far_standing_right_3to6_01
pc_far_sitting_leanback_left_3to6_01
pc_far_sitting_leanback_right_3to6_01
pc_far_two_person_standing_lr_3to6_01
pc_far_two_person_sitting_leanback_lr_3to6_01
```

Each session should collect 3m, 4m, 5m, and 6m, with 60 seconds stable capture per distance and 10 seconds transition between distances. RGB and associated-point logging should be enabled, disappearance should be manually noted, and chair/sensor geometry should stay fixed.

## 11. Next Exact Step

Run the first associated-point far-range smoke session, `pc_far_standing_center_3to6_01`, across 3m/4m/5m/6m. Then run `analysis\build_sparse_moe_dataset_preview.py` and inspect the manifest to confirm that range bands, valid-frame rates, point-count summaries, and TID continuity are correct before collecting the remaining sitting subtype and two-person sessions.
