# RadarPostureNet-v2 Sparse-MoE Architecture

## 1. Goal

RadarPostureNet-v2 Sparse-MoE is the final far-range architecture for per-TID posture classification on IWR6843 people-tracking output. It is designed around the observed failure mode:

- posture is reliable through roughly 3m,
- tracking may still work after 3m,
- posture evidence becomes sparse and unreliable at 4m, 5m, and eventually 6m,
- standing and sitting subtypes do not degrade the same way under sparse point evidence.

The goal is not a global rule such as "low confidence after 3m." The model must learn the relationship between range, point sparsity, evidence quality, and posture shape.

The sparse-distance profile in `analysis_outputs/sparse_distance_profile` found the current break at 4m in the registered cleaned data. FAR 3-5m posture accuracy was lower than NEAR, and FAR sitting accuracy dropped much more than FAR standing accuracy. The current registry has little or no 6m data, so EDGE behavior must be collected before runtime replacement.

## 2. Inputs

### Associated Point Sequence

```text
shape: T x N x F
```

`T` is the temporal window length:

- 32 frames for fast response,
- 48 frames for balanced response,
- 64 frames for maximum stability.

`N` is the maximum associated points per TID per frame:

- default: 64.

Point features:

- `relative_x_m`
- `relative_y_m`
- `relative_z_m`
- `height_above_ground_m`
- `point_range_m`
- `point_doppler_mps`
- `point_snr`
- `valid_mask`

The point tensor comes from `mmwave_associated_points.csv`, grouped by `session_id`, `tid`, and `frame`.

### Track Features

Track features are frame-level per TID:

- `target_range_m`
- `target_z_m`
- `target_vx_mps`
- `target_vy_mps`
- `target_vz_mps`
- `speed`
- `geom_pts`
- `NO_POINTS`
- `LOW_POINTS`
- `OK`
- `track_age` if available
- `pose_switch_count`
- `ui_visible` if available

### Old Model Features

The old ONNX output is auxiliary input and optional weak teacher signal, not ground truth:

- `old_stand_prob`
- `old_sit_prob`
- `old_move_prob`
- `old_lie_prob`
- `old_fall_prob`

### Sparsity Features

Sparsity and range are explicit inputs:

- `range_band`: `NEAR`, `FAR`, `EDGE`
- `point_count_mean`
- `point_count_std`
- `NO_POINTS_rate_window`
- `LOW_POINTS_rate_window`
- `SNR_mean`
- `SNR_std`
- `valid_frame_rate`

Distance bands:

- `NEAR`: `<= 3m`
- `FAR`: `> 3m and <= 5m`
- `EDGE`: `> 5m`

## 3. Architecture Summary

```text
Associated point sequence
  -> target-centered point normalization
  -> point-cloud encoder
  -> near-range posture expert
  -> far-sparse posture expert

Track metadata sequence
  -> track/quality encoder

Sparsity/range profile features
  -> sparsity/range encoder
  -> gating network

Old ONNX probabilities
  -> probability embedding

Expert outputs + track embedding + sparsity embedding + old-model embedding
  -> temporal encoder
  -> coarse posture head
  -> sitting subtype head
  -> reliability/confidence head
  -> range-evidence-mode head
```

## 4. Modules

### 1. Target-Centered Point Normalization

Each point is expressed relative to the tracked target center:

```text
relative_x_m = point_x_m - target_x_m
relative_y_m = point_y_m - target_y_m
relative_z_m = point_z_m - target_z_m
```

The model also receives height above ground and point range as context. Target centering makes posture shape the primary signal instead of absolute room position.

### 2. Point-Cloud Encoder

Use a PointNet or Point Transformer style frame encoder:

```text
T x N x F -> T x D_point
```

Recommended first implementation:

- shared per-point MLP,
- masked max pooling,
- masked mean pooling,
- optional attention pooling,
- concatenate pooled features.

Later upgrade:

- lightweight Point Transformer blocks with relative point offsets and SNR-aware attention.

### 3. Track/Quality Encoder

Track and quality features pass through an MLP:

```text
T x K -> T x D_track
```

This branch tells the model when the target is stable, moving, low quality, missing points, or producing unreliable geometry.

### 4. Sparsity/Range Encoder

Window-level sparsity features pass through:

- range-band embedding,
- MLP for continuous sparsity values,
- fusion layer.

```text
S -> D_sparse
```

This branch explicitly represents the evidence mode. It lets the network learn that a shape seen with 25 dense near points is not equivalent to the same class with two intermittent far points.

### 5. Near-Range Posture Expert

The near expert is optimized for dense 1-3m evidence:

- stronger reliance on point geometry,
- finer sitting-vs-standing separation,
- subtype cues from vertical spread, top point height, centroid height, and body volume.

It should train mainly on real NEAR windows and dense windows from FAR only when evidence quality is high.

### 6. Far-Sparse Posture Expert

The far expert is optimized for 3-6m sparse evidence:

- lower dependence on exact point counts,
- more dependence on stable temporal cues,
- awareness of NO_POINTS and LOW_POINTS patterns,
- robustness to missing high-z or low-z returns,
- ability to return ambiguous evidence instead of forcing a wrong posture.

It trains on real FAR/EDGE windows plus synthetic sparse versions of dense NEAR windows.

### 7. Gating Network

The gating network uses sparsity/range embedding and track quality:

```text
[D_sparse, D_track_summary] -> softmax([near_weight, far_weight])
```

Output:

- `near_weight`
- `far_weight`

The final per-frame expert embedding is:

```text
expert_embedding = near_weight * near_expert + far_weight * far_expert
```

The gate is not a hard threshold. At 3m with strong point evidence it may still prefer the near expert; at 3m with sparse/unstable evidence it can shift toward the far expert.

### 8. Temporal Encoder

Use a two-stage temporal encoder:

1. TCN for local motion and short-term stability.
2. Transformer encoder for longer context and intermittent dropouts.

```text
T x D_fused -> T x D_temporal -> D_window
```

The TCN catches short pose transitions and missing-frame bursts. The Transformer catches persistent range/sparsity patterns and repeated posture ambiguity.

### 9. Coarse Posture Head

Classes:

- `STANDING`
- `SITTING`
- `MOVING`
- `UNKNOWN`

The coarse head is the primary runtime posture output. `UNKNOWN` is a valid learned output for weak evidence, especially EDGE.

### 10. Sitting Subtype Head

Classes:

- `STANDING`
- `SITTING_LEAN_BACK`
- `SITTING_UPRIGHT`
- `SITTING_LEAN_FORWARD`

This head is trained for diagnostics and future UI use. It also forces the shared representation to distinguish sitting shapes instead of collapsing all sparse sitting into a single weak class.

### 11. Reliability/Confidence Head

Classes:

- `HIGH`
- `MEDIUM`
- `LOW`

This head learns whether the evidence supports the posture label. It is trained from point counts, dropout, SNR, valid-frame rate, stability, and observed correctness.

### 12. Range-Evidence-Mode Head

Classes:

- `NEAR_DENSE`
- `FAR_SPARSE`
- `EDGE_WEAK`

This auxiliary head regularizes the sparsity/range encoder. It makes the network explicitly preserve evidence mode information all the way through training.

## 5. How sparsity is taught to the model

### Real Sparsity Labels

Each training window gets two evidence labels:

```text
range_band = NEAR / FAR / EDGE
sparsity_level = DENSE / MODERATE / SPARSE / EXTREME_SPARSE
```

These are derived from:

- range in meters,
- associated point count,
- `NO_POINTS_rate`,
- `LOW_POINTS_rate`,
- valid-frame rate,
- SNR distribution.

Initial rule for `sparsity_level`:

- `DENSE`: valid-frame rate >= 0.9, mean associated points >= 12, low/no-points rate < 0.1.
- `MODERATE`: valid-frame rate >= 0.75, mean associated points >= 6.
- `SPARSE`: valid-frame rate >= 0.5 or mean associated points >= 2.
- `EXTREME_SPARSE`: below `SPARSE`, frequent disappearance, or repeated NO_POINTS frames.

These thresholds should be recalibrated after real far-range point logs are collected.

### Synthetic Sparsity Augmentation

For near-range examples, generate sparse versions:

- random point dropout: 25%, 50%, 75%,
- drop low-SNR points,
- drop high-SNR points sometimes,
- drop high-z points sometimes,
- drop low-z points sometimes,
- mask random frames,
- simulate NO_POINTS frames,
- limit max points to 1, 2, 4, or 8,
- jitter SNR and Doppler.

The posture label remains the same. The reliability target may be reduced as evidence becomes too weak.

### Teacher-Student Sparse Degradation

Use dense near-range examples as teacher examples:

```text
dense near-range representation -> posture label
synthetic sparse version -> same posture label, lower reliability if evidence is weak
```

The far expert learns how the same posture degrades under sparse evidence. This is the core reason to use synthetic sparsity: it creates controlled paired examples where posture is constant but evidence quality changes.

### Confidence Target

Reliability labels:

```text
HIGH:
  enough points, stable, correct posture evidence

MEDIUM:
  sparse but stable

LOW:
  too sparse, frequent disappearance, high ambiguity
```

`LOW` does not mean the person is absent. It means posture evidence is weak enough that a forced `STANDING` or `SITTING` label is not trustworthy.

## 6. Why This Is Better Than Simpler Alternatives

### Better Than Global Thresholds

Global thresholds such as "after 3m low confidence" throw away useful 4m evidence and cannot distinguish stable sparse sitting from random missing points. Sparse-MoE learns continuous evidence quality and can still classify far windows when the evidence supports it.

### Better Than Only A Second-Stage RandomForest

A RandomForest on summary features can improve old model outputs, but it cannot directly learn target-centered point shape over time. Sparse-MoE uses full associated point sequences, old probabilities, and reliability features in one temporal model.

### Better Than Retraining The Old ONNX Model

The old model was built around fixed 176-feature windows and limited point summaries. Retraining it keeps the same bottleneck. Sparse-MoE uses all associated points up to a cap, keeps dropout information, and separates near-dense from far-sparse behavior.

### Better Than Using Absolute Coordinates

Absolute coordinates can memorize protocol distance, side, or room layout. Target-centered point normalization makes posture geometry the main signal. Range remains as evidence quality context, not as a shortcut label.

## 7. First Implementation Target

The first trainable implementation should use:

- `T=32` and `T=48` experiments,
- `N=64`,
- PointNet-style masked encoder,
- TCN plus one or two Transformer encoder layers,
- two experts with softmax gate,
- four heads: coarse posture, subtype, reliability, range-evidence mode.

Runtime replacement is intentionally out of scope until grouped validation passes the acceptance criteria in the training plan.
