# RadarPostureNet-v2 Sparse-MoE Training Plan

## 1. Objective

Train a sparsity-aware, distance-conditioned RadarPostureNet-v2 model that keeps near-range posture quality while improving far-range sitting and avoiding forced wrong labels at the edge of usable radar evidence.

The current sparse profile shows:

- NEAR (`<=3m`) posture accuracy: `0.6595`
- FAR (`>3m and <=5m`) posture accuracy: `0.4427`
- NEAR sitting accuracy: `0.5395`
- FAR sitting accuracy: `0.2050`
- NEAR standing accuracy: `0.9001`
- FAR standing accuracy: `0.9282`
- practical degradation starts at 4m in the current cleaned registry

The training plan therefore focuses on far-range sitting/subtype recovery and reliability calibration, not only standing preservation.

## 2. Training Data Requirements

Required:

- full associated-point logs,
- cleaned segment labels,
- session registry,
- old model probabilities,
- track metadata,
- distance and position metadata.

The existing smoke `mmwave_associated_points.csv` verifies tensor construction, but it is not enough to train. The far-range protocol must be collected first.

Minimum target coverage before first full training:

- standing center 3m, 4m, 5m, 6m,
- sitting lean-back center 3m, 4m, 5m, 6m,
- sitting upright center 3m, 4m, 5m, 6m,
- sitting lean-forward center 3m, 4m, 5m, 6m,
- left/right standing at 3m-6m,
- left/right sitting lean-back at 3m-6m,
- two-person left/right standing at 3m-6m,
- two-person left/right sitting lean-back at 3m-6m.

## 3. Training Phases

### Phase 1: Pretrain Shared Encoders

Pretrain shared point, track, sparsity, and temporal encoders on all valid sessions.

Objectives:

- reconstruct or predict masked point-count summaries,
- predict range evidence mode,
- predict sparsity level,
- learn stable target-centered point representations.

Use both real and augmented sparse windows.

### Phase 2: Train Near Expert

Train the near expert on `<=3m` dense windows.

Focus:

- high-quality target-centered posture geometry,
- standing vs sitting separation,
- sitting subtype separation,
- low false `SITTING` on standing.

Freeze or lightly train shared encoders depending on validation stability.

### Phase 3: Train Far Expert

Train the far expert on:

- real `>3m` sparse windows,
- synthetic sparse near windows,
- 5m and 6m edge cases when available.

Focus:

- robustness to intermittent points,
- recognition of sitting subtypes when only partial geometry remains,
- learning when evidence is too weak and should be `UNKNOWN` or `LOW` reliability.

### Phase 4: Train Gating Network And Reliability Head

Train the gate to blend experts from sparsity and range evidence, not from posture label leakage.

Inputs:

- sparsity embedding,
- track-quality summary,
- valid-frame rate,
- point-count statistics,
- range band.

Targets:

- learned expert weighting through posture loss,
- range evidence mode,
- reliability class.

Regularization:

- encourage near expert weight for dense NEAR windows,
- encourage far expert weight for sparse FAR/EDGE windows,
- do not force hard gates near the 3m boundary.

### Phase 5: Fine-Tune End-To-End

Fine-tune the full model:

- point encoder,
- track encoder,
- sparsity encoder,
- near expert,
- far expert,
- gate,
- temporal encoder,
- all heads.

Use grouped validation after every run. Do not choose a checkpoint from random frame accuracy.

## 4. Losses

Primary losses:

- coarse posture classification loss,
- sitting subtype loss,
- reliability/confidence loss,
- range mode loss,
- sparsity level loss.

Regularization losses:

- temporal smoothness regularization,
- gate entropy regularization with weak target priors,
- optional old-ONNX distillation loss with low weight.

Recommended weighting:

```text
coarse posture: 1.00
sitting subtype: 0.50
reliability: 0.35
range mode: 0.20
sparsity level: 0.20
temporal smoothness: 0.05
old-ONNX distillation: 0.05 or less
```

The old ONNX loss must stay low weight because the old model is known to fail on sitting and far sparse evidence.

## 5. Synthetic Sparsity Augmentation

Apply to near dense windows:

- random point dropout: 25%, 50%, 75%,
- drop low-SNR points,
- drop high-SNR points sometimes,
- drop high-z or low-z points sometimes,
- mask random frames,
- simulate NO_POINTS frames,
- limit max points to 1, 2, 4, 8,
- jitter SNR,
- jitter Doppler.

Label policy:

- posture label remains unchanged,
- subtype label remains unchanged,
- reliability target may degrade,
- range mode becomes synthetic sparse mode if the augmentation is strong enough.

This creates paired examples where the same posture appears under dense and sparse evidence.

## 6. Validation Splits

No random frame split.

Use grouped validation:

- leave-one-session-out,
- train `<=3m`, test `>3m`,
- train center, test left/right,
- train single-person, test two-person,
- train one recording day, test another day,
- test 4m, 5m, and 6m separately.

Additional required holdouts:

- hold out one sitting subtype at FAR to test subtype generalization,
- hold out one subject if multiple subjects are collected,
- hold out one chair placement if chair geometry changes.

## 7. Metrics

Report:

- standing accuracy,
- sitting accuracy,
- sitting subtype accuracy,
- false `SITTING` on `STANDING`,
- false `SITTING` on `standing_3m`,
- false `SITTING` on `standing_4m/5m`,
- false `STANDING` on `SITTING`,
- reliability calibration,
- unknown/low-confidence rate,
- accuracy by range band,
- accuracy by distance,
- accuracy by position,
- accuracy by people_count.

Sparse-specific metrics:

- accuracy by `sparsity_level`,
- reliability by `sparsity_level`,
- gate near/far weights by distance,
- gate near/far weights by valid-frame rate,
- accuracy on synthetic sparse paired examples,
- FAR sitting accuracy by subtype.

## 8. Acceptance Criteria

### NEAR `<=3m`

Required:

- standing accuracy `>= 95%`,
- sitting accuracy improves or stays stable,
- false `SITTING` on standing remains low,
- reliability head does not over-mark good NEAR evidence as `LOW`.

### FAR `3-5m`

Required:

- false sitting on standing `<= 5-8%`,
- sitting improves over old model,
- FAR sitting subtype accuracy improves over old model where labels exist,
- reliability head marks ambiguous cases low confidence,
- left/right and two-person cases do not collapse.

### EDGE `>5m`

Required:

- do not force wrong posture,
- if posture evidence is weak, output `UNKNOWN` or `LOW` confidence,
- reliability calibration is more important than raw forced accuracy,
- report 6m separately from 5m.

## 9. Model Selection Rule

Choose the checkpoint by grouped validation, in this priority order:

1. NEAR standing safety.
2. FAR false sitting on standing.
3. FAR sitting accuracy.
4. Reliability calibration.
5. EDGE unknown/low-confidence correctness.
6. Subtype accuracy.

A model that improves average accuracy but increases dangerous standing-to-sitting errors should not be integrated.

## 10. Runtime Integration Boundary

This training plan does not change runtime posture logic. Runtime replacement is blocked until:

- full associated-point data is collected,
- full tensors are built,
- grouped validation passes acceptance,
- shadow-mode runtime comparison is added,
- rollback path is defined.

The old runtime remains unchanged for this task.
