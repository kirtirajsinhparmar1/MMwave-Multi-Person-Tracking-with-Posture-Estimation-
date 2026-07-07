# RadarPostureNet-v2 Architecture

## 1. Problem Definition

RadarPostureNet-v2 is a per-tracked-person posture model for IWR6843 people-tracking logs. The required runtime output is a stable coarse posture label, primarily `STANDING` vs `SITTING`, plus confidence and reliability/visibility state. The offline dataset also contains sitting subtype labels:

- `STANDING`
- `SITTING_LEAN_BACK`
- `SITTING_UPRIGHT`
- `SITTING_LEAN_FORWARD`

The model must learn from user-provided segment protocols, not from displayed UI posture. Displayed posture and old ONNX probabilities are allowed as auxiliary input signals because they are runtime-observable, but they are not ground truth.

## 2. Why Tracking Is Not The Problem

The current recordings already contain TIDs, target position, target velocity, target height metadata, and old pose rows. The failure being addressed is not basic target tracking; it is posture interpretation under:

- distance changes from 1m to 5m
- center, left, and right placement
- two-person simultaneous recordings
- upright and lean-forward sitting, which the old runtime often confuses
- UI disappearance, missing pose rows, low point quality, and TID instability

Tracking outputs are therefore treated as inputs and reliability evidence. The posture model must not change tracking behavior.

## 3. Why Absolute Coordinates Should Not Be The Main Posture Signal

Absolute `x/y/z` and range can accidentally encode the protocol instead of posture. For example, if most standing examples are at one location and most sitting examples are at another, a model can pass a random window split while failing on a new side position or range.

The primary posture signal must be target-centered geometry and temporal shape:

- relative point heights and spreads around the tracked target
- temporal height/velocity/quality changes within the same TID
- old ONNX probability stability and margins as auxiliary evidence
- visibility and low-point states as reliability evidence

Range and people count may remain as context features, but grouped validation must prove the model is not relying on absolute placement as a shortcut.

## 4. Coordinate-Invariant Target-Centered Normalization

For the full model, every frame is normalized per assigned TID:

1. Associate each point-cloud row to a TID using `trackIndex`, `target_index`, or equivalent point-to-target association.
2. Subtract the target centroid or track position from each associated point.
3. Express point features as relative coordinates: `dx`, `dy`, `dz`, radial distance from target, relative height rank, SNR, Doppler, and optional point quality.
4. Normalize scale with robust per-window statistics such as target height, vertical spread, median absolute deviation, and range-aware quality factors.
5. Preserve a small set of context values separately: range, speed, people count, cfg family, point count, and visibility state.

For the lite model, current logs do not contain associated point tensors. The fallback uses target metadata, old model probabilities, quality flags, point counts, display stability, and TID stability, while excluding expected distance/position labels from training features.

## 5. Handling Distance 1m-5m

The model is trained across 1m, 2m, 3m, 4m, and 5m segments. Distance is handled in three ways:

- target-centered geometry removes absolute range as the main shape source in the full architecture
- range and quality are retained as context because radar point quality naturally changes with distance
- validation reports accuracy separately by distance, with explicit 5m reporting and a standing_3m false-sitting check

Acceptance requires protecting standing at 3m because that is a known failure case. A model that improves sitting but mislabels standing_3m as sitting is not safe for replacement.

## 6. Handling Center/Left/Right Position

The protocol includes center/front, left side, and right side single-person recordings. The two-person sessions also include simultaneous left and right people.

Left/right assignment is an offline labeling step only. It should use verified lateral ordering within each segment, not a hard-coded coordinate convention. If the lateral ordering is weak or TIDs are unstable, the assignment confidence must be `LOW`.

Validation must report:

- center accuracy
- left accuracy
- right accuracy
- side-vs-center gap

A model that only works at center/front is not acceptable for replacement.

## 7. Handling Two-Person Sessions

Two-person sessions create two person-instances per simultaneous segment:

- `left_person`
- `right_person`

The segment time is shared, but TID assignment and quality metrics are per person. The model should receive people count and TID stability as context and reliability features. It must not merge both people into one target or silently discard one side.

Validation must include a single-person vs two-person breakdown. Replacement is blocked if two-person accuracy collapses relative to single-person performance.

## 8. Handling Disappearance/Render Uncertainty

Disappearance is a model-relevant signal, not a reason to silently discard data. The cleaning and dataset steps keep:

- track missing rate
- pose row missing rate
- TID switches or competing TIDs
- range jumps
- `NO_POINTS`
- `LOW_POINTS`
- render visibility not confirmed
- UI display unknown rate

RadarPostureNet-v2 includes a reliability/visibility output. Runtime should expose that state so a posture label can be shown as confident, degraded, or low visibility instead of falsely precise.

## 9. Coarse Labels Vs Fine Sitting Subtype Labels

The required runtime safety target is coarse posture:

- `STANDING`
- `SITTING`

The subtype target is useful for diagnostics and future UX:

- `SITTING_LEAN_BACK`
- `SITTING_UPRIGHT`
- `SITTING_LEAN_FORWARD`

The full architecture should train both a coarse head and a subtype head. The lite bounded pass evaluates coarse sitting accuracy by subtype. A future accepted model should export subtype predictions only if grouped subtype validation is strong enough.

## 10. How Old Data And New Data Are Combined

The old posture runtime and ONNX model are preserved under `old_architecture`. Their outputs are used in three ways:

- as baselines in model comparison
- as auxiliary runtime-observable input features
- as optional teacher/distillation signals for future training

They are not labels. The ground truth labels come only from the user-provided session registry and segment protocols.

## 11. How Old ONNX Model Is Used As Auxiliary Teacher/Input

The old ONNX model provides useful weak evidence:

- `prob_standing`
- `prob_sitting`
- optional `prob_lying`
- optional `prob_falling`
- probability margins such as `sit_minus_stand`
- temporal stability and switching rates

In the full architecture these probabilities are embedded and fused with point and track features. In the lite architecture they are central input features because raw point tensors are missing. The model must learn when to override the old ONNX output, especially for upright and lean-forward sitting, while protecting standing.

## 12. Full Architecture If Point-Cloud Data Exists

`RadarPostureNet-v2-full`:

```text
per-TID associated point sequence
-> target-centered point normalization
-> PointNet / Point Transformer point encoder
-> track + quality MLP encoder
-> old ONNX probability embedding
-> fusion
-> TCN + Temporal Transformer
-> coarse posture head
-> sitting subtype head
-> reliability/visibility head
```

Frame-level inputs:

- associated point tensor per TID: `dx`, `dy`, `dz`, SNR, Doppler, point quality
- target metadata: range, target height, velocity, acceleration, confidence, associated point count
- old ONNX probabilities and margins
- quality indicators: low/no points, TID switch, pose missing, render visible/unknown

Temporal inputs:

- 1s, 2s, and 3s windows
- overlapping windows for training
- grouped validation by session/segment/person, not random windows

Heads:

- coarse posture: `STANDING`, `SITTING`
- subtype: `STANDING`, `SITTING_LEAN_BACK`, `SITTING_UPRIGHT`, `SITTING_LEAN_FORWARD`
- reliability: `OK`, `DEGRADED`, `LOW_VISIBILITY`

This is the preferred architecture because posture is fundamentally a target-centered geometry and motion problem.

## 13. Lite Architecture If Only Logs/Probabilities Exist

`RadarPostureNet-v2-lite`:

```text
temporal windows of:
old ONNX probabilities
track metadata
range
velocity
geom_pts
NO_POINTS/LOW_POINTS/OK
display stability
render visibility
TID stability
-> TCN/Transformer or strong tabular temporal model
-> coarse posture head
-> subtype head if labels exist
-> reliability/visibility head
```

The bounded implementation trained strong tabular/temporal-lite candidates from window summary features. This is the best feasible path for current logs, but it is inherently limited because it cannot see target-centered point geometry.

Current-lite feature categories:

- old probability means/stds and sitting-vs-standing margins
- range, target z, speed, and point-count summaries
- quality rates for `NO_POINTS`, `LOW_POINTS`, and `OK`
- display stability rates, used as input/baseline evidence only
- tracking/pose presence rates, disappearance rate, TID switch count
- people count and cfg family

Ground-truth labels remain protocol labels.

## 14. Training Strategy

Training is bounded:

- choose full only if associated point-cloud tensors exist
- otherwise train lite
- evaluate old runtime baselines
- train at most three classical model families and bounded neural-lite candidates
- max neural epochs: 80
- early stopping patience: 10
- no unbounded hyperparameter search

The completed pass trained lite candidates because full point-cloud rows were missing.

## 15. Validation Strategy

Main validation must be grouped. The completed pass uses leave-one-session-out validation. Required reports include:

- overall accuracy
- standing accuracy
- sitting accuracy
- upright sitting accuracy
- lean-back sitting accuracy
- lean-forward sitting accuracy
- false `SITTING` on standing
- false `SITTING` on standing_3m
- false `STANDING` on sitting
- accuracy by distance
- accuracy by position
- accuracy by people count/session
- confusion matrix

Random window/frame splits are not acceptable as the main result because adjacent windows from the same segment are highly correlated.

## 16. UI/UX Integration Plan

Runtime integration is allowed only after acceptance passes. Any integration must be behind explicit flags:

```powershell
--pose-v2-enable
--pose-v2-model "<path>"
--pose-v2-mode shadow
--pose-v2-mode replace
--pose-v2-log
--pose-v2-debug
```

Modes:

- `shadow`: run v2 and log predictions, but keep old posture output controlling the UI
- `replace`: use v2 posture output for UI/human model display

Shadow output should include old pose, v2 pose, final pose, confidence, reliability, assigned TID, and reason. Replace mode should be unavailable or clearly unsafe if grouped acceptance is not met.

## 17. Runtime Rollback Plan

Default runtime behavior remains old behavior. Rollback is:

1. remove `--pose-v2-enable`, or
2. switch `--pose-v2-mode replace` to `--pose-v2-mode shadow`, or
3. point `--pose-v2-model` back to a previous accepted candidate.

Because the completed model did not pass acceptance, no runtime files were changed and there is nothing to roll back.

## 18. Acceptance Criteria

A replacement model must pass all criteria:

1. Standing accuracy >= 95%.
2. False `SITTING` on standing_3m <= 5%.
3. Sitting accuracy improves over old runtime.
4. Upright sitting improves over old runtime.
5. Lean-forward sitting improves over old runtime.
6. Left/right position gap is not severe.
7. Two-person accuracy does not collapse.
8. 5m is reported separately.
9. Validation is grouped by session/position/person-count, not random frame split.

Current-pass decision:

- full point-cloud architecture is preferred
- current logs do not contain associated raw point-cloud rows
- lite dataset is available and was trained
- best lite candidate improved sitting but did not protect standing or standing_3m
- runtime replacement is blocked
