# Interim Existing Data Use Report

## 1. Executive Summary

Existing lite logs are still useful, but not for a posture replacement model. This pass reused the existing registry, cleaned segments, lite dataset, and prior training outputs to build a detailed failure map, audit RGB as a possible teacher, and train an offline conservative posture reliability model.

The result is analysis-only. Runtime behavior was not changed. No tracking, cfg, ONNX model, UI behavior, or replacement posture model was modified or exported.

Main result: the current data can identify where posture fails, but it cannot support a safe replacement model without associated point-cloud logging. RGB is partially usable for review and coarse pose context, but current keypoints lack knees/ankles, so RGB cannot automatically provide high-confidence frame-level posture ground truth. The reliability model improved trusted accuracy only by rejecting most windows, so it did not pass acceptance.

## 2. What Can Be Done With Current Data

Current data can support:

- failure mapping by session, distance, subpose, position, and people count,
- old displayed posture versus protocol-label comparison,
- raw probability posture comparison where old probabilities are available,
- disappearance, NO_POINTS, LOW_POINTS, geom point, and UI visibility correlation analysis,
- RGB sync and keypoint availability audit,
- RGB-assisted manual review candidates,
- offline reliability-model experiments.

All 9 user-provided sessions were included:

- `session_20260703_205540`
- `sitting_ab_default_cfg`
- `sitting_ab_static_retention_cfg`
- `sitting_relative_gate_refined_live_test`
- `session_20260704_145249`
- `session_20260704_150636`
- `session_20260704_152302`
- `session_20260706_173741`
- `session_20260706_175519`

## 3. What Cannot Be Done Without Point-Cloud Logs

Current logs do not contain the per-point, per-TID associated point-cloud tensors needed for the full coordinate-invariant architecture. That blocks:

- full RadarPostureNet-v2 target-centered point-cloud training,
- reliable body-shape learning at 3m-5m,
- robust upright-sitting versus standing separation,
- robust left/right and two-person geometry modeling,
- production-safe replacement of the old runtime posture output.

## 4. Failure Map Summary

Failure-map outputs were written under:

`analysis_outputs\interim_existing_data_use\failure_map`

Key metrics from existing old displayed posture:

- Aggregate standing accuracy by subpose group: 91.68%.
- Aggregate `SITTING_LEAN_BACK` accuracy: 40.76%.
- Aggregate `SITTING_UPRIGHT` accuracy: 39.63%.
- Aggregate `SITTING_LEAN_FORWARD` accuracy: 53.35%.
- False SITTING on standing_3m aggregate: 3.97%.
- 5m overall accuracy: 39.46%.
- 5m sitting accuracy: 11.24%.
- 4m sitting accuracy: 24.30%.
- `sitting_ab_static_retention_cfg` sitting accuracy: 0.91%.
- `session_20260706_175519` sitting lean-back accuracy: 27.33%.

Worst practical failure areas:

- 5m sitting, especially all sitting subtypes.
- 4m sitting, especially upright and lean-back.
- `SITTING_UPRIGHT`, which often maps to STANDING.
- `SITTING_LEAN_BACK` at longer range and side positions.
- Static-retention cfg sitting data in `sitting_ab_static_retention_cfg`.

Failure answers:

- Worst pose/subpose: sitting subtypes, especially `SITTING_UPRIGHT` and `SITTING_LEAN_BACK`.
- Worst distance: 5m overall; 4m is also weak for sitting.
- Worst position: CENTER and LEFT are weak in aggregate, but worst combined rows include both LEFT and RIGHT long-range sitting cases.
- Two-person degradation: not in aggregate old displayed accuracy, but this is not evidence of safety because point-cloud association is missing.
- 3m standing false sitting: aggregate is 3.97%, but `sitting_relative_gate_refined_live_test` reached 16.00%.
- Upright sitting becoming standing: yes, false STANDING on sitting was 55.27% for `SITTING_UPRIGHT`.
- Lean-forward becoming standing/moving: yes, false STANDING on sitting was 40.96% for `SITTING_LEAN_FORWARD`.
- Failures correlate with low geometry evidence: NO_POINTS rates are high, including 78.34% at 3m, 84.50% at 4m, and 82.90% at 5m.

## 5. RGB Teacher Summary

RGB audit outputs were written under:

`analysis_outputs\interim_existing_data_use\rgb_teacher_audit`

Status: partially usable.

Findings:

- RGB video, RGB keypoints, RGB tracks, RGB frames, and sync rows exist for all 9 sessions.
- Mean keypoint score is approximately 0.720.
- Shoulders and hips are available enough to compute rough torso features.
- Two-person RGB tracks can distinguish left/right in `session_20260704_145249` and `session_20260704_150636`.
- Knees and ankles are not available in the current keypoint schema, so body-height and knee/hip sitting geometry cannot be computed.
- Automatic RGB-derived STANDING/SITTING candidate labels were not high-confidence enough to use as ground truth.

RGB-assisted label outputs were written under:

`analysis_outputs\interim_existing_data_use\rgb_assisted_labels`

Those labels are analysis/review aids only. They should not be used as runtime labels or as unverified training ground truth.

## 6. Reliability Model Result

Reliability outputs were written under:

`analysis_outputs\interim_existing_data_use\reliability_model`

Best model: `RandomForestClassifier`.

Result: did not pass acceptance.

Key metrics:

- Trust coverage: 7.76%.
- Trusted accuracy: 88.71%.
- Trusted false SITTING on standing_3m: 1.96%.
- Baseline false SITTING on standing_3m: 3.97%.
- Correct standing preservation: 14.03%.
- Correct sitting preservation: 8.81%.
- Wrong rejection rate: 97.67%.
- Uncertain or low-visibility rate: 89.58%.

The model reduced trusted false sitting on standing_3m, but mostly by rejecting too many windows. It did not preserve enough correct standing or correct sitting to be useful for runtime integration.

## 7. Whether Any Interim Improvement Is Possible

An offline reliability gate is possible for analysis, but it is not ready for runtime use. The best reliability model was too conservative and did not pass acceptance. No interim runtime improvement should be integrated from current lite logs.

## 8. Why No Posture Replacement Model Is Integrated

No replacement posture model was integrated because:

- the previous lite replacement model failed acceptance,
- current logs lack associated point-cloud tensors,
- sitting subtypes fail heavily at 4m-5m,
- the conservative reliability model rejected too much data,
- RGB labels are only partially usable and not ground truth.

Runtime behavior remains unchanged.

## 9. What To Collect Next

Collect new sessions only after associated point-cloud logging is available and validated.

Priority next sessions:

- `pc_standing_center_1to5_01`
- `pc_sitting_leanback_center_1to5_01`
- `pc_sitting_upright_center_1to5_01`
- `pc_sitting_leanforward_center_1to5_01`
- `pc_standing_lr_1to5_01`
- `pc_sitting_leanback_lr_1to5_01`
- `pc_sitting_upright_lr_1to5_01`
- `pc_sitting_leanforward_lr_1to5_01`
- `pc_two_person_standing_lr_1to5_01`
- `pc_two_person_sitting_leanback_lr_1to5_01`
- `pc_two_person_sitting_upright_lr_1to5_01`
- `pc_two_person_sitting_leanforward_lr_1to5_01`
- `pc_static_retention_sitting_leanback_center_1to5_01`, only if intentionally comparing cfg behavior.

Treat 5m as limited/low-confidence until full point-cloud logs prove reliability.

## 10. Exact Next Engineering Step

Finish non-invasive associated point-cloud logging, run a 30-second smoke test, validate `mmwave_associated_points.csv`, then collect the targeted point-cloud sessions. After that, rebuild the full RadarPostureNet-v2 dataset and train the coordinate-invariant full model.

## Final Decision Table

| question | answer | evidence |
|---|---|---|
| Can existing data still be used? | yes | It produced failure maps, RGB audit outputs, RGB-assisted review rows, and an offline reliability-model evaluation. |
| Can it train a replacement model? | no | The previous lite model failed acceptance; full point-cloud tensors are unavailable. |
| Can it train a reliability model? | yes, offline only | A `RandomForestClassifier` reliability model trained, but failed acceptance. |
| Can RGB keypoints help labels? | partial | All sessions have RGB/sync data, but knees and ankles are unavailable, so automatic labels are not ground truth. |
| What is the worst posture case? | long-range sitting, especially upright and lean-back | `SITTING_UPRIGHT` accuracy 39.63%; `SITTING_LEAN_BACK` accuracy 40.76%; several 4m combined cases are near zero. |
| What is the worst distance? | 5m | 5m overall accuracy 39.46%; 5m sitting accuracy 11.24%. |
| What is the worst position? | center/left in aggregate, with severe combined failures on both left and right | CENTER accuracy 54.08%; LEFT 55.40%; worst combined rows include 4m left upright and 4m right lean-back. |
| Do two-person sessions degrade posture? | not in aggregate, but not proven safe | Two-person aggregate accuracy 68.35% versus one-person 55.86%, but associated point tensors are missing. |
| Should we collect more data before point logging? | no, except manual RGB review | More lite-only training is unlikely to solve the posture geometry problem. |
| What should we do next? | validate associated point logging, then collect targeted sessions | Full RadarPostureNet-v2 requires per-frame, per-TID associated point-cloud logs. |
