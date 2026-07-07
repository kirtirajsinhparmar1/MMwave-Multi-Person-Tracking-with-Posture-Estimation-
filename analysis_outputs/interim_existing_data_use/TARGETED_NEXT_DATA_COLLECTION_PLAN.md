# Targeted Next Data Collection Plan

Generated from the existing cleaned posture registry, lite dataset, failure map, RGB teacher audit, and conservative reliability analysis.

## 1. Conditions That Need More Data

The most important missing data is not more lite-only posture data. The next useful collection should happen after per-frame, per-TID associated point-cloud logging is enabled.

Priority conditions:

- 5m sitting is the weakest distance condition. Existing old displayed posture sitting accuracy at 5m is 11.24%, with 73.94% false STANDING on SITTING.
- 4m sitting is also weak. Existing old displayed posture sitting accuracy at 4m is 24.30%.
- Sitting subtype coverage needs stronger full point-cloud data, especially `SITTING_UPRIGHT` and `SITTING_LEAN_BACK`. Existing old displayed accuracies are 39.63% and 40.76%.
- `sitting_ab_static_retention_cfg` should be repeated with associated point-cloud logging because its old displayed sitting accuracy was 0.91%, making it a high-value failure-control session.
- Standing at 3m still needs protection checks. Aggregate false SITTING on standing_3m was 3.97%, but `sitting_relative_gate_refined_live_test` had 16.00%.
- Left/right sitting at 3m-5m needs repeat collection with associated point logging. Some combined conditions had near-zero old displayed accuracy, including 4m `SITTING_UPRIGHT` LEFT and 4m `SITTING_LEAN_BACK` RIGHT.
- Two-person sitting sessions need full point-cloud logging even though the lite aggregate did not collapse. The existing logs have high NO_POINTS rates and no per-point target-centered shape tensors.

## 2. Conditions Already Useful As References

These conditions are useful as controls, not as evidence that runtime replacement is safe:

- Standing is comparatively stronger than sitting in the old runtime. Existing aggregate old displayed standing accuracy was 91.68%.
- 1m-2m recordings are better than 4m-5m and should be used as near-range reference controls.
- Two-person aggregate old displayed accuracy was 68.35%, higher than one-person aggregate at 55.86%, but that does not prove two-person handling is solved because point association and per-person shape evidence are missing.
- `session_20260706_173741` is a useful standing side-position control session; old displayed standing accuracy was 90.04%.

## 3. Conditions That Need Point-Cloud Logging First

Do not spend another bounded pass tuning lite replacement posture models before collecting associated point logs.

Point-cloud logging is required for:

- target-centered body-shape tensors,
- posture geometry at 4m-5m,
- separating upright sitting from standing,
- separating lean-forward sitting from standing or moving,
- measuring left/right target shape independently,
- diagnosing NO_POINTS and LOW_POINTS as data quality signals instead of treating them as invisible model failures.

## 4. Recommended Sessions To Collect After Associated Point Logging

Use the default cfg first unless intentionally running an A/B cfg comparison. Keep RGB video visible enough for manual review, with full body in frame where possible.

Recommended exact new sessions:

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
- `pc_static_retention_sitting_leanback_center_1to5_01`, only if deliberately comparing against `sitting_ab_static_retention_cfg`.

Each segment should target 60 seconds after a short transition period. Mark the start/end protocol verbally or with a manual note so the RGB/video review has clear boundaries.

## 5. Two-Person Collection

Collect two-person data again with associated point logging. Existing two-person sessions:

- `session_20260704_145249`
- `session_20260704_150636`

These did not collapse in aggregate lite metrics, but the absence of associated point tensors means the final full architecture still cannot learn target-centered shape per TID from them.

## 6. Left/Right Position Collection

Collect left/right single-person and two-person data at all distances, but prioritize 3m, 4m, and 5m. Existing left/right posture behavior is uneven:

- LEFT overall old displayed accuracy: 55.40%.
- RIGHT overall old displayed accuracy: 75.31%.
- Worst combined cases include 4m `SITTING_UPRIGHT` LEFT and 4m `SITTING_LEAN_BACK` RIGHT.

## 7. 5m Handling

Treat 5m as low-confidence or limited-range until associated point-cloud data proves otherwise. Existing 5m old displayed accuracy was 39.46%, and sitting accuracy was only 11.24%.

The next full-data collection should still include 5m so the model can learn when posture is reliable versus not reliable at range.

## 8. RGB Manual Verification

RGB is partially usable now:

- All 9 existing sessions have RGB video, keypoints, tracks, frames, and sync rows.
- Shoulders and hips are present enough to compute rough torso features.
- Knees and ankles are not available in the current keypoint schema, so automatic sitting/standing teacher labels are not strong enough to be treated as ground truth.

Use RGB for manual verification, transition detection, and left/right review. Do not use RGB-derived labels as ground truth without manual confirmation or improved lower-body keypoint logging.

## 9. Existing Sessions Used For This Plan

The plan used all user-provided sessions:

- `session_20260703_205540`
- `sitting_ab_default_cfg`
- `sitting_ab_static_retention_cfg`
- `sitting_relative_gate_refined_live_test`
- `session_20260704_145249`
- `session_20260704_150636`
- `session_20260704_152302`
- `session_20260706_173741`
- `session_20260706_175519`

## 10. Exact Next Step

Finish and validate associated point-cloud logging, run a 30-second smoke test, validate `mmwave_associated_points.csv`, then collect the targeted sessions above. After that, rebuild the full coordinate-invariant RadarPostureNet-v2 dataset using per-TID target-centered point tensors.
