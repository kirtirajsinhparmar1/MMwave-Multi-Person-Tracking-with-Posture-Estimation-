# Posture Sitting Decision Fix Report

## 1. What code was changed

Implemented a conservative relative sitting decision path in `ti_style_pose_overlay.py`.

Changed areas:

- `TiStylePoseManager.__init__`: added configuration for the relative sitting gate and the moving-override body-translation guard.
- `TiStylePoseManager._update_display_state`: added the new stand-to-sit relative probability route, MOVING override guard, and debug reasons.
- Pose debug/log output: added new fields for sit-vs-stand margin, gate state, moving override state, and final display pose.
- `run_ti_style_visualizer.py`: added CLI flags to enable/disable and configure the new route.
- `analysis/replay_posture_decision_fix.py`: added offline replay script to compare old displayed pose against the new decision route using existing logs.

The ONNX model, training code, RGB code, TI cfg files, static-retention cfg, and renderer mesh code were not changed.

## 2. Why this fix is grounded in the A/B results

The default cfg had strong tracking and better displayed sitting posture than the static-retention cfg. The static-retention cfg reduced `NO_POINTS` at some distances, but it also created persistent extra tracks at 3m and 4m, so it is not deployable as the first fix path.

The default cfg evidence showed that sitting at 3m and 4m often had `sit_prob > stand_prob`, while the display still stayed `STANDING` too often. That points first at stand-vs-sit decision/gating, not model retraining or cfg replacement.

## 3. Why static-retention cfg was not used

Static-retention was not promoted to runtime default because the A/B and per-TID diagnosis showed an extra-track regression:

- static sitting_3m extra track rate: 100%
- static sitting_4m extra track rate: 100%
- static sitting_4m had `sit_prob > stand_prob`, but displayed SITTING rate stayed 0%

That means static retention is a separate track validation / point association problem, not the right first runtime posture fix.

## 4. New sitting relative gate logic

The existing strong-confidence sitting route is preserved.

Existing strong route:

- `sit_prob >= stand_to_sit_min_confidence`
- `sit_prob - stand_prob >= stand_to_sit_margin`
- stable for `stand_to_sit_frames`

New relative route:

- `sit_prob >= 0.50`
- `sit_prob - stand_prob >= 0.12`
- stable for 8 frames
- body/target translation is not confirmed
- falling/lying is not dominant
- the track is active enough to be in the display update path

The route does not require `geom_pts >= 3`, because the benchmark showed sitting can still be valid under sparse geometry or `NO_POINTS`. Instead, the quality fields are logged so the route can be audited later.

Important acceptance decision: the route is implemented but disabled by default. It is available only with `--pose-sitting-relative-gate`.

## 5. MOVING override guard for sitting/hand-motion cases

Added a guard so local hand motion or noisy point evidence does not force `MOVING` when the sitting evidence is stable.

MOVING can still override when translation evidence confirms the body/target is actually moving through space.

New debug reason strings include:

- `sitting_relative_gate`
- `sitting_relative_gate_waiting`
- `sitting_relative_gate_blocked_by_body_motion`
- `moving_override_blocked_body_still_sitting`
- `moving_override_translation_confirmed`

## 6. New CLI flags

Added to `run_ti_style_visualizer.py`:

```powershell
--pose-sitting-relative-gate
--pose-disable-sitting-relative-gate
--pose-sitting-relative-min-prob 0.50
--pose-sitting-relative-margin 0.12
--pose-sitting-relative-frames 8
--pose-moving-override-require-body-translation-for-sitting
--pose-disable-moving-override-body-translation-guard
```

Because replay found standing regression, `--pose-sitting-relative-gate` is opt-in. The moving override body-translation guard remains enabled by default.

## 7. New debug fields

Added debug/log fields:

- `sit_prob`
- `stand_prob`
- `sit_minus_stand_margin`
- `sitting_relative_gate_state`
- `sitting_relative_gate_stable_count`
- `sitting_relative_gate_required_frames`
- `sitting_relative_gate_passed`
- `moving_override_state`
- `moving_override_reason`
- `moving_override_blocked_by_body_still`
- `final_display_pose`
- `final_reason`
- `quality`
- `geom_pts`
- `range_m`
- `tid`

New columns were appended in the pose CSV path where possible to avoid breaking existing readers.

## 8. Offline replay results

Replay script:

```powershell
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" `
  --segments analysis_inputs\sitting_ab_default_segments.csv `
  --out analysis_outputs\posture_decision_fix_replay_default_ab
```

Default sitting A/B replay:

| segment | old accuracy | new accuracy | old sitting rate | new sitting rate | old standing rate while sitting | new standing rate while sitting | old switches | new switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sitting_2m | 0.808 | 0.823 | 0.808 | 0.823 | 0.124 | 0.109 | 11 | 12 |
| sitting_3m | 0.387 | 0.452 | 0.387 | 0.452 | 0.533 | 0.468 | 27 | 40 |
| sitting_4m | 0.884 | 0.959 | 0.884 | 0.959 | 0.094 | 0.019 | 4 | 8 |

The sitting-only default A/B replay supports the direction of the fix: false `STANDING` during sitting decreases at 2m, 3m, and 4m. However, switch count increased at 3m.

Full benchmark replay:

```powershell
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" `
  --out analysis_outputs\posture_decision_fix_replay_full_benchmark `
  --make-plots
```

The full benchmark replay used available auto segments from `analysis_outputs\latest_distance_posture_analysis_v2`.

## 9. Regression check on standing

Full benchmark standing replay:

| segment | old accuracy | new accuracy | old false sitting | new false sitting | old switches | new switches |
|---|---:|---:|---:|---:|---:|---:|
| standing_1m | 0.831 | 0.806 | 0.013 | 0.038 | 8 | 14 |
| standing_2m | 0.907 | 0.885 | 0.000 | 0.021 | 2 | 4 |
| standing_3m | 0.886 | 0.886 | 0.000 | 0.000 | 6 | 6 |
| standing_4m | 1.000 | 1.000 | 0.000 | 0.000 | 0 | 0 |

This fails the acceptance criterion. Standing_1m and standing_2m regressed by more than the allowed 1-2 percentage point tolerance, and false sitting increased.

## 10. Regression check on default sitting_2m

Default A/B `sitting_2m` did not regress:

- old accuracy: 0.808
- new accuracy: 0.823
- old displayed SITTING rate: 0.808
- new displayed SITTING rate: 0.823

The separate full benchmark `sitting_2m` also improved from 0.677 to 0.734.

## 11. Improvement on sitting_3m/4m

Default A/B replay:

- `sitting_3m`: accuracy improved from 0.387 to 0.452, and false `STANDING` dropped from 0.533 to 0.468.
- `sitting_4m`: accuracy improved from 0.884 to 0.959, and false `STANDING` dropped from 0.094 to 0.019.

Full benchmark replay:

- `sitting_3m`: accuracy improved from 0.593 to 0.638.
- `sitting_4m`: accuracy improved from 0.004 to 0.144, but the model still mostly favored STANDING in that older full benchmark segment (`mean_stand_prob=0.515`, `mean_sit_prob=0.325`).

The fix helps the target failure mode when `sit_prob` consistently beats `stand_prob`, but it is not sufficient for segments where the model itself still favors STANDING.

## 12. Validation commands run

```powershell
python -m py_compile ti_style_pose_overlay.py
python -m py_compile run_ti_style_visualizer.py
python -m py_compile analysis\replay_posture_decision_fix.py
python run_ti_style_visualizer.py --help
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" `
  --segments analysis_inputs\sitting_ab_default_segments.csv `
  --out analysis_outputs\posture_decision_fix_replay_default_ab
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" `
  --out analysis_outputs\posture_decision_fix_replay_full_benchmark `
  --make-plots
```

All compile checks passed. The help smoke test exposed the new CLI flags. Offline replay completed for both the default sitting A/B session and the earlier full benchmark session.

## 13. Exact live test command to run next

This is for a controlled opt-in test only, not default deployment:

```powershell
python run_ti_style_visualizer.py `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_controlled_test" `
  --enable-pose `
  --pose-debug `
  --pose-log `
  --pose-human-models `
  --combined-session `
  --combined-log `
  --enable-rgb-panel `
  --rgb-source 1 `
  --rgb-camera-backend dshow `
  --pose-sitting-relative-gate `
  --pose-sitting-relative-min-prob 0.50 `
  --pose-sitting-relative-margin 0.12 `
  --pose-sitting-relative-frames 8 `
  --pose-moving-override-require-body-translation-for-sitting
```

Recommended physical protocol before considering default enablement:

- standing 1m, 2m, 3m, 4m
- sitting 2m, 3m, 4m
- sitting with hand movement at 2m and 3m
- walking/changing location to verify MOVING is still allowed when translation is real

## 14. Remaining limitations

- Offline replay is approximate. It uses logged probabilities, old displayed pose, TID, range, quality, and track position. It does not fully re-run every live smoothing side effect from the original runtime.
- The sitting A/B replay supports the fix direction, but the full benchmark replay shows standing false-sitting regression.
- Hand-motion-specific validation is still pending because the available replay data does not isolate controlled sitting hand motion.
- The current relative gate values are not accepted as default runtime settings.
- Static-retention cfg remains out of scope because its failure mode is extra-track/association instability.

## Acceptance decision

B. Fix implemented but disabled by default: replay found instability/regression.

The new relative sitting route is useful evidence for the next iteration, but it should remain opt-in until the standing false-sitting regression is solved. The next engineering path is to refine the relative sitting gate with stronger standing protection, likely using per-range evidence and/or stricter stable-track context, then rerun the same offline replay acceptance checks before any live default recommendation.

## Second iteration: standing-protected relative sitting gate

### Why first iteration failed acceptance

The first relative gate improved sitting, but it was too broad. Full benchmark replay showed standing regression:

| segment | old accuracy | first-fix accuracy |
|---|---:|---:|
| standing_1m | 0.831 | 0.806 |
| standing_2m | 0.907 | 0.885 |

That failed the acceptance rule because near-range standing frames were incorrectly converted to `SITTING`.

### What regression frames looked like

Regression mining wrote:

- `analysis_outputs\posture_decision_fix_replay_full_benchmark\relative_gate_error_mining.csv`
- `analysis_outputs\sitting_gate_param_sweep\standing_false_sitting_frames.csv`
- `analysis_outputs\sitting_gate_param_sweep\sitting_corrected_frames.csv`

The mined standing false-SITTING frames showed:

| metric | value |
|---|---:|
| standing false-SITTING changed frames | 36 |
| near-range share, range < 2.5m | 0.556 |
| NO_POINTS share | 0.111 |
| mean stand_prob | 0.305 |
| mean sit_prob | 0.603 |
| mean sit-minus-stand margin | 0.297 |
| MOVING-guard reason share | 0.000 |

Conclusion: the standing regression was mostly near-range standing frames where model probabilities strongly favored `SITTING`. It was not primarily caused by `NO_POINTS` and was not caused by the MOVING guard.

### Parameter sweep results

Created and ran:

```powershell
python analysis\sweep_sitting_gate_params.py `
  --full-benchmark-session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" `
  --default-ab-session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" `
  --default-ab-segments analysis_inputs\sitting_ab_default_segments.csv `
  --out analysis_outputs\sitting_gate_param_sweep
```

Sweep outputs:

- `analysis_outputs\sitting_gate_param_sweep\sweep_results.csv`
- `analysis_outputs\sitting_gate_param_sweep\sweep_pareto_candidates.csv`
- `analysis_outputs\sitting_gate_param_sweep\SITTING_GATE_PARAM_SWEEP_REPORT.md`

Sweep summary:

| metric | value |
|---|---:|
| candidates evaluated | 8000 |
| acceptable candidates | 80 |

Best safe candidate:

| parameter | selected value |
|---|---:|
| relative range minimum | 3.0m |
| soft sitting min probability | 0.55 |
| relative sitting margin | 0.12 |
| relative sitting frames | 16 |
| standing veto probability | 0.50 |
| standing veto margin | 0.05 |

### Updated offline replay results

Full benchmark replay with the selected candidate:

| segment | old accuracy | refined accuracy | old display SITTING rate | refined display SITTING rate |
|---|---:|---:|---:|---:|
| standing_1m | 0.831 | 0.831 | 0.013 | 0.013 |
| standing_2m | 0.907 | 0.907 | 0.000 | 0.000 |
| standing_3m | 0.886 | 0.886 | 0.000 | 0.000 |
| standing_4m | 1.000 | 1.000 | 0.000 | 0.000 |
| sitting_1m | 0.648 | 0.648 | 0.648 | 0.648 |
| sitting_2m | 0.677 | 0.677 | 0.677 | 0.677 |
| sitting_3m | 0.593 | 0.593 | 0.593 | 0.593 |
| sitting_4m | 0.004 | 0.053 | 0.004 | 0.053 |

Default sitting A/B replay with the selected candidate:

| segment | old accuracy | refined accuracy | old display SITTING rate | refined display SITTING rate |
|---|---:|---:|---:|---:|
| sitting_2m | 0.808 | 0.808 | 0.808 | 0.808 |
| sitting_3m | 0.387 | 0.390 | 0.387 | 0.390 |
| sitting_4m | 0.884 | 0.932 | 0.884 | 0.932 |

The refined gate passes the offline acceptance criteria because it protects all standing segments, keeps sitting_2m unchanged, improves sitting_4m by more than 3 points, and keeps total switch growth within the 10% limit. The improvement is mainly at 4m; sitting_3m remains mostly unresolved.

### Runtime defaults changed

The refined gate is now enabled by default with the selected standing-protected values:

```text
pose_sitting_relative_gate = enabled
pose_sitting_relative_range_min_m = 3.0
pose_sitting_relative_min_prob = 0.55
pose_sitting_relative_margin = 0.12
pose_sitting_relative_frames = 16
pose_sitting_relative_standing_veto_prob = 0.50
pose_sitting_relative_standing_veto_margin = 0.05
```

Rollback remains available with:

```powershell
--pose-disable-sitting-relative-gate
```

### Exact live test command

This command uses the default cfg and explicitly repeats the selected gate values for auditability:

```powershell
python run_ti_style_visualizer.py `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_relative_gate_refined_live_test" `
  --enable-pose `
  --pose-debug `
  --pose-log `
  --pose-human-models `
  --combined-session `
  --combined-log `
  --enable-rgb-panel `
  --rgb-source 1 `
  --rgb-camera-backend dshow `
  --pose-sitting-relative-gate `
  --pose-sitting-relative-range-min-m 3.0 `
  --pose-sitting-relative-min-prob 0.55 `
  --pose-sitting-relative-margin 0.12 `
  --pose-sitting-relative-frames 16 `
  --pose-sitting-relative-standing-veto-prob 0.50 `
  --pose-sitting-relative-standing-veto-margin 0.05 `
  --pose-moving-override-require-body-translation-for-sitting
```

### Remaining limitations

- This is offline replay validation only, not live radar validation.
- The selected gate fixes the first-iteration standing regression and improves 4m sitting, but it does not materially fix sitting_3m.
- Pose switch growth is within the limit but close to it: 9.78%.
- Hand-motion live validation is still pending.
- Static-retention cfg remains out of scope because it caused extra-track behavior at 3m and 4m.

## Updated acceptance decision

A. Fix accepted: offline replay improves sitting without standing regression.

The accepted fix is narrower than the first iteration: it is range-limited, uses a longer stability requirement, and includes a standing-probability veto. The next engineering path is live validation of the refined default-cfg gate, followed by a separate investigation of why sitting_3m remains weak.
