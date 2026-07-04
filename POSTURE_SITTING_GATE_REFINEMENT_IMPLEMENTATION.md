# Posture Sitting Gate Refinement Implementation

## 1. Regression cause summary

The first relative sitting gate improved sitting but regressed standing_1m and standing_2m. Error mining showed the false `SITTING` frames were mostly near-range standing frames where the model probabilities favored `SITTING`:

- standing false-SITTING changed frames: 36
- near-range share, range < 2.5m: 0.556
- NO_POINTS share: 0.111
- mean stand_prob: 0.305
- mean sit_prob: 0.603
- mean sit-minus-stand margin: 0.297
- MOVING-guard reason share: 0.000

So the fix needed standing protection, not MOVING-guard changes.

## 2. Scripts created/updated

Created:

- `analysis\sweep_sitting_gate_params.py`

Updated:

- `analysis\replay_posture_decision_fix.py`
- `ti_style_pose_overlay.py`
- `run_ti_style_visualizer.py`
- `POSTURE_SITTING_DECISION_FIX_REPORT.md`

Created reports:

- `POSTURE_SITTING_GATE_REFINEMENT_REPORT.md`
- `POSTURE_SITTING_GATE_REFINEMENT_IMPLEMENTATION.md`

Generated outputs:

- `analysis_outputs\sitting_gate_param_sweep\relative_gate_error_mining.csv`
- `analysis_outputs\sitting_gate_param_sweep\standing_false_sitting_frames.csv`
- `analysis_outputs\sitting_gate_param_sweep\sitting_corrected_frames.csv`
- `analysis_outputs\sitting_gate_param_sweep\sweep_results.csv`
- `analysis_outputs\sitting_gate_param_sweep\sweep_pareto_candidates.csv`
- `analysis_outputs\sitting_gate_param_sweep\SITTING_GATE_PARAM_SWEEP_REPORT.md`
- `analysis_outputs\posture_decision_fix_replay_full_benchmark_refined`
- `analysis_outputs\posture_decision_fix_replay_default_ab_refined`

## 3. Sweep parameter ranges

| parameter | values |
|---|---|
| range_min_for_relative_gate_m | 0.0, 2.0, 2.5, 2.75, 3.0 |
| soft_sitting_min_prob | 0.50, 0.52, 0.55, 0.58, 0.60 |
| relative_sitting_margin | 0.12, 0.15, 0.18, 0.20, 0.25 |
| relative_sitting_frames | 8, 10, 12, 16 |
| standing_veto_prob | 0.50, 0.55, 0.60, 0.65 |
| standing_veto_margin | 0.05, 0.08, 0.10, 0.12 |

## 4. Best candidate

The sweep evaluated 8000 candidates and found 80 acceptable candidates.

Selected candidate:

| parameter | selected value |
|---|---:|
| range_min_for_relative_gate_m | 3.0 |
| soft_sitting_min_prob | 0.55 |
| relative_sitting_margin | 0.12 |
| relative_sitting_frames | 16 |
| standing_veto_prob | 0.50 |
| standing_veto_margin | 0.05 |

Replay result:

| segment | old accuracy | candidate accuracy |
|---|---:|---:|
| full standing_1m | 0.831 | 0.831 |
| full standing_2m | 0.907 | 0.907 |
| full standing_3m | 0.886 | 0.886 |
| full standing_4m | 1.000 | 1.000 |
| full sitting_4m | 0.004 | 0.053 |
| default A/B sitting_2m | 0.808 | 0.808 |
| default A/B sitting_3m | 0.387 | 0.390 |
| default A/B sitting_4m | 0.884 | 0.932 |

## 5. Runtime code changes

`ti_style_pose_overlay.py`:

- Enabled the refined relative sitting gate by default.
- Added `sitting_relative_range_min_m`.
- Added `sitting_relative_standing_veto_prob`.
- Added `sitting_relative_standing_veto_margin`.
- Required range and standing-veto checks before the relative gate can collect stable frames.
- Added block reasons:
  - `sitting_relative_gate_blocked_range`
  - `sitting_relative_gate_blocked_standing_veto`
- Added debug/log fields:
  - `sitting_relative_gate_range_min_m`
  - `sitting_relative_gate_range_ok`
  - `sitting_relative_standing_veto_prob`
  - `sitting_relative_standing_veto_margin`
  - `sitting_relative_standing_veto_ok`

`run_ti_style_visualizer.py`:

- Added `--pose-sitting-relative-range-min-m`.
- Added `--pose-sitting-relative-standing-veto-prob`.
- Added `--pose-sitting-relative-standing-veto-margin`.
- Updated defaults to the selected candidate.
- Kept `--pose-disable-sitting-relative-gate` for rollback.

`analysis\replay_posture_decision_fix.py`:

- Added `--relative-range-min-m`.
- Added `--standing-veto-prob`.
- Added `--standing-veto-margin`.
- Wrote selected parameter values into replay reports.

## 6. Whether the gate is enabled by default

Yes. The refined gate is enabled by default because the selected candidate passed offline acceptance criteria.

Default runtime values:

```text
pose_sitting_relative_gate = enabled
pose_sitting_relative_range_min_m = 3.0
pose_sitting_relative_min_prob = 0.55
pose_sitting_relative_margin = 0.12
pose_sitting_relative_frames = 16
pose_sitting_relative_standing_veto_prob = 0.50
pose_sitting_relative_standing_veto_margin = 0.05
```

Disable with:

```powershell
--pose-disable-sitting-relative-gate
```

## 7. Validation commands run

```powershell
python -m py_compile ti_style_pose_overlay.py
python -m py_compile run_ti_style_visualizer.py
python -m py_compile analysis\replay_posture_decision_fix.py
python -m py_compile analysis\sweep_sitting_gate_params.py
python run_ti_style_visualizer.py --help
python analysis\sweep_sitting_gate_params.py `
  --full-benchmark-session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" `
  --default-ab-session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" `
  --default-ab-segments analysis_inputs\sitting_ab_default_segments.csv `
  --out analysis_outputs\sitting_gate_param_sweep
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\session_20260703_205540" `
  --out analysis_outputs\posture_decision_fix_replay_full_benchmark_refined `
  --make-plots `
  --relative-range-min-m 3.0 `
  --relative-min-prob 0.55 `
  --relative-margin 0.12 `
  --relative-frames 16 `
  --standing-veto-prob 0.50 `
  --standing-veto-margin 0.05
python analysis\replay_posture_decision_fix.py `
  --session "C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs\sitting_ab_default_cfg" `
  --segments analysis_inputs\sitting_ab_default_segments.csv `
  --out analysis_outputs\posture_decision_fix_replay_default_ab_refined `
  --make-plots `
  --relative-range-min-m 3.0 `
  --relative-min-prob 0.55 `
  --relative-margin 0.12 `
  --relative-frames 16 `
  --standing-veto-prob 0.50 `
  --standing-veto-margin 0.05
```

All commands completed successfully. No live radar validation is claimed.

## 8. Exact next live test command

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

Recommended live protocol:

- standing 1m, 2m, 3m, 4m
- sitting 2m, 3m, 4m
- sitting with hand motion
- walking/changing location to confirm MOVING still works when translation is real
