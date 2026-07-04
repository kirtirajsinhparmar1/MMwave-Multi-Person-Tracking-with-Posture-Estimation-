# Posture Sitting Gate Refinement Report

## Executive conclusion

The first relative sitting gate helped sitting but caused false `SITTING` on standing_1m and standing_2m. Regression mining showed those bad frames were mostly near-range standing frames where the model probabilities themselves favored `SITTING`, not a MOVING-guard interaction.

The second iteration adds standing protection: range minimum, longer stability, and a standing-probability veto. Offline sweep found an acceptable candidate and replay validation passed the acceptance criteria. The gate is enabled by default with the selected conservative values.

Important limitation: this mainly improves sitting_4m. It does not materially solve sitting_3m.

## Why the first iteration failed

First-fix replay result:

| segment | old accuracy | first-fix accuracy |
|---|---:|---:|
| standing_1m | 0.831 | 0.806 |
| standing_2m | 0.907 | 0.885 |

The first gate was too broad because it allowed moderate relative sitting probability at near range.

## Regression frame mining

Outputs:

- `analysis_outputs\sitting_gate_param_sweep\relative_gate_error_mining.csv`
- `analysis_outputs\sitting_gate_param_sweep\standing_false_sitting_frames.csv`
- `analysis_outputs\sitting_gate_param_sweep\sitting_corrected_frames.csv`

| metric | value |
|---|---:|
| standing false-SITTING changed frames | 36 |
| near-range share, range < 2.5m | 0.556 |
| NO_POINTS share | 0.111 |
| mean stand_prob | 0.305 |
| mean sit_prob | 0.603 |
| mean sit-minus-stand margin | 0.297 |
| MOVING-guard reason share | 0.000 |

Interpretation: standing_1m and standing_2m regressed because the relative gate trusted model sitting probability at near range. False `SITTING` frames were not mostly `NO_POINTS`, and they were not caused by the MOVING body-translation guard.

## Parameter sweep

Created:

- `analysis\sweep_sitting_gate_params.py`

Sweep ranges:

| parameter | values |
|---|---|
| range_min_for_relative_gate_m | 0.0, 2.0, 2.5, 2.75, 3.0 |
| soft_sitting_min_prob | 0.50, 0.52, 0.55, 0.58, 0.60 |
| relative_sitting_margin | 0.12, 0.15, 0.18, 0.20, 0.25 |
| relative_sitting_frames | 8, 10, 12, 16 |
| standing_veto_prob | 0.50, 0.55, 0.60, 0.65 |
| standing_veto_margin | 0.05, 0.08, 0.10, 0.12 |

Sweep outputs:

- `analysis_outputs\sitting_gate_param_sweep\sweep_results.csv`
- `analysis_outputs\sitting_gate_param_sweep\sweep_pareto_candidates.csv`
- `analysis_outputs\sitting_gate_param_sweep\SITTING_GATE_PARAM_SWEEP_REPORT.md`

| metric | value |
|---|---:|
| candidates evaluated | 8000 |
| acceptable candidates | 80 |

## Best safe candidate

| parameter | selected value |
|---|---:|
| range_min_for_relative_gate_m | 3.0 |
| soft_sitting_min_prob | 0.55 |
| relative_sitting_margin | 0.12 |
| relative_sitting_frames | 16 |
| standing_veto_prob | 0.50 |
| standing_veto_margin | 0.05 |

## Replay validation

Full benchmark replay:

| segment | old accuracy | candidate accuracy | delta |
|---|---:|---:|---:|
| standing_1m | 0.831 | 0.831 | 0.000 |
| standing_2m | 0.907 | 0.907 | 0.000 |
| standing_3m | 0.886 | 0.886 | 0.000 |
| standing_4m | 1.000 | 1.000 | 0.000 |
| sitting_1m | 0.648 | 0.648 | 0.000 |
| sitting_2m | 0.677 | 0.677 | 0.000 |
| sitting_3m | 0.593 | 0.593 | 0.000 |
| sitting_4m | 0.004 | 0.053 | +0.049 |

Default sitting A/B replay:

| segment | old accuracy | candidate accuracy | delta |
|---|---:|---:|---:|
| sitting_2m | 0.808 | 0.808 | 0.000 |
| sitting_3m | 0.387 | 0.390 | +0.003 |
| sitting_4m | 0.884 | 0.932 | +0.048 |

Acceptance result:

| criterion | result |
|---|---|
| no standing segment drops by more than 0.5 points | pass |
| standing false-SITTING does not increase by more than 0.5 points | pass |
| sitting_2m does not drop by more than 1 point | pass |
| sitting_3m or sitting_4m improves by at least 3 points | pass, via sitting_4m |
| pose switch count does not increase by more than 10% | pass, 9.78% |

## Runtime changes

Updated `ti_style_pose_overlay.py`:

- Relative sitting gate default is enabled.
- Added range minimum check.
- Added standing-probability veto.
- Added debug/log fields for range and veto status.

Updated `run_ti_style_visualizer.py`:

- Added `--pose-sitting-relative-range-min-m`.
- Added `--pose-sitting-relative-standing-veto-prob`.
- Added `--pose-sitting-relative-standing-veto-margin`.
- Updated defaults to the selected sweep candidate.

Updated `analysis\replay_posture_decision_fix.py`:

- Added replay parameters for range minimum and standing veto.
- Added parameter reporting.

## Final decision

A. Fix accepted: offline replay improves sitting without standing regression.

This is not a broad posture retune. It is a standing-protected, range-limited relative transition path for default-cfg cases where sitting probability consistently exceeds standing probability.

## Remaining limitations

- No live radar validation has been claimed.
- sitting_3m remains mostly unresolved.
- The switch-count increase passes but is close to the 10% ceiling.
- Hand-motion validation remains a live-test item.
- Static-retention cfg was not used because it created persistent extra tracks at 3m and 4m.

## Exact live test command

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
