# Distance/Posture Analysis Pipeline

## What the script does
`analysis/analyze_distance_posture_session.py` builds an offline report for continuous mmWave + RGB benchmark sessions. It discovers session files, normalizes different CSV schemas, slices the run into standing/sitting distance segments, computes tracking metrics separately from posture metrics, creates diagnostic CSVs/plots, and writes markdown/HTML reports.

## How latest sessions are found
The script scans `logs`, `..\logs`, `C:\Users\UBESC\Desktop\Combined MMwave and RGB\logs`, and any `--log-root` paths. Candidate directories are ranked by modified time and by whether they contain useful CSV/JSON/log/video files. `--latest` selects the newest useful candidate.

## What files are parsed
Common files include `mmwave_frames.csv`, `mmwave_tracks.csv`, `mmwave_pose.csv`, `pose_predictions_ui.csv`, `targets.csv`, `frames_summary.csv`, `rgb_frames.csv`, `rgb_tracks.csv`, `rgb_keypoints.csv`, `sync_index.csv`, `events.csv`, `events.jsonl`, `combined_events.csv`, `session_metadata.json`, and `videos/rgb_annotated.mp4`.

## How automatic segmentation works
The expected ground-truth order is standing at 1m/2m/3m/4m followed by sitting at 1m/2m/3m/4m. Auto segmentation searches for stable range plateaus in that order and trims 5 seconds from each end when possible. If range plateaus cannot be inferred for all segments, it writes warnings and falls back to equal-time best-effort segments.

## How to manually override segments
Create or edit `analysis_outputs/latest_distance_posture_analysis/segments_manual_template.csv` with:

```text
segment_id,expected_pose,expected_distance_m,start_time_s,end_time_s
```

Then rerun with:

```powershell
python analysis\analyze_distance_posture_session.py --log-root "..\logs" --latest --out analysis_outputs\latest_distance_posture_analysis --manual-segments analysis_outputs\latest_distance_posture_analysis\segments_manual_template.csv --make-plots
```

## Tracking metrics
Tracking metrics include presence/dropout, range MAE/RMSE/bias, position jitter, TID continuity/switches, active-track and ghost/shadow rates, plus an interpretable tracking score with components reported separately.

## Posture metrics
Posture metrics use `display_pose`/`final_label` as the main UI prediction and keep tracking independent. Metrics include accuracy, pose distribution, false MOVING/FALLING/LYING/UNKNOWN rates, latency to first/stable correct prediction, switch rates, and breakdowns by quality/geom points/association/reason fields.

## Plot list
The script writes timeline, tracking, posture, and summary plots under `plots/`, including range by track, active track count, display pose, quality/geom points, stand/sit probabilities, tracking presence/range error/jitter/ghosts/TID switches, XY scatter, posture accuracy/confusion/distribution, quality breakdowns, moving false positives, false falling rates, stable-correct latency, tracking-vs-posture summary, and failure heatmap.

## Output directory structure
Outputs are written under the selected `--out` directory, including `file_inventory.csv`, `segments_auto.csv`, metric CSVs, event CSVs, `warnings.txt`, `plots/`, `DISTANCE_POSTURE_BENCHMARK_REPORT.md`, and `DISTANCE_POSTURE_BENCHMARK_REPORT.html`.

## Validation commands run

```powershell
python -m py_compile analysis\analyze_distance_posture_session.py
python analysis\analyze_distance_posture_session.py --log-root "..\logs" --latest --out analysis_outputs\latest_distance_posture_analysis --make-plots
```

## Known limitations
Automatic segment boundaries are best effort and must be inspected before treating results as final. Equal-time fallback is intentionally conservative and flagged in warnings. RGB is supplementary unless a clean ground-truth label exists. Text fallback parsing only extracts common pose-debug patterns.

## Exact command to run

```powershell
python analysis\analyze_distance_posture_session.py --log-root "..\logs" --latest --out analysis_outputs\latest_distance_posture_analysis --make-plots
```
