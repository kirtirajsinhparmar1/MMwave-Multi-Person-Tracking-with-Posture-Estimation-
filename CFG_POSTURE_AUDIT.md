# ODS 6m Posture CFG Audit

## Files created

- `cfg/ODS_6m_posture_tuned.cfg`
- `CFG_POSTURE_AUDIT.md`

The original TI file was preserved and was not modified:
  
```text
C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg
```

The posture-tuned copy currently keeps the original numeric cfg values and adds comments around posture-sensitive tracker and mount fields. This is intentional: the first useful test is an A/B run with the same radar command values but explicit posture logging and sensor-mount awareness.

## Important cfg parameters found

### SDK and chirp setup

```text
channelCfg 15 7 0
```

Controls enabled RX/TX channel masks and cascading mode. It affects antenna aperture and therefore angle/height estimation. For standing/sitting, bad antenna/channel setup would degrade vertical geometry. Leave alone now because this is board/demo-specific TI default.

```text
profileCfg 0 60.75 30.00 25.00 59.10 657930 0 54.71 1 96 2950.00 2 1 36
```

Controls chirp profile timing, slope, ADC samples, sampling rate, and RX gain. It affects range resolution, maximum range, and point-cloud density. It can affect body-shape detail, but changing it is not conservative. Leave alone now.

```text
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
```

Controls chirp indices and TX antenna usage. This affects virtual antenna geometry and elevation/azimuth quality. Leave alone now.

```text
frameCfg 0 2 96 0 55.00 1 0
```

Controls chirp loop/frame timing. The 55 ms frame periodicity influences tracking smoothness and how many frames are needed for sitting/standing stability. Leave alone now.

### Detection and point-cloud density

This cfg does not contain legacy `cfarCfg`, `cfarFovCfg`, or `aoaFovCfg` lines. The People Tracking equivalent detection-layer commands present here are:

```text
dynamicRACfarCfg -1 4 4 2 2 8 12 4 12 5.00 8.00 0.40 1 1
staticRACfarCfg -1 6 2 2 2 8 8 6 4 8.00 15.00 0.30 0 0
dynamicRangeAngleCfg -1 0.75 0.0010 1 0
dynamic2DAngleCfg -1 3.0 0.0300 1 0 1 0.30 0.85 8.00
staticRangeAngleCfg -1 0 8 8
fovCfg -1 70.0 70.0
```

`dynamicRACfarCfg` controls dynamic range-angle CFAR windows, guard sizes, range/angle thresholds, sidelobe threshold, second pass, and dynamic flag. Lower thresholds can increase body points, but also increase noise and ghosts. Leave unchanged for the first posture cfg.

`staticRACfarCfg` controls static range-angle CFAR behavior. Stationary seated posture can depend on static returns, but loosening this too early may add clutter. Leave unchanged now; consider a sensitivity variant only after comparing `geom_pts` and `quality`.

`dynamicRangeAngleCfg` and `dynamic2DAngleCfg` control dynamic range/angle search and peak extraction behavior. These can affect how many usable body points are emitted, especially for low-motion seated targets. Leave unchanged now.

`staticRangeAngleCfg -1 0 8 8` has static processing disabled by the second argument `0`. This may matter for stationary seated targets. Do not change in this conservative cfg; if seated targets remain sparse, compare against TI's `ODS_6m_staticRetention.cfg`, which adds `fineMotionCfg`.

`fovCfg -1 70.0 70.0` sets azimuth/elevation field of view. If the physical sensor is tilted such that seated bodies fall near the edge of elevation FOV, point density can drop. Leave unchanged unless live debug shows the person is outside FOV.

No `clutterRemoval` command is present in this cfg. Static clutter behavior is therefore controlled by the People Tracking detection/tracker commands rather than a visible `clutterRemoval` line.

No `guiMonitor` line is present in this cfg. The People Tracking lab uses its own output/control path.

### Boundary boxes

```text
staticBoundaryBox -3 3 0.5 7.5 0 3
```

Controls the region where static/presence-related processing applies: x -3 to 3 m, y 0.5 to 7.5 m, z 0 to 3 m. It includes 1 m, 2-4 m, 4-6 m, floor height, chair height, and standing body height. Risk: anything closer than 0.5 m is excluded for static handling. Leave unchanged now.

```text
boundaryBox -4 4 0 8 0 3
```

Controls the main tracking/scenery region: x -4 to 4 m, y 0 to 8 m, z 0 to 3 m. It covers the posture test area and expected seated/standing heights. This is not obviously restrictive. Leave unchanged now.

```text
presenceBoundaryBox -3 3 0.5 7.5 0 3
```

Controls early presence region. It covers the target posture area except very close range below 0.5 m. Leave unchanged now.

Risk to watch: if live debug shows seated/lower-body returns below `z=0` or outside x/y due to mount transform mismatch, seated points may be excluded. The current boxes themselves are not the likely cause of 2-4 m upright-sitting weakness.

## Sensor Mount Calibration

```text
sensorPosition 2 0 15
```

Syntax confirmed from TI source/user guide:

```text
sensorPosition <Z height meters> <AzimuthTilt degrees> <ElevationTilt degrees>
```

Current configured mount:

- sensor height: `2` m
- azimuth tilt: `0` degrees
- elevation tilt: `15` degrees

This matters for standing/sitting because vertical body geometry and floor-relative height depend on the radar's assumed mounting height and tilt. If the physical sensor is not actually 2 m high with 15 degrees elevation tilt, the tracker may project body points and target heights incorrectly. Distance-dependent posture errors can result because the height error grows with range when tilt is wrong.

If the radar is physically tilted downward differently, use the existing syntax and edit only the third value after measuring the mount:

```text
sensorPosition 2 0 <measured_elevation_tilt_degrees>
```

If the radar height differs, edit the first value:

```text
sensorPosition <measured_height_m> 0 15
```

Do not guess these values from posture output alone. Measure physical height and approximate tilt, then compare logs with `--pose-use-sensor-calibration`.

## Tracker and static-target audit

```text
gatingParam 3 2 2 3 4
```

Controls tracker gating volume and limits. If too tight, seated targets with weaker or lower point clusters may fail to stay associated. Current values are TI defaults and not obviously posture-hostile. Leave unchanged now.

```text
stateParam 3 3 12 500 5 6000
```

Controls tracker state transitions: detection-to-active, detection-to-free, active-to-free, static-to-free, exit-to-free, and sleep-to-free. The long static/sleep values should help retain stationary people. Leave unchanged now.

```text
allocationParam 20 100 0.1 20 0.5 20
```

Controls allocation thresholds such as SNR, minimum velocity/points, distance, and velocity separation. If too aggressive, stationary seated targets may allocate poorly. Current values are TI defaults. Leave unchanged now.

```text
maxAcceleration 0.1 0.1 0.1
```

Controls tracker acceleration assumptions. Low values favor smooth slow human motion, which is reasonable for posture testing. Leave unchanged now.

```text
trackingCfg 1 2 800 30 46 96 55
```

Controls tracker-level operation and frame-related parameters. It affects track stability and output capacity. Leave unchanged now.

Static seated targets can still be weak if static point extraction is sparse. If `geom_pts` remains zero after the association fix, the next controlled experiment should compare this cfg against the TI `ODS_6m_staticRetention.cfg` rather than making several simultaneous changes.

## Tuned cfg variants created

Only one variant was created:

```text
cfg/ODS_6m_posture_tuned.cfg
```

No wide-boundary or sensitive-CFAR variant was created because the existing boundaries already cover the requested posture ranges and CFAR/tracker changes would be speculative without live `geom_pts` evidence.

## Exact command to run next

```powershell
cd "C:\Users\UBESC\Desktop\Combined MMwave and RGB\mmwave_pose_ui_4ec2b00"

python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "cfg\ODS_6m_posture_tuned.cfg" `
  --out logs\posture_cfg_tuned_test `
  --enable-pose `
  --pose-model "model_experiments\outputs\ti_4class_clean_recording_robust_1600_fast\ti_pose_model.onnx" `
  --pose-log `
  --pose-debug `
  --pose-3d-labels `
  --pose-min-associated-points-for-inference 1 `
  --pose-allow-target-only `
  --pose-human-models `
  --pose-human-model-mode overlay_box `
  --pose-human-model-dir "ui_human_pose_models" `
  --pose-ground-plane `
  --pose-ground-z 0.0 `
  --pose-assoc-debug `
  --pose-assoc-method hybrid `
  --pose-assoc-nearest-radius-m 0.75 `
  --pose-assoc-min-points-good 3 `
  --pose-moving-require-translation `
  --pose-moving-translation-window 8 `
  --pose-moving-translation-min-m 0.25 `
  --pose-strong-stand-sit-near-margin 0.12 `
  --pose-strong-stand-sit-mid-margin 0.18 `
  --pose-strong-stand-sit-far-margin 0.25 `
  --pose-use-standing-baseline `
  --pose-standing-baseline-min-frames 20 `
  --pose-sitting-drop-near-m 0.20 `
  --pose-sitting-drop-mid-m 0.25 `
  --pose-sitting-drop-far-m 0.35 `
  --pose-sitting-drop-min-sit-prob 0.30 `
  --pose-use-sensor-calibration `
  --pose-sensor-height-m 2.0 `
  --pose-sensor-pitch-deg 15.0 `
  --pose-sensor-roll-deg 0.0 `
  --pose-sensor-yaw-deg 0.0 `
  --pose-floor-z-m 0.0 `
  --pose-range-near-max 2.0 `
  --pose-range-mid-max 4.0 `
  --enable-rgb-panel `
  --enable-rgb-posture `
  --rgb-repo "C:\Users\UBESC\Desktop\Combined MMwave and RGB\RGB Posture Estmation\Human-Falling-Detect-Tracks" `
  --rgb-source 0 `
  --rgb-camera-backend auto `
  --rgb-device cpu `
  --rgb-no-action `
  --rgb-show-skeleton `
  --rgb-show-detected `
  --combined-log `
  --combined-status-panel `
  --log-root "..\logs" `
  --rgb-log-keypoints
```

The pose calibration flags above mirror the cfg mount assumption: height `2.0` m and pitch/elevation `15.0` degrees. If the physical mount differs, change both the cfg `sensorPosition` line and the matching pose calibration flags for the test.

## Physical test protocol

1. Test original `ODS_6m_default.cfg` first.
2. Test `cfg\ODS_6m_posture_tuned.cfg` second.
3. Use the same sensor position, same chair, and same distance.
4. Stand 20 sec at 2 m.
5. Sit upright 20 sec.
6. Stand 20 sec.
7. Repeat at 3 m and 4 m.
8. Repeat with sensor tilted slightly downward if possible.

Record:

- `geom_pts`
- `quality`
- `range_zone`
- `stand_prob`
- `sit_prob`
- `geometry_decision`
- displayed pose
- false `MOVING` switches

## Validation

Run requested:

```powershell
python run_ti_style_visualizer.py --help
```

Status: passed. The command printed help successfully and confirmed the configured `--cfg` option plus the posture debug/calibration flags are available.

No live COM6/COM7 validation has been claimed.

## Recommended next physical test

Use the exact command above with `cfg\ODS_6m_posture_tuned.cfg`. The key pass/fail signal is whether `geom_pts` increases above zero and whether `geometry_decision=SITTING` appears during upright sitting after `baseline_ready=true`.
