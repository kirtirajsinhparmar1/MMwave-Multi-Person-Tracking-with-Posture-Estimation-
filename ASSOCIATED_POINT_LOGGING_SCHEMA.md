# Associated Point Logging Schema

## 1. Purpose

`mmwave_associated_points.csv` records the raw radar points associated to each tracked target/TID in each frame. The file is intended for RadarPostureNet-v2-full training, where the model needs target-centered point-cloud geometry rather than only old pose probabilities or track metadata.

This logger is non-invasive. It does not change tracking, posture inference, smoothing, UI display, human model rendering, RGB processing, cfg files, or model outputs.

## 2. When The Log Is Enabled

The log is written only when the TI-style UI is launched with:

```powershell
--enable-pose --pose-log-associated-points
```

If `--pose-log-associated-points` is omitted, no associated point-cloud CSV is created.

## 3. Output File Path

The file is created under the `--out` session directory:

```text
<out>\mmwave_associated_points.csv
```

The default format is CSV. Parquet is not currently implemented.

## 4. Full Column Schema

```csv
session_id,frame,timestamp_s,tid,track_index,point_index,association_source,association_confidence,point_x_m,point_y_m,point_z_m,point_range_m,point_azimuth_deg,point_elevation_deg,point_doppler_mps,point_snr,point_noise,point_quality,target_x_m,target_y_m,target_z_m,target_range_m,target_azimuth_deg,target_elevation_deg,target_vx_mps,target_vy_mps,target_vz_mps,relative_x_m,relative_y_m,relative_z_m,relative_range_m,relative_radial_m,relative_lateral_m,height_above_ground_m,is_valid_point,geom_pts_for_tid,points_total_frame,quality_label_for_tid,old_display_pose,old_model_stand_prob,old_model_sit_prob,old_model_move_prob,old_model_lie_prob,old_model_fall_prob
```

## 5. Field Definitions

| Field | Definition |
| --- | --- |
| `session_id` | Session identifier. Uses `--session-id` when available, otherwise the `--out` folder name. |
| `frame` | Radar frame number from the parsed TI output dictionary. |
| `timestamp_s` | Host wall-clock timestamp in seconds when the pose manager logged the frame. |
| `tid` | Tracked target ID. |
| `track_index` | Point association track index when available; nearest-neighbor associations use `-1`. |
| `point_index` | Point row index inside the frame point cloud. |
| `association_source` | Method that supplied the associated points. |
| `association_confidence` | Numeric confidence assigned by source: target-index sources `1.0`, nearest sources `0.5`, unknown/unassociated sources `0.0`. |
| `point_x_m`, `point_y_m`, `point_z_m` | Radar point coordinates in meters. |
| `point_range_m` | Euclidean range of the point from the radar. |
| `point_azimuth_deg` | `atan2(x, y)` in degrees. |
| `point_elevation_deg` | `atan2(z, sqrt(x^2 + y^2))` in degrees. |
| `point_doppler_mps` | Point Doppler in meters per second when provided by the parser. |
| `point_snr` | Point SNR when provided by the parser. |
| `point_noise` | Point noise if present in the parsed point row; otherwise blank. |
| `point_quality` | `OK`, `LOW_SNR`, or `NO_POINTS` for summary rows. |
| `target_x_m`, `target_y_m`, `target_z_m` | Target centroid position from `trackData`. |
| `target_range_m`, `target_azimuth_deg`, `target_elevation_deg` | Range and angles computed from target coordinates. |
| `target_vx_mps`, `target_vy_mps`, `target_vz_mps` | Target velocity from `trackData`. |
| `relative_x_m`, `relative_y_m`, `relative_z_m` | Target-centered point coordinates, computed as point minus target. |
| `relative_range_m` | Euclidean range of the target-centered point. |
| `relative_radial_m` | Same as `relative_y_m` under the current TI display convention. |
| `relative_lateral_m` | Same as `relative_x_m` under the current TI display convention. |
| `height_above_ground_m` | `point_z_m - ground_z`. |
| `is_valid_point` | `1` for a real associated point row; `0` for an optional no-points summary row. |
| `geom_pts_for_tid` | Total associated points for that TID/frame before max-point selection. |
| `points_total_frame` | Total point-cloud rows in the frame. |
| `quality_label_for_tid` | Existing pose quality label for that TID/frame, such as `OK`, `LOW_POINTS`, or `NO_POINTS`. |
| `old_display_pose` | Existing displayed posture label. This is logged for analysis only and is not ground truth. |
| `old_model_*_prob` | Existing old-model probability columns when available. Missing model classes remain blank. |

## 6. Coordinate Convention

The logger follows the existing TI people-tracking coordinate convention used by this project: `x` is lateral, `y` is forward/range, and `z` is vertical. Azimuth is computed as `atan2(x, y)` and elevation as `atan2(z, sqrt(x^2 + y^2))`.

Target-centered coordinates are:

```text
relative_x_m = point_x_m - target_x_m
relative_y_m = point_y_m - target_y_m
relative_z_m = point_z_m - target_z_m
```

`ground_z` comes from the existing pose ground setting. If no custom value is provided, the runtime default is `0.0`.

## 7. Association Source Definitions

Known source values:

| Value | Meaning |
| --- | --- |
| `target_index` | Points matched through TLV target indexes or point track-index columns. |
| `nearest` | Points selected by proximity to the target when target-index association is unavailable or disabled. |
| `hybrid_target_index` | Hybrid mode used target-index association. |
| `hybrid_nearest` | Hybrid mode fell back to nearest association. |
| `unassociated` | Reserved for explicitly unassociated rows. |
| `unknown` | Source was unavailable. |
| `auto_none` | Existing auto mode found no target-index or nearest points. |
| `index` | Existing synonym for index-based association if encountered. |

## 8. Missing Value Policy

Columns are never omitted. If a field is unavailable, the logger writes a blank CSV value. This is expected for `point_noise` on compressed point-cloud rows and for old model classes that are not emitted by the current model.

When no associated points exist for a TID/frame, the logger writes one summary row with `is_valid_point=0`, `geom_pts_for_tid=0`, and `point_quality=NO_POINTS`. This keeps dropout evidence without creating one empty row per missing point.

## 9. Performance Considerations

The logger is disabled by default. When enabled, it buffers rows and flushes periodically. The default per-TID cap is 64 points per frame:

```powershell
--pose-associated-points-max-per-tid 64
```

If a TID has more than the cap, selection keeps a mix of high-SNR points, high and low vertical points, and spatially spread remaining points. It does not use the old top-5 highest-z feature selection.

Row write failures are caught so logging does not crash the UI.

## 10. How This Supports RadarPostureNet-v2-full

The full model needs per-TID point sequences in a target-centered coordinate system. This log provides:

- frame and TID grouping,
- per-point x/y/z/doppler/SNR fields,
- target-centered relative coordinates,
- association source and quality evidence,
- no-points/dropout rows,
- old model outputs as auxiliary teacher/input columns, not labels.

Labels should still come from user-provided segment protocols, not from `old_display_pose`.
