# Sparse Far-Range Data Collection Protocol

## Purpose

Collect full associated-point radar posture data that teaches RadarPostureNet-v2 Sparse-MoE how posture evidence degrades from 3m through 6m. These sessions should keep tracking, RGB, chair geometry, sensor mounting, and labeling conditions consistent so the model can learn sparsity and distance effects instead of collection artifacts.

This protocol does not change TI cfg files, runtime posture logic, RGB code, or the existing ONNX runtime path.

## Required Logging

Use the current TI-style visualizer/capture path with:

- RGB enabled.
- Associated point logging enabled.
- Pose logging enabled when available.
- The same chair for all sitting sessions.
- The same sensor height, pitch, yaw, and room layout for each collection block.
- Manual notes for target disappearance, re-acquisition, wrong TID, or obvious chair/person occlusion.

The expected files per session are:

- `mmwave_associated_points.csv`
- `mmwave_pose.csv`
- `mmwave_tracks.csv`
- RGB recordings or synchronized RGB-derived labels when available.
- Session notes that identify distance, position, people count, expected pose, and expected subpose.

## Sessions

Collect these sessions first:

```text
pc_far_standing_center_3to6_01
pc_far_sitting_leanback_center_3to6_01
pc_far_sitting_upright_center_3to6_01
pc_far_sitting_leanforward_center_3to6_01
pc_far_standing_left_3to6_01
pc_far_standing_right_3to6_01
pc_far_sitting_leanback_left_3to6_01
pc_far_sitting_leanback_right_3to6_01
pc_far_two_person_standing_lr_3to6_01
pc_far_two_person_sitting_leanback_lr_3to6_01
```

## Distances

Each session should include:

```text
3m, 4m, 5m, 6m
```

Use measured floor marks from the radar origin. Keep the subject centered on the mark for CENTER sessions and use matched left/right lateral offsets for LEFT and RIGHT sessions.

## Timing

For each distance:

```text
60 seconds stable posture capture
10 seconds transition to the next distance
```

Do not merge the transition into the stable segment labels. Mark transition ranges as `UNKNOWN` or exclude them from posture training windows.

## Per-Session Notes

For every session, record:

- Session ID.
- Date and time.
- Sensor height, pitch, and approximate room layout.
- Chair identity and chair position for sitting captures.
- Distance marks used.
- Expected pose and expected subpose.
- Position: `CENTER`, `LEFT`, or `RIGHT`.
- People count.
- Any disappearance/re-acquisition intervals.
- Any TID swaps or competing objects.
- Any unsafe or invalid segments.

## Command Template

Use the current working run command with associated-point logging enabled. Keep cfg unchanged:

```powershell
python run_ti_style_visualizer.py `
  --cli COM7 `
  --data COM6 `
  --cfg "C:\Users\UBESC\Desktop\radar_toolbox_4_00_00_05\source\ti\examples\Industrial_and_Personal_Electronics\People_Tracking\3D_People_Tracking\chirp_configs\ODS_6m_default.cfg" `
  --out logs\<session_id> `
  --enable-pose `
  --pose-log `
  --pose-log-associated-points
```

If the local launcher uses a different exact flag name for associated-point logging, use the implemented associated-point logging flag from the current runner. The required output is `mmwave_associated_points.csv`.

## Collection Order

1. Run one `pc_far_standing_center_3to6_01` smoke pass and verify `mmwave_associated_points.csv`, `mmwave_pose.csv`, and `mmwave_tracks.csv` are populated.
2. Inspect one minute of logs for nonzero associated points and stable TID continuity.
3. Collect all CENTER single-person sessions.
4. Collect LEFT and RIGHT single-person sessions.
5. Collect two-person sessions only after single-person logs look valid.
6. Run `analysis\build_sparse_moe_dataset_preview.py` after the first block to verify tensor construction.

## Quality Gates

Accept a session only if:

- The expected files exist.
- The session has stable frame progression.
- At least one target is tracked through most stable segments.
- Distance labels are unambiguous.
- Transitions are labeled or excluded.
- Manual disappearance notes are available for weak far-range intervals.

Reject or mark partial if:

- COM capture was interrupted.
- The subject moved away from the marked distance during a stable segment.
- The chair changed between sitting subtypes.
- RGB was missing when the session was intended for RGB-assisted labeling.
- TID swaps make the target identity unrecoverable.

## Minimum First Block

The first useful Sparse-MoE training block should include at least:

- All four CENTER sessions.
- LEFT and RIGHT standing sessions.
- LEFT and RIGHT lean-back sitting sessions.
- One two-person standing session.
- One two-person lean-back sitting session.

This gives the model explicit examples of dense 3m, sparse 4m/5m, edge 6m, lateral position effects, and two-person sparsity/tracking pressure.
