# RGB Teacher Audit Report

RGB teacher status: PARTIALLY USABLE

The existing RGB files can assist offline label review, sync checks, transition discovery, and weak torso-angle analysis. They are not treated as ground truth by this report because lower-body keypoints are missing, camera perspective is not calibrated, two-person identity alignment is not verified, and lean-forward/lean-back direction requires manual review.

## Summary

- Sessions audited: 9
- Sessions with RGB video: 9
- Sessions with RGB keypoints: 9
- Sessions with sync_index.csv: 9
- Mean keypoint score across sessions: 0.720
- Mean candidate STANDING/SITTING label rate: 0.000

## Decision

RGB teacher is partially usable now: it can align synchronized video/keypoint tracks to mmWave time and support manual review or weak torso-angle analysis, but missing knee/ankle coverage prevents robust automatic frame-level posture/subpose labels.

## Notes

- Shoulders, hips, knees, and ankles are required for torso/body-ratio features.
- Two-person sessions can often distinguish left/right tracks by bbox x-position, but this is not the same as verified mmWave TID alignment.
- Lean-forward vs lean-back is not reliably inferred from front-facing RGB keypoints alone.
