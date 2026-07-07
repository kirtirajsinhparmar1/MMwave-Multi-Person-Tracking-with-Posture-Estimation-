# Old Posture Architecture Summary

The old runtime builds a TI-style 176-feature window per TID from target metadata and the selected highest-z associated points. The feature vector is 22 channels over 8 frames: target z/velocity/acceleration plus five selected point y/z/SNR triplets. The ONNX model in `ti_4class_clean_recording_robust_1600_fast` predicts STANDING, SITTING, LYING, and FALLING.

Runtime posture is not the raw model output. `ti_style_pose_overlay.py` applies per-TID smoothing, confidence thresholds, motion overrides, standing/sitting stability counts, height-drop/fall gates, sitting gates, range-zone logic, and a relative sitting gate before deciding the displayed posture and human model asset.

Known limitations: the old model was trained on prior TI-style feature windows, not the current user-collected standing/sitting session registry; current logs often record only model probabilities and track metadata, not raw associated point tensors; absolute/range-sensitive behavior caused standing/sitting confusion at specific distances; and UI rendering/dropout evidence is mixed into runtime behavior rather than modeled as a separate reliability output.

A new architecture is needed to use user-provided protocols as ground truth, validate by held-out sessions/positions/person counts, separate posture from visibility/reliability, and avoid absolute-coordinate shortcuts as the main posture signal.
