# RGB Assisted Label Report

RGB-assisted labels are offline candidates only. They are not ground truth and are not used by runtime posture decisions.

- Candidate rows: 57489
- Known STANDING/SITTING candidate rows: 0
- Disagreement rows: 57489

The current RGB keypoints usually include shoulders and hips, but knees/ankles are absent in these sessions. Therefore most rows remain UNKNOWN or UNCERTAIN and should be used for manual review, sync checks, and transition discovery rather than automatic posture supervision.
