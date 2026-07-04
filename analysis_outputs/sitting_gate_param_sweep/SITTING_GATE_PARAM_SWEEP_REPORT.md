# Sitting Gate Parameter Sweep Report

## Regression Frame Mining

Standing false-SITTING changed frames: 36
Near-range (<2.5m) share: 0.556
NO_POINTS share: 0.111
Mean stand_prob: 0.305
Mean sit_prob: 0.603
Mean sit-minus-stand margin: 0.297
MOVING-guard reason share: 0.000

## Sweep

Candidates evaluated: 8000
Acceptable candidates: 80

## Best Safe Candidate

| range_min_for_relative_gate_m | soft_sitting_min_prob | relative_sitting_margin | relative_sitting_frames | standing_veto_prob | standing_veto_margin | full_standing_1m_old | full_standing_1m_candidate | full_standing_2m_old | full_standing_2m_candidate | full_standing_3m_old | full_standing_3m_candidate | full_standing_4m_old | full_standing_4m_candidate | full_sitting_1m_old | full_sitting_1m_candidate | full_sitting_2m_old | full_sitting_2m_candidate | full_sitting_3m_old | full_sitting_3m_candidate | full_sitting_4m_old | full_sitting_4m_candidate | full_standing_false_sitting_rate_old | full_standing_false_sitting_rate_candidate | full_sitting_false_standing_rate_old | full_sitting_false_standing_rate_candidate | full_pose_switch_count_old | full_pose_switch_count_candidate | default_ab_sitting_2m_old | default_ab_sitting_2m_candidate | default_ab_sitting_3m_old | default_ab_sitting_3m_candidate | default_ab_sitting_4m_old | default_ab_sitting_4m_candidate | default_ab_standing_false_sitting_rate_old | default_ab_standing_false_sitting_rate_candidate | default_ab_sitting_false_standing_rate_old | default_ab_sitting_false_standing_rate_candidate | default_ab_pose_switch_count_old | default_ab_pose_switch_count_candidate | acceptable | failure_reasons | max_default_ab_sitting_3m_4m_gain | max_full_sitting_3m_4m_gain | total_switch_increase_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 0.55 | 0.12 | 16 | 0.5 | 0.05 | 0.8306748466257668 | 0.8306748466257668 | 0.9067909454061251 | 0.9067909454061251 | 0.8859060402684564 | 0.8859060402684564 | 1.0 | 1.0 | 0.6476964769647696 | 0.6476964769647696 | 0.6770428015564203 | 0.6770428015564203 | 0.593103448275862 | 0.593103448275862 | 0.003898635477582846 | 0.05263157894736842 | 0.0038691523039043265 | 0.0038691523039043265 | 0.43614202437731847 | 0.42289348171701113 | 50 | 56 | 0.807843137254902 | 0.807843137254902 | 0.38741258741258744 | 0.3902097902097902 | 0.8839390386869871 | 0.9320046893317703 | nan | nan | 0.2810567734682406 | 0.26840921866216977 | 42 | 45 | True |  | 0.04806565064478319 | 0.04873294346978557 | 0.09782608695652174 |

## Acceptance Criteria

A candidate must keep every standing segment within 0.5 percentage points of old accuracy, keep standing false-SITTING increase within 0.5 percentage points, avoid sitting_2m regression over 1 point, improve sitting_3m or sitting_4m by at least 3 points, and keep switch-count growth within 10%.