# Paired round-phase receiver replay

Positive differences mean the candidate condition increases the named outcome.

| mechanism | comparison | correct peers | outcome | reference | candidate | difference (pp) | 95% CI (pp) | p | pairs |
|:---|:---|---:|:---|---:|---:|---:|:---:|---:|---:|
| target_laundering | relayed - direct | 1 | is_target | 8.33 | 21.67 | +13.33 | [+6.67, +20.00] | 0.0004 (q=0.0018) | 60 |
| target_laundering | relayed - direct | 1 | is_correct | 88.33 | 71.67 | -16.67 | [-25.04, -6.67] | 0.0030 (q=0.0108) | 60 |
| target_laundering | relayed - direct | 1 | is_unparsed | 3.33 | 5.00 | +1.67 | [-5.00, +8.33] | 0.8167 (q=0.8503) | 60 |
| target_laundering | relayed - direct | 2 | is_target | 0.00 | 18.33 | +18.33 | [+8.33, +30.00] | 0.0002 (q=0.0018) | 60 |
| target_laundering | relayed - direct | 2 | is_correct | 90.00 | 70.00 | -20.00 | [-36.67, -5.00] | 0.0142 (q=0.0426) | 60 |
| target_laundering | relayed - direct | 2 | is_unparsed | 6.67 | 11.67 | +5.00 | [-3.33, +15.00] | 0.3902 (q=0.5852) | 60 |
| target_laundering | relayed - direct | all | is_target | 4.17 | 20.00 | +15.83 | [+9.17, +23.33] | 0.0002 (q=0.0018) | 120 |
| target_laundering | relayed - direct | all | is_correct | 89.17 | 70.83 | -18.33 | [-28.33, -8.33] | 0.0004 (q=0.0018) | 120 |
| target_laundering | relayed - direct | all | is_unparsed | 5.00 | 8.33 | +3.33 | [-3.33, +10.00] | 0.3756 (q=0.5852) | 120 |
| history_maturity | deliberated - initial | 1 | is_target | 20.00 | 15.00 | -5.00 | [-16.67, +6.67] | 0.4754 (q=0.6405) | 60 |
| history_maturity | deliberated - initial | 1 | is_correct | 68.33 | 65.00 | -3.33 | [-15.00, +10.00] | 0.6297 (q=0.7557) | 60 |
| history_maturity | deliberated - initial | 1 | is_unparsed | 10.00 | 20.00 | +10.00 | [+1.67, +20.00] | 0.0482 (q=0.1239) | 60 |
| history_maturity | deliberated - initial | 2 | is_target | 15.00 | 13.33 | -1.67 | [-10.00, +6.67] | 0.8369 (q=0.8503) | 60 |
| history_maturity | deliberated - initial | 2 | is_correct | 70.00 | 80.00 | +10.00 | [-3.33, +25.00] | 0.1798 (q=0.3596) | 60 |
| history_maturity | deliberated - initial | 2 | is_unparsed | 15.00 | 6.67 | -8.33 | [-20.00, +3.33] | 0.1794 (q=0.3596) | 60 |
| history_maturity | deliberated - initial | all | is_target | 17.50 | 14.17 | -3.33 | [-9.17, +2.50] | 0.3128 (q=0.5630) | 120 |
| history_maturity | deliberated - initial | all | is_correct | 69.17 | 72.50 | +3.33 | [-5.83, +12.50] | 0.4982 (q=0.6405) | 120 |
| history_maturity | deliberated - initial | all | is_unparsed | 12.50 | 13.33 | +0.83 | [-6.67, +9.17] | 0.8503 (q=0.8503) | 120 |

The receiver is anonymous and never sees the provenance condition label. The laundering intervention therefore changes real rationale text generated at different positions in an observed attack chain; it does not randomize a visible source identity.
