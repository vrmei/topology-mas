# Target provenance diagnostic

Direct and relayed target messages are compared within the same previous state, round, incoming CTOU counts, and receiver scope.

| n | scope | P(next=T), direct | P(next=T), relayed | difference (pp) | 95% CI (pp) | matched rows |
|---:|:---|---:|---:|---:|:---:|---:|
| 5 | all | 5.74 | 23.32 | -17.58 | [-21.95, -13.11] | 17196 |
| 5 | internal | 2.79 | 19.46 | -16.67 | [-23.56, -10.14] | 4909 |
| 5 | readout | 7.80 | 25.64 | -17.84 | [-23.50, -12.18] | 6921 |
| 6 | all | 1.99 | 15.35 | -13.36 | [-17.46, -9.27] | 20103 |
| 6 | internal | 2.00 | 17.66 | -15.66 | [-21.08, -10.66] | 7396 |
| 6 | readout | 1.70 | 12.83 | -11.14 | [-17.19, -5.72] | 8125 |
| 7 | all | 1.38 | 16.12 | -14.74 | [-18.19, -10.96] | 40327 |
| 7 | internal | 1.45 | 17.59 | -16.15 | [-21.10, -11.28] | 19155 |
| 7 | readout | 0.99 | 13.51 | -12.52 | [-16.40, -8.27] | 11188 |
| 8 | all | 1.13 | 12.56 | -11.42 | [-13.93, -8.75] | 52267 |
| 8 | internal | 1.47 | 13.63 | -12.16 | [-15.74, -8.47] | 26065 |
| 8 | readout | 0.50 | 10.38 | -9.88 | [-12.95, -6.78] | 14373 |

This is an observational matched comparison. Because the receiver is not shown a trusted attacker label, 'origin' mainly indexes how the target rationale was generated and transformed, not an independently randomized source identity.
