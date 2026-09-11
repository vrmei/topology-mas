# Round-conditioned CTOU analysis

T=1 and T=2 are prefix endpoints from the recorded T=3 trajectories, not independent reruns.

## Primary readout contrasts

Exact previous state and incoming CTOU counts are held fixed. Positive values mean the transition is more likely in Round 2 than Round 1.

| transition | n | R1 | R2 | difference (pp) | task 95% CI (pp) | task q | graph q | matched weight |
|:---|---:|---:|---:|---:|:---:|---:|---:|---:|
| C_to_O | 5 | 1.42 | 1.41 | -0.01 | [-0.33, +0.29] | 0.9435 | 0.9675 | 8075 |
| C_to_O | 6 | 1.29 | 1.35 | +0.06 | [-0.19, +0.37] | 0.8369 | 0.9062 | 8603 |
| C_to_O | 7 | 1.16 | 1.46 | +0.30 | [+0.02, +0.61] | 0.1599 | 0.0040 | 15275 |
| C_to_O | 8 | 1.09 | 1.16 | +0.07 | [-0.21, +0.36] | 0.8369 | 0.9062 | 16662 |
| C_to_T | 5 | 4.72 | 1.51 | -3.22 | [-4.43, -2.15] | 0.0010 | 0.0010 | 8075 |
| C_to_T | 6 | 3.86 | 1.07 | -2.79 | [-3.86, -1.83] | 0.0010 | 0.0010 | 8603 |
| C_to_T | 7 | 3.34 | 0.81 | -2.52 | [-3.53, -1.63] | 0.0010 | 0.0010 | 15275 |
| C_to_T | 8 | 2.99 | 0.68 | -2.31 | [-3.39, -1.42] | 0.0010 | 0.0010 | 16662 |
| C_to_notC | 5 | 10.96 | 4.71 | -6.26 | [-7.69, -4.87] | 0.0010 | 0.0010 | 8075 |
| C_to_notC | 6 | 9.46 | 3.73 | -5.73 | [-7.09, -4.49] | 0.0010 | 0.0010 | 8603 |
| C_to_notC | 7 | 8.65 | 3.34 | -5.30 | [-6.72, -4.01] | 0.0010 | 0.0010 | 15275 |
| C_to_notC | 8 | 8.02 | 2.86 | -5.17 | [-6.67, -3.96] | 0.0010 | 0.0010 | 16662 |
| OU_to_C | 5 | 29.20 | 33.17 | +3.97 | [+0.81, +8.74] | 0.0560 | 0.0040 | 1832 |
| OU_to_C | 6 | 28.71 | 31.32 | +2.60 | [-0.25, +6.91] | 0.1026 | 0.0227 | 1850 |
| OU_to_C | 7 | 28.62 | 31.50 | +2.87 | [+0.07, +7.20] | 0.0920 | 0.0040 | 3201 |
| OU_to_C | 8 | 31.85 | 31.49 | -0.36 | [-2.94, +2.86] | 0.8606 | 0.6437 | 3367 |
| O_to_C | 5 | 21.73 | 18.41 | -3.32 | [-5.41, -0.23] | 0.0453 | 0.0093 | 1299 |
| O_to_C | 6 | 19.42 | 16.86 | -2.56 | [-4.92, +0.20] | 0.0760 | 0.0790 | 1337 |
| O_to_C | 7 | 21.22 | 17.96 | -3.26 | [-6.10, -0.43] | 0.0453 | 0.0020 | 2374 |
| O_to_C | 8 | 22.74 | 17.47 | -5.28 | [-9.36, -1.89] | 0.0080 | 0.0020 | 2427 |
| U_to_C | 5 | 60.94 | 69.31 | +8.38 | [+1.21, +15.32] | 0.0520 | 0.0340 | 169 |
| U_to_C | 6 | 69.70 | 74.82 | +5.12 | [-2.71, +13.90] | 0.2812 | 0.1599 | 132 |
| U_to_C | 7 | 63.73 | 74.34 | +10.61 | [+2.66, +19.22] | 0.0280 | 0.0040 | 281 |
| U_to_C | 8 | 63.60 | 66.10 | +2.50 | [-2.38, +7.22] | 0.2939 | 0.2729 | 261 |

## Interpretation boundary

- A matched round contrast is observational. It can identify residual association after CTOU controls, not a causal effect of a round label.
- The symmetric decomposition is restricted to CTOU cells supported in both rounds; unmatched-support coverage is reported separately in the CSV.
- GPU replay is gated by the preregistered practical and uncertainty criteria in docs/round_phase_mechanism_plan.md.
