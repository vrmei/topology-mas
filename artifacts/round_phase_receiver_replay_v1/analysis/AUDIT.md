# Replay audit

- Requests/results: 480/480
- Technical failures: 0
- Pair invariant violations: 0
- Finish reasons: {'stop': 469, 'length': 11}

## Treatment length audit

```text
                   reference_tokens  candidate_tokens  reference_chars  candidate_chars
history_maturity              170.0             217.5            609.5            856.5
target_laundering              44.0             224.5            133.0            898.0
```

## Stop-only sensitivity

```text
        mechanism correct_peer_count  stop_only_pairs  target_difference  correct_difference
target_laundering                  1               58           0.137931           -0.137931
target_laundering                  2               58           0.189655           -0.206897
target_laundering                all              116           0.163793           -0.172414
 history_maturity                  1               56          -0.071429           -0.017857
 history_maturity                  2               57          -0.035088            0.070175
 history_maturity                all              113          -0.053097            0.026549
```
