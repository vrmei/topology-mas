# Original AIME Round-0 utility calibration

## qwen3-4b

- U0: `0.440` (task-bootstrap 95% CI `0.290`–`0.593`)
- Parse rate: `0.820`
- Accuracy conditional on a valid answer: `0.537`
- Length-stop rate: `0.157`
- Task bands: `{'floor_0_to_0.1': 12, 'informative_0.2_to_0.8': 11, 'ceiling_0.9_to_1.0': 7}`
- Contest utility: `{'AIME_I': 0.4533333333333333, 'AIME_II': 0.4266666666666667}`

## llama31-8b

- U0: `0.000` (task-bootstrap 95% CI `0.000`–`0.000`)
- Parse rate: `0.387`
- Accuracy conditional on a valid answer: `0.000`
- Length-stop rate: `0.560`
- Task bands: `{'floor_0_to_0.1': 30, 'informative_0.2_to_0.8': 0, 'ceiling_0.9_to_1.0': 0}`
- Contest utility: `{'AIME_I': 0.0, 'AIME_II': 0.0}`

## Paired task-level contrasts

- llama31-8b − qwen3-4b: `-0.440` (95% CI `-0.593`–`-0.297`)
