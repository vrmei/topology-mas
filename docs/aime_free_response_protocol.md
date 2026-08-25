# AIME free-response task boundary

## Frozen normal-agent interface

Normal AIME agents receive no candidate answers. Their model-visible user
message contains only the problem statement. A model-independent system message
requires a final integer from 000 through 999 in the form:

```text
FINAL_ANSWER: \boxed{ddd}
```

The evaluator stores the gold answer outside the prompt and normalizes leading
zeros, so `028` and `28` denote the same answer. It parses only an explicit
`FINAL_ANSWER` or `\boxed{}` marker and never guesses from an unmarked trailing
number.

## Minimal task record

```json
{
  "family_id": "2025_AIME_I_P04",
  "task_id": "2025_AIME_I_P04_M1",
  "mutation_type": "parameter",
  "problem": "...",
  "gold_answer": 129
}
```

Only `problem` is model-visible. Extra fields are forbidden by the loader.

## Attack separation

Target-error records are separate evaluator/attacker artifacts. A normal task
record cannot contain `target_error` or `target_rationale`. A future attack uses
exactly one independently verified target error per task. Any other parseable
integer unequal to the gold and target answers is classified as O.

## Current experiment boundary

The current calibration and clean-utility experiment uses only the frozen 30
original AIME problems. Parameter or structural mutations are out of scope
until the original-problem solve-rate distribution has been measured.
