# GSM8K numeric summary protocol v1

This protocol reruns the existing homogeneous Llama/GSM8K graph experiment with
the same communication abstraction as the Qwen/AIME summary experiment.  The
benchmark-specific answer contract is numeric rather than AIME's three-digit
integer contract.

## Frozen model settings

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Private solve: `temperature=0.6`, `top_p=0.9`, no `top_k`, maximum 20,000 tokens
- Public summary: `temperature=0`, `top_p=1`, `top_k=-1`, one model call,
  maximum 4,096 tokens
- Server context: 98,304 tokens
- No peer-message truncation and no full/summary switching by graph density

The private solve and public summary system prompts are frozen as
`NUMERIC_SOLVE_SYSTEM_PROMPT` and `NUMERIC_SUMMARY_SYSTEM_PROMPT` in
`src/topology_mas/execution/numeric_summary_protocol.py`.

## Communication contract

At Round 0 each normal node receives only the problem.  At every later round a
normal receiver sees:

1. the original problem;
2. its own previous full solution;
3. every incoming neighbor's validated public summary.

The full solution is never sent across an edge.  The summary model cannot control
the terminal state: Python extracts the full solution's explicit numeric answer
and appends either the same `FINAL_ANSWER: \boxed{...}` or `FINAL_ANSWER:
UNPARSED`.  A summary may not invent, discard, or change this state.

## Frozen Round-0 sampling

For each of the fixed 50 GSM8K tasks:

1. independently generate 80 solve-then-summary records (`K80`);
2. retain C, O, and U without state-based filtering;
3. deterministically sample 64 record IDs without replacement (`K64`);
4. for each task×graph cell, deterministically sample five distinct records from
   that task's K64 pool and permute them over the five structural node IDs;
5. pair that cell's clean run and its four attacker-position runs on the same
   five records and node assignment.

Thus K64 is the true preselection pool.  Five is only the number of node states
materialized for one n=5 cell, not the pool size.

## GPU task pool

K80 generation uses a global request pool.  Each equivalent vLLM endpoint exposes
a fixed number of client-side slots.  Any completed request immediately returns
its endpoint slot to a shared queue, and the next pending solve or summary uses the
next available slot.  Formal graph execution uses the same idea at cell level:
finished graph cells release capacity immediately.  Scheduling never changes the
request seed, prompt, graph assignment, or K64 selection.

## Planned graph cells

- `n=5`, readout node 4, `H=3`
- `m=4,...,15`: five rooted-nonisomorphic graphs per level
- `m=16`: the unique complete graph
- 61 graphs × 50 tasks = 3,050 clean cells
- 61 graphs × 50 tasks × four non-readout attacker positions = 12,200 attack cells

All runs use summary-only cross-node messages and fixed target attacks.  No result
from the older full-rationale Llama/GSM8K experiment is reused as a generation in
this protocol.
