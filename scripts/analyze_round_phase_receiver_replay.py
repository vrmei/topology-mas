#!/usr/bin/env python3
"""Analyze paired target-laundering and history-maturity receiver replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRASTS = {
    "target_laundering": ("direct", "relayed"),
    "history_maturity": ("initial", "deliberated"),
}
OUTCOMES = ("is_target", "is_correct", "is_unparsed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def task_bootstrap(values: pd.Series, replicates: int, seed: int) -> tuple[float, float, float]:
    array = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    lower = int((draws <= 0).sum())
    upper = int((draws >= 0).sum())
    p = min(1.0, 2 * (min(lower, upper) + 1) / (replicates + 1))
    return float(lo), float(hi), float(p)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(float)
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(raw) / np.arange(1, len(raw) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted((args.run_dir / "results").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or len(frame) != status["expected"]:
        raise ValueError(f"run is not complete: {status}")
    if frame.request_id.duplicated().any():
        raise ValueError("duplicate request IDs")
    result_rows = []
    paired_rows = []
    for mechanism, (reference, candidate) in CONTRASTS.items():
        selected = frame[frame.mechanism.eq(mechanism)]
        peer_levels: list[int | str] = [*sorted(selected.correct_peer_count.unique()), "all"]
        for correct_peer_count in peer_levels:
            group = selected if correct_peer_count == "all" else selected[selected.correct_peer_count.eq(correct_peer_count)]
            for outcome_index, outcome in enumerate(OUTCOMES):
                pivot = group.pivot(index=["pair_id", "task_id"], columns="condition", values=outcome)
                if reference not in pivot or candidate not in pivot or pivot[[reference, candidate]].isna().any().any():
                    raise ValueError(f"incomplete pair for {mechanism}, peers={correct_peer_count}, {outcome}")
                pivot["difference"] = pivot[candidate].astype(float) - pivot[reference].astype(float)
                per_task = pivot.groupby("task_id").difference.mean()
                lo, hi, p = task_bootstrap(
                    per_task,
                    args.bootstrap_reps,
                    args.seed + 1_000 * (99 if correct_peer_count == "all" else int(correct_peer_count))
                    + 10 * outcome_index + (0 if mechanism == "target_laundering" else 100_000),
                )
                result_rows.append({
                    "mechanism": mechanism,
                    "reference_condition": reference,
                    "candidate_condition": candidate,
                    "correct_peer_count": correct_peer_count,
                    "outcome": outcome,
                    "pairs": len(pivot),
                    "tasks": len(per_task),
                    "reference_rate": float(pivot[reference].mean()),
                    "candidate_rate": float(pivot[candidate].mean()),
                    "paired_difference": float(pivot.difference.mean()),
                    "task_ci95_low": lo,
                    "task_ci95_high": hi,
                    "task_bootstrap_p": p,
                })
                for task_id, value in per_task.items():
                    paired_rows.append({
                        "mechanism": mechanism,
                        "correct_peer_count": correct_peer_count,
                        "outcome": outcome,
                        "task_id": task_id,
                        "task_mean_difference": value,
                    })
    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result_rows)
    summary["task_bootstrap_q_bh"] = benjamini_hochberg(summary.task_bootstrap_p)
    summary.to_csv(args.out / "paired_effects.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.out / "paired_task_effects.csv", index=False)
    frame.drop(columns=["raw_output"], errors="ignore").to_csv(
        args.out / "outcomes_compact.csv.gz", index=False, compression="gzip"
    )
    lines = [
        "# Paired round-phase receiver replay", "",
        "Positive differences mean the candidate condition increases the named outcome.", "",
        "| mechanism | comparison | correct peers | outcome | reference | candidate | difference (pp) | 95% CI (pp) | p | pairs |",
        "|:---|:---|---:|:---|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.mechanism} | {row.candidate_condition} - {row.reference_condition} | "
            f"{row.correct_peer_count} | {row.outcome} | {100*row.reference_rate:.2f} | "
            f"{100*row.candidate_rate:.2f} | {100*row.paired_difference:+.2f} | "
            f"[{100*row.task_ci95_low:+.2f}, {100*row.task_ci95_high:+.2f}] | "
            f"{row.task_bootstrap_p:.4f} (q={row.task_bootstrap_q_bh:.4f}) | {row.pairs} |"
        )
    lines += ["", "The receiver is anonymous and never sees the provenance condition label. The laundering intervention therefore changes real rationale text generated at different positions in an observed attack chain; it does not randomize a visible source identity."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "manifest.json").write_text(
        json.dumps({
            "analysis": "round-phase-receiver-replay-analysis-v1",
            "run_dir": str(args.run_dir),
            "requests": len(frame),
            "tasks": int(frame.task_id.nunique()),
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
