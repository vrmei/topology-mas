#!/usr/bin/env python3
"""Analyze one or more original-AIME Round-0 utility calibration runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from topology_mas.experiments.aime_utility import atomic_json


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20250825)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def task_bootstrap_interval(
    rates: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [
        fmean(rates[rng.randrange(len(rates))] for _ in rates)
        for _ in range(samples)
    ]
    return quantile(draws, 0.025), quantile(draws, 0.975)


def summarize_run(
    *,
    label: str,
    run_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"run {label!r} is not complete")
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")
    if len(outcomes) != int(status["expected"]):
        raise ValueError(f"run {label!r} outcome count differs from manifest")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row["task_id"])].append(row)
    if set(grouped) != set(status["task_ids"]):
        raise ValueError(f"run {label!r} task IDs differ from manifest")

    per_task: list[dict[str, Any]] = []
    for task_id in status["task_ids"]:
        rows = grouped[str(task_id)]
        expected_replicates = int(status["replicates"])
        if len(rows) != expected_replicates:
            raise ValueError(f"{label}/{task_id} does not have all replicates")
        successes = sum(bool(row["is_correct"]) for row in rows)
        parsed = sum(bool(row["is_parsed"]) for row in rows)
        per_task.append(
            {
                "model_label": label,
                "task_id": task_id,
                "contest": "AIME_I" if "_I_" in task_id else "AIME_II",
                "problem_number": int(task_id.rsplit("P", 1)[1]),
                "replicates": expected_replicates,
                "successes": successes,
                "solve_rate": successes / expected_replicates,
                "valid_answers": parsed,
                "parsed_rate": parsed / expected_replicates,
                "accuracy_on_valid_answer": successes / parsed if parsed else None,
                "mean_output_tokens": fmean(
                    float(row["output_tokens"] or 0) for row in rows
                ),
                "length_finish_rate": sum(
                    row.get("finish_reason") == "length" for row in rows
                )
                / expected_replicates,
            }
        )
    rates = [float(row["solve_rate"]) for row in per_task]
    low, high = task_bootstrap_interval(
        rates,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    overall = {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "model": status["identity"]["model"],
        "sampling": status["identity"]["sampling"],
        "tasks": len(per_task),
        "replicates_per_task": int(status["replicates"]),
        "requests": len(outcomes),
        "round_zero_utility": fmean(rates),
        "task_bootstrap_95_ci": [low, high],
        "between_task_sd": pstdev(rates),
        "parsed_rate": fmean(float(row["parsed_rate"]) for row in per_task),
        "accuracy_on_valid_answer": (
            sum(int(row["successes"]) for row in per_task)
            / sum(int(row["valid_answers"]) for row in per_task)
        ),
        "length_finish_rate": fmean(
            float(row["length_finish_rate"]) for row in per_task
        ),
        "task_band_counts": {
            "floor_0_to_0.1": sum(rate <= 0.1 for rate in rates),
            "informative_0.2_to_0.8": sum(0.2 <= rate <= 0.8 for rate in rates),
            "ceiling_0.9_to_1.0": sum(rate >= 0.9 for rate in rates),
        },
        "contest_utility": {
            contest: fmean(
                float(row["solve_rate"])
                for row in per_task
                if row["contest"] == contest
            )
            for contest in ("AIME_I", "AIME_II")
        },
    }
    return overall, per_task


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rates(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = sorted({str(row["model_label"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, contest in zip(axes, ("AIME_I", "AIME_II"), strict=True):
        for label in labels:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model_label"] == label and row["contest"] == contest
                ),
                key=lambda row: int(row["problem_number"]),
            )
            axis.plot(
                [int(row["problem_number"]) for row in selected],
                [float(row["solve_rate"]) for row in selected],
                marker="o",
                label=label,
            )
        axis.set(title=contest.replace("_", " "), xlabel="Problem number", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Round-0 solve rate")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    summaries: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    for index, (label, raw_path) in enumerate(args.run):
        summary, tasks = summarize_run(
            label=label,
            run_dir=raw_path.resolve(),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        summaries.append(summary)
        all_tasks.extend(tasks)

    pairwise: list[dict[str, Any]] = []
    if len(summaries) > 1:
        rates = {
            label: {
                str(row["task_id"]): float(row["solve_rate"])
                for row in all_tasks
                if row["model_label"] == label
            }
            for label in labels
        }
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                task_ids = sorted(set(rates[left]) & set(rates[right]))
                differences = [rates[right][task] - rates[left][task] for task in task_ids]
                low, high = task_bootstrap_interval(
                    differences,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 100 + len(pairwise),
                )
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "mean_task_solve_rate_difference_right_minus_left": fmean(
                            differences
                        ),
                        "task_bootstrap_95_ci": [low, high],
                        "shared_tasks": len(task_ids),
                    }
                )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_task_solve_rates.csv", all_tasks)
    atomic_json(
        output / "summary.json",
        {
            "analysis_version": "aime-original-round0-analysis-v1",
            "bootstrap_samples": args.bootstrap_samples,
            "models": summaries,
            "pairwise_model_contrasts": pairwise,
        },
    )
    plot_rates(output / "per_problem_solve_rates.png", all_tasks)
    lines = ["# Original AIME Round-0 utility calibration", ""]
    for summary in summaries:
        low, high = summary["task_bootstrap_95_ci"]
        lines.extend(
            [
                f"## {summary['label']}",
                "",
                f"- U0: `{summary['round_zero_utility']:.3f}` "
                f"(task-bootstrap 95% CI `{low:.3f}`–`{high:.3f}`)",
                f"- Parse rate: `{summary['parsed_rate']:.3f}`",
                f"- Accuracy conditional on a valid answer: "
                f"`{summary['accuracy_on_valid_answer']:.3f}`",
                f"- Length-stop rate: `{summary['length_finish_rate']:.3f}`",
                f"- Task bands: `{summary['task_band_counts']}`",
                f"- Contest utility: `{summary['contest_utility']}`",
                "",
            ]
        )
    if pairwise:
        lines.extend(["## Paired task-level contrasts", ""])
        for row in pairwise:
            low, high = row["task_bootstrap_95_ci"]
            lines.append(
                f"- {row['right']} − {row['left']}: "
                f"`{row['mean_task_solve_rate_difference_right_minus_left']:.3f}` "
                f"(95% CI `{low:.3f}`–`{high:.3f}`)"
            )
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
