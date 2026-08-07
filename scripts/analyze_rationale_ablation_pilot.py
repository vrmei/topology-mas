"""Analyze paired full-rationale versus answer-only attack outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ANALYSIS_VERSION = "rationale-ablation-analysis-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--answer-only-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_pairs(
    prepared: Path, answer_root: Path, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(manifest["source_run_root"])
    task_ids = set(str(task_id) for task_id in manifest["task_ids"])
    pair_rows: list[pd.DataFrame] = []
    clean_rows: list[pd.DataFrame] = []
    pair_keys = ["task_id", "graph_id", "experiment_seed", "assignment_seed", "attack_node"]
    clean_keys = ["task_id", "graph_id", "experiment_seed", "assignment_seed"]
    for descriptor in manifest["selected_graphs"]:
        key = str(descriptor["stratum"])
        graph_id = str(descriptor["graph_id"])
        full_root = source / "strata" / key / "analysis-v1"
        answer_analysis = answer_root / "strata" / key / "analysis-v1"
        full = pd.DataFrame(read_jsonl(full_root / "paired_attacks.jsonl"))
        answer = pd.DataFrame(read_jsonl(answer_analysis / "paired_attacks.jsonl"))
        full = full.loc[full["task_id"].isin(task_ids) & (full["graph_id"] == graph_id)]
        answer = answer.loc[
            answer["task_id"].isin(task_ids) & (answer["graph_id"] == graph_id)
        ]
        selected_columns = [
            *pair_keys,
            "target_answer",
            "induced_readout_target",
            "paired_accuracy_drop",
            "attack_final_matches_target",
            "max_induced_nonattacker_count",
            "first_induced_readout_target_round",
        ]
        merged = full[selected_columns].merge(
            answer[selected_columns],
            on=pair_keys,
            suffixes=("_full", "_answer_only"),
            validate="one_to_one",
        )
        merged["stratum"] = key
        pair_rows.append(merged)

        full_clean = pd.DataFrame(read_jsonl(full_root / "run_metrics.jsonl"))
        answer_clean = pd.DataFrame(read_jsonl(answer_analysis / "run_metrics.jsonl"))
        full_clean = full_clean.loc[
            (full_clean["condition"] == "clean")
            & full_clean["task_id"].isin(task_ids)
            & (full_clean["graph_id"] == graph_id)
        ]
        answer_clean = answer_clean.loc[
            (answer_clean["condition"] == "clean")
            & answer_clean["task_id"].isin(task_ids)
            & (answer_clean["graph_id"] == graph_id)
        ]
        clean_columns = [*clean_keys, "final_parsed_answer", "final_correct"]
        clean = full_clean[clean_columns].merge(
            answer_clean[clean_columns],
            on=clean_keys,
            suffixes=("_full", "_answer_only"),
            validate="one_to_one",
        )
        clean["stratum"] = key
        clean_rows.append(clean)

    pairs = pd.concat(pair_rows, ignore_index=True)
    clean = pd.concat(clean_rows, ignore_index=True)
    exposure_path = source / "posthoc-conditional-exposure-v1" / "condition_features.csv"
    exposure = pd.read_csv(exposure_path)[
        ["task_id", "graph_id", "attack_node", "degroot_target_exposure"]
    ]
    pairs = pairs.merge(
        exposure,
        on=["task_id", "graph_id", "attack_node"],
        how="left",
        validate="one_to_one",
    )
    return pairs, clean


def crossed_bootstrap(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    graph_ids = sorted(frame["graph_id"].unique())
    task_ids = sorted(frame["task_id"].unique())
    graph_index = {value: index for index, value in enumerate(graph_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    row_graph = frame["graph_id"].map(graph_index).to_numpy(dtype=int)
    row_task = frame["task_id"].map(task_index).to_numpy(dtype=int)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        graph_sample = rng.integers(0, len(graph_ids), size=len(graph_ids))
        task_sample = rng.integers(0, len(task_ids), size=len(task_ids))
        graph_weights = np.bincount(graph_sample, minlength=len(graph_ids))
        task_weights = np.bincount(task_sample, minlength=len(task_ids))
        weights = graph_weights[row_graph] * task_weights[row_task]
        draws[replicate] = np.average(values, weights=weights)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(
    pairs: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    full = pairs["induced_readout_target_full"].astype(int).to_numpy()
    answer = pairs["induced_readout_target_answer_only"].astype(int).to_numpy()
    target_difference = full - answer
    target_low, target_high = crossed_bootstrap(
        pairs, target_difference, replicates=replicates, rng=rng
    )
    full_drop = pairs["paired_accuracy_drop_full"].to_numpy(dtype=float)
    answer_drop = pairs["paired_accuracy_drop_answer_only"].to_numpy(dtype=float)
    drop_difference = full_drop - answer_drop
    drop_low, drop_high = crossed_bootstrap(
        pairs, drop_difference, replicates=replicates, rng=rng
    )
    discordant_full = int(np.sum((full == 1) & (answer == 0)))
    discordant_answer = int(np.sum((full == 0) & (answer == 1)))
    discordant_total = discordant_full + discordant_answer
    diagnostic_p = (
        float(binomtest(discordant_full, discordant_total, 0.5).pvalue)
        if discordant_total
        else 1.0
    )
    summary = pd.DataFrame(
        [
            {
                "outcome": "induced_readout_target",
                "full_rationale_mean": float(full.mean()),
                "answer_only_mean": float(answer.mean()),
                "paired_difference_full_minus_answer": float(target_difference.mean()),
                "ci95_low": target_low,
                "ci95_high": target_high,
            },
            {
                "outcome": "paired_accuracy_drop",
                "full_rationale_mean": float(full_drop.mean()),
                "answer_only_mean": float(answer_drop.mean()),
                "paired_difference_full_minus_answer": float(drop_difference.mean()),
                "ci95_low": drop_low,
                "ci95_high": drop_high,
            },
        ]
    )
    pairs = pairs.copy()
    pairs["exposure_third"] = pd.qcut(
        pairs["degroot_target_exposure"], q=3, labels=("low", "middle", "high")
    )
    strata = (
        pairs.groupby("exposure_third", observed=True, sort=True)
        .agg(
            rows=("task_id", "size"),
            exposure_mean=("degroot_target_exposure", "mean"),
            full_rationale_rate=("induced_readout_target_full", "mean"),
            answer_only_rate=("induced_readout_target_answer_only", "mean"),
        )
        .reset_index()
    )
    strata["paired_difference_full_minus_answer"] = (
        strata["full_rationale_rate"] - strata["answer_only_rate"]
    )
    diagnostic = {
        "full_only_successes": discordant_full,
        "answer_only_successes": discordant_answer,
        "discordant_pairs": discordant_total,
        "exact_binomial_mcnemar_p_unclustered": diagnostic_p,
    }
    return summary, strata, diagnostic


def render_report(
    integrity: dict[str, Any],
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    diagnostic: dict[str, Any],
) -> str:
    lines = [
        "# Matched rationale-ablation pilot",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity",
        "",
    ]
    for key, value in integrity.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Paired outcomes",
            "",
            "| outcome | full | answer only | difference | 95% crossed-bootstrap CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['outcome']} | {row['full_rationale_mean']:.4f} | "
            f"{row['answer_only_mean']:.4f} | "
            f"{row['paired_difference_full_minus_answer']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Exposure thirds",
            "",
            "| exposure | rows | mean exposure | full | answer only | difference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in strata.iterrows():
        lines.append(
            f"| {row['exposure_third']} | {int(row['rows'])} | "
            f"{row['exposure_mean']:.4f} | {row['full_rationale_rate']:.4f} | "
            f"{row['answer_only_rate']:.4f} | "
            f"{row['paired_difference_full_minus_answer']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Discordant-pair diagnostic",
            "",
            f"- Full-only successes: {diagnostic['full_only_successes']}",
            f"- Answer-only successes: {diagnostic['answer_only_successes']}",
            f"- Discordant pairs: {diagnostic['discordant_pairs']}",
            (
                "- Unclustered exact McNemar/binomial p: "
                f"{diagnostic['exact_binomial_mcnemar_p_unclustered']:.6g}"
            ),
            "",
            "## Claim guardrails",
            "",
            "- The primary comparison preserves exact task–graph–node pairing.",
            "- A difference identifies a rationale-presence effect under this protocol.",
            "- It does not separate semantic content from message length or explanation tokens.",
            "- The exact discordance p-value ignores clustering; the crossed interval is primary.",
            "- This 20-task result is a pilot and must be expanded before a paper-level claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    prepared = args.prepared_dir.resolve()
    answer_root = args.answer_only_run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(prepared / "manifest.json")
    run_status = read_json(answer_root / "runner_status.json")
    if run_status.get("status") != "completed":
        raise RuntimeError("answer-only run must be completed before analysis")
    pairs, clean = selected_pairs(prepared, answer_root, manifest)
    expected_pairs = sum(
        manifest["task_count"] * (int(graph["node_count"]) - 1)
        for graph in manifest["selected_graphs"]
    )
    clean_matches = (
        (clean["final_parsed_answer_full"] == clean["final_parsed_answer_answer_only"])
        & (clean["final_correct_full"] == clean["final_correct_answer_only"])
    )
    targets_match = pairs["target_answer_full"] == pairs["target_answer_answer_only"]
    integrity = {
        "passed": bool(
            len(pairs) == expected_pairs
            and targets_match.all()
            and clean_matches.all()
            and pairs["degroot_target_exposure"].notna().all()
        ),
        "paired_conditions": len(pairs),
        "expected_paired_conditions": expected_pairs,
        "graphs": int(pairs["graph_id"].nunique()),
        "tasks": int(pairs["task_id"].nunique()),
        "target_mismatches": int((~targets_match).sum()),
        "clean_drift_conditions": int((~clean_matches).sum()),
        "missing_exposure": int(pairs["degroot_target_exposure"].isna().sum()),
    }
    if not integrity["passed"]:
        raise RuntimeError("rationale-ablation integrity audit failed")
    rng = np.random.default_rng(args.seed)
    summary, strata, diagnostic = summarize(
        pairs, replicates=args.bootstrap_replicates, rng=rng
    )
    pairs.to_csv(output / "paired_conditions.csv", index=False)
    clean.to_csv(output / "clean_drift_audit.csv", index=False)
    summary.to_csv(output / "paired_summary.csv", index=False)
    strata.to_csv(output / "exposure_thirds.csv", index=False)
    (output / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    (output / "discordant_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        render_report(integrity, summary, strata, diagnostic), encoding="utf-8"
    )
    result_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "prepared_dir": str(prepared),
        "answer_only_run_root": str(answer_root),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "integrity_passed": integrity["passed"],
    }
    (output / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result_manifest, indent=2))


if __name__ == "__main__":
    main()
