"""Inferential summaries for a dense-m utility/robustness pilot.

Intervals resample tasks and remain conditional on the sampled topologies. Empirical
graph spread is reported separately so the two uncertainty sources are not conflated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ("u0", "ut", "delta_u", "r_mean", "r_worst", "d_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--audit-final-turns",
        action="store_true",
        help="Scan raw traces to audit output-cap and final parsing failures.",
    )
    return parser.parse_args()


def load_runs(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stratum in sorted((root / "strata").glob("n*_m*")):
        path = stratum / "analysis-v1" / "run_metrics.jsonl"
        if not path.exists():
            continue
        frame = pd.read_json(path, lines=True)
        n_text, m_text = stratum.name.split("_")
        frame["stratum"] = stratum.name
        frame["n"] = int(n_text[1:])
        frame["m"] = int(m_text[1:])
        frames.append(frame)
    if not frames:
        raise ValueError("no analyzed strata found")
    return pd.concat(frames, ignore_index=True)


def topology_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (n, m, graph_id), frame in runs.groupby(["n", "m", "graph_id"], sort=True):
        clean = frame[frame.condition.eq("clean")]
        attack = frame[frame.condition.eq("attack")]
        per_position = attack.groupby("attack_node").final_correct.mean()
        ut = float(clean.final_correct.mean())
        u0 = float(clean.readout_round_zero_correct.mean())
        r_mean = float(attack.final_correct.mean())
        rows.append(
            {
                "n": int(n),
                "m": int(m),
                "normalized_density": float(m / (n - 1) ** 2),
                "graph_id": graph_id,
                "tasks": int(clean.task_id.nunique()),
                "u0": u0,
                "ut": ut,
                "delta_u": ut - u0,
                "r_mean": r_mean,
                "r_worst": float(per_position.min()),
                "d_mean": ut - r_mean,
            }
        )
    return pd.DataFrame(rows)


def task_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (n, m, graph_id, task_id), frame in runs.groupby(
        ["n", "m", "graph_id", "task_id"], sort=True
    ):
        clean = frame[frame.condition.eq("clean")]
        attack = frame[frame.condition.eq("attack")]
        if len(clean) != 1:
            raise ValueError(f"expected one clean run for {graph_id}/{task_id}")
        ut = float(clean.final_correct.iloc[0])
        u0 = float(clean.readout_round_zero_correct.iloc[0])
        r_mean = float(attack.final_correct.mean())
        rows.append(
            {
                "n": int(n),
                "m": int(m),
                "graph_id": graph_id,
                "task_id": task_id,
                "u0": u0,
                "ut": ut,
                "delta_u": ut - u0,
                "r_mean": r_mean,
                "r_worst": float(attack.groupby("attack_node").final_correct.mean().min()),
                "d_mean": ut - r_mean,
            }
        )
    return pd.DataFrame(rows)


def task_level_by_m(task_graph: pd.DataFrame) -> pd.DataFrame:
    return (
        task_graph.groupby(["n", "m", "task_id"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["n", "m", "task_id"])
    )


def bootstrap_summary(
    task_m: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (n, m), frame in task_m.groupby(["n", "m"], sort=True):
        values = frame[list(METRICS)].to_numpy(float)
        draws = rng.integers(0, len(values), size=(replicates, len(values)))
        boot = values[draws].mean(axis=1)
        row: dict[str, object] = {
            "n": int(n),
            "m": int(m),
            "normalized_density": float(m / (n - 1) ** 2),
            "tasks": len(values),
        }
        for index, metric in enumerate(METRICS):
            row[f"{metric}_estimate"] = float(values[:, index].mean())
            row[f"{metric}_ci95_low"] = float(np.quantile(boot[:, index], 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(boot[:, index], 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def adjacent_contrasts(
    task_m: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for n, n_frame in task_m.groupby("n", sort=True):
        levels = sorted(n_frame.m.unique())
        for low_m, high_m in zip(levels, levels[1:], strict=True):
            low = n_frame[n_frame.m.eq(low_m)].set_index("task_id")
            high = n_frame[n_frame.m.eq(high_m)].set_index("task_id")
            task_ids = sorted(set(low.index) & set(high.index))
            low_values = low.loc[task_ids, list(METRICS)].to_numpy(float)
            high_values = high.loc[task_ids, list(METRICS)].to_numpy(float)
            deltas = high_values - low_values
            draws = rng.integers(0, len(task_ids), size=(replicates, len(task_ids)))
            boot = deltas[draws].mean(axis=1)
            for index, metric in enumerate(METRICS):
                lo = float(np.quantile(boot[:, index], 0.025))
                hi = float(np.quantile(boot[:, index], 0.975))
                rows.append(
                    {
                        "n": int(n),
                        "low_m": int(low_m),
                        "high_m": int(high_m),
                        "metric": metric,
                        "difference": float(deltas[:, index].mean()),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "tasks": len(task_ids),
                    }
                )
    return pd.DataFrame(rows)


def graph_spread(graphs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (n, m), frame in graphs.groupby(["n", "m"], sort=True):
        row: dict[str, object] = {"n": int(n), "m": int(m), "graphs": len(frame)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = (
                float(frame[metric].std(ddof=1)) if len(frame) > 1 else np.nan
            )
            row[f"{metric}_range"] = float(frame[metric].max() - frame[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(graphs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [("m", metric) for metric in METRICS] + [
        ("ut", "r_mean"),
        ("ut", "d_mean"),
    ]
    for n, frame in graphs.groupby("n", sort=True):
        for x, y in pairs:
            result = spearmanr(frame[x], frame[y])
            rows.append(
                {
                    "n": int(n),
                    "x": x,
                    "y": y,
                    "rho": float(result.statistic),
                    "p_value_descriptive": float(result.pvalue),
                    "topologies": len(frame),
                    "warning": "topologies are sampled and m has ties; p-value is descriptive",
                }
            )
    return pd.DataFrame(rows)


def pareto_front(graphs: pd.DataFrame) -> pd.DataFrame:
    selected: list[bool] = []
    for row in graphs.itertuples():
        peers = graphs[graphs.n.eq(row.n)]
        dominated = (
            peers.ut.ge(row.ut)
            & peers.r_mean.ge(row.r_mean)
            & (peers.ut.gt(row.ut) | peers.r_mean.gt(row.r_mean))
        ).any()
        selected.append(not bool(dominated))
    result = graphs.copy()
    result["pareto_ut_rmean"] = selected
    return result


def final_state_rates(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (n, m), frame in runs.groupby(["n", "m"], sort=True):
        for condition, selected in frame.groupby("condition", sort=True):
            parsed = selected[~selected.final_answer_state.eq("unparsed")]
            rows.append(
                {
                    "n": int(n),
                    "m": int(m),
                    "condition": condition,
                    "runs": len(selected),
                    "correct_rate": float(selected.final_correct.mean()),
                    "target_error_rate": float(
                        selected.final_answer_state.eq("target_error").mean()
                    ),
                    "other_error_rate": float(
                        selected.final_answer_state.eq("other_error").mean()
                    ),
                    "unparsed_rate": float(
                        selected.final_answer_state.eq("unparsed").mean()
                    ),
                    "parsed_only_accuracy": float(parsed.final_correct.mean()),
                }
            )
    return pd.DataFrame(rows)


def final_turn_cap_audit(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing = 0
    for stratum in sorted((root / "strata").glob("n*_m*")):
        n_text, m_text = stratum.name.split("_")
        n, m = int(n_text[1:]), int(m_text[1:])
        for path in (stratum / "batch" / "traces").glob("*.json"):
            stored = json.loads(path.read_text(encoding="utf-8"))["trace"]
            final = stored.get("final_raw_output")
            candidates = [
                turn for turn in stored.get("turns", []) if turn.get("raw_output") == final
            ]
            if not candidates:
                missing += 1
                continue
            turn = max(
                candidates,
                key=lambda value: (
                    value.get("round_index", -1),
                    value.get("node_id", -1),
                ),
            )
            output_tokens = turn.get("output_tokens")
            rows.append(
                {
                    "n": n,
                    "m": m,
                    "condition": stored["condition"],
                    "final_state": stored["final_answer_state"],
                    "output_tokens": output_tokens,
                    "at_cap": bool(
                        output_tokens is not None
                        and output_tokens >= stored["execution_settings"]["max_output_tokens"]
                    ),
                    "finish_reason": turn.get("finish_reason"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["unparsed"] = frame.final_state.eq("unparsed")
    summary = (
        frame.groupby(["n", "m", "condition"], as_index=False)
        .agg(
            runs=("unparsed", "size"),
            unparsed_rate=("unparsed", "mean"),
            at_cap_rate=("at_cap", "mean"),
            unparsed_count=("unparsed", "sum"),
        )
        .sort_values(["n", "m", "condition"])
    )
    audit = {
        "traces": len(frame),
        "missing_final_turn": missing,
        "overall_at_cap_rate": float(frame.at_cap.mean()),
        "overall_unparsed_rate": float(frame.unparsed.mean()),
        "p_at_cap_given_unparsed": float(frame.loc[frame.unparsed, "at_cap"].mean()),
        "p_unparsed_given_at_cap": float(frame.loc[frame.at_cap, "unparsed"].mean()),
        "finish_reasons": {
            str(key): int(value)
            for key, value in frame.finish_reason.value_counts(dropna=False).items()
        },
    }
    return summary, audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_root)
    graphs = topology_metrics(runs)
    tasks = task_metrics(runs)
    task_m = task_level_by_m(tasks)
    rng = np.random.default_rng(args.seed)

    pareto_front(graphs).to_csv(args.output_dir / "topology_metrics.csv", index=False)
    task_m.to_csv(args.output_dir / "task_metrics_by_m.csv", index=False)
    bootstrap_summary(task_m, rng, args.bootstrap_replicates).to_csv(
        args.output_dir / "m_response_task_bootstrap.csv", index=False
    )
    adjacent_contrasts(task_m, rng, args.bootstrap_replicates).to_csv(
        args.output_dir / "adjacent_m_contrasts.csv", index=False
    )
    graph_spread(graphs).to_csv(args.output_dir / "graph_spread.csv", index=False)
    correlations(graphs).to_csv(args.output_dir / "topology_correlations.csv", index=False)
    final_state_rates(runs).to_csv(
        args.output_dir / "final_state_rates.csv", index=False
    )
    final_turn_audit: dict[str, object] | None = None
    if args.audit_final_turns:
        cap_summary, final_turn_audit = final_turn_cap_audit(args.run_root)
        cap_summary.to_csv(args.output_dir / "final_turn_cap_audit.csv", index=False)
    manifest = {
        "analysis_version": "dense-pilot-inference-v1",
        "run_root": str(args.run_root.resolve()),
        "tasks": int(runs.task_id.nunique()),
        "topologies": len(graphs),
        "strata": int(graphs[["n", "m"]].drop_duplicates().shape[0]),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "uncertainty": {
            "task_bootstrap": "paired task bootstrap after averaging sampled graphs within m",
            "graph_spread": "empirical SD and range over sampled graphs, reported separately",
            "scope": "conditional on model, prompt, task subset, sampled graphs, and one run seed",
        },
        "claim_limits": [
            "50 tasks are suitable for pilot discrimination, not stable small-effect claims",
            "five graphs per non-complete stratum do not identify the full legal graph population",
            "adjacent m contrasts are paired by task but not by graph",
            "Spearman p-values are descriptive because m has ties and graphs are sampled",
        ],
        "final_turn_audit": final_turn_audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
