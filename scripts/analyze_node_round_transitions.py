"""Summarize density-conditioned node-round transitions from paired MAS traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_node_round_adoption import extract_updates, read_json, read_jsonl

ANALYSIS_VERSION = "node-round-transitions-v1"
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20_260_812
TASK_CHUNK_SIZE = 100


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    eligible: Callable[[pd.DataFrame], pd.Series]
    success: Callable[[pd.DataFrame], pd.Series]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def metrics() -> tuple[Metric, ...]:
    def previous_correct(frame: pd.DataFrame) -> pd.Series:
        return frame.previous_attack_state.eq("correct")

    def previous_target(frame: pd.DataFrame) -> pd.Series:
        return frame.previous_attack_state.eq("target")

    def previous_induced(frame: pd.DataFrame) -> pd.Series:
        return frame.previous_induced_target_state.eq(1)

    def received_target(frame: pd.DataFrame) -> pd.Series:
        return frame.received_target.eq(1)

    def received_induced(frame: pd.DataFrame) -> pd.Series:
        return frame.received_induced_target.eq(1)
    return (
        Metric(
            "received_target_per_update",
            "P(any target message received) among eligible benign updates",
            lambda x: pd.Series(True, index=x.index),
            received_target,
        ),
        Metric(
            "received_induced_target_per_update",
            "P(attack-induced target message received) among eligible benign updates",
            lambda x: pd.Series(True, index=x.index),
            received_induced,
        ),
        Metric(
            "c_to_t_given_target_exposed",
            "P(C->T | previous C and any target received)",
            lambda x: previous_correct(x) & received_target(x),
            lambda x: x.current_attack_state.eq("target"),
        ),
        Metric(
            "induced_c_to_t_given_induced_target_exposed",
            (
                "P(C->T and paired clean current != T | previous C and an "
                "attack-induced target received)"
            ),
            lambda x: previous_correct(x) & received_induced(x),
            lambda x: x.current_attack_state.eq("target")
            & ~x.current_clean_state.eq("target"),
        ),
        Metric(
            "c_to_o_given_target_exposed",
            "P(C->O | previous C and any target received); unparsed is separate",
            lambda x: previous_correct(x) & received_target(x),
            lambda x: x.current_attack_state.eq("other"),
        ),
        Metric(
            "c_to_u_given_target_exposed",
            "P(C->unparsed | previous C and any target received)",
            lambda x: previous_correct(x) & received_target(x),
            lambda x: x.current_attack_state.eq("unparsed"),
        ),
        Metric(
            "t_to_c",
            "P(T->C | previous T)",
            previous_target,
            lambda x: x.current_attack_state.eq("correct"),
        ),
        Metric(
            "t_to_t",
            "P(T->T | previous T)",
            previous_target,
            lambda x: x.current_attack_state.eq("target"),
        ),
        Metric(
            "t_to_o",
            "P(T->O | previous T); unparsed is separate",
            previous_target,
            lambda x: x.current_attack_state.eq("other"),
        ),
        Metric(
            "induced_t_to_c",
            "P(T->C | previous attack-induced T)",
            previous_induced,
            lambda x: x.current_attack_state.eq("correct"),
        ),
        Metric(
            "t_to_c_given_target_exposed",
            "P(T->C | previous T and current target exposure)",
            lambda x: previous_target(x) & received_target(x),
            lambda x: x.current_attack_state.eq("correct"),
        ),
        Metric(
            "t_to_c_given_no_target_exposure",
            "P(T->C | previous T and no current target exposure)",
            lambda x: previous_target(x) & ~received_target(x),
            lambda x: x.current_attack_state.eq("correct"),
        ),
    )


def add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = frame.copy()
    fixed["regime"] = "fixed_t3"
    depth = frame.loc[
        frame.round_index + frame.receiver_distance_to_readout <= frame.graph_depth
    ].copy()
    depth["regime"] = "graph_depth"
    return pd.concat([fixed, depth], ignore_index=True)


def sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    keys = ["regime", "stratum", "task_id"]
    for metric in metrics():
        eligible = metric.eligible(frame).astype(bool)
        success = metric.success(frame).astype(bool) & eligible
        selected = frame.loc[eligible, keys].copy()
        selected["numerator"] = success.loc[eligible].astype(int).to_numpy()
        selected["denominator"] = 1
        grouped = selected.groupby(keys, as_index=False)[["numerator", "denominator"]].sum()
        all_tasks = frame[keys].drop_duplicates()
        grouped = all_tasks.merge(grouped, on=keys, how="left", validate="one_to_one")
        grouped[["numerator", "denominator"]] = grouped[
            ["numerator", "denominator"]
        ].fillna(0)
        grouped["metric"] = metric.name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def ratio_interval(
    frame: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    numerator = frame.numerator.to_numpy(dtype=float)
    denominator = frame.denominator.to_numpy(dtype=float)
    total_denominator = denominator.sum()
    if total_denominator == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(numerator.sum() / total_denominator)
    draws = rng.integers(0, len(frame), size=(replicates, len(frame)))
    sampled_num = numerator[draws].sum(axis=1)
    sampled_den = denominator[draws].sum(axis=1)
    valid = sampled_den > 0
    values = sampled_num[valid] / sampled_den[valid]
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_rates(
    stats: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    descriptions = {metric.name: metric.description for metric in metrics()}
    rows = []
    for (regime, stratum, metric), group in stats.groupby(
        ["regime", "stratum", "metric"], sort=True
    ):
        point, low, high = ratio_interval(group, rng, replicates)
        n_text, m_text = stratum.split("_")
        rows.append(
            {
                "regime": regime,
                "stratum": stratum,
                "n": int(n_text[1:]),
                "m": int(m_text[1:]),
                "metric": metric,
                "description": descriptions[metric],
                "numerator": int(group.numerator.sum()),
                "denominator": int(group.denominator.sum()),
                "estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "tasks": int(group.task_id.nunique()),
                "ci_scope": "task bootstrap conditional on selected graphs",
            }
        )
    return pd.DataFrame(rows)


def density_contrasts(
    stats: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    rows = []
    for (regime, metric), selected in stats.groupby(["regime", "metric"], sort=True):
        for n in (5, 8):
            strata = sorted(
                (
                    value
                    for value in selected.stratum.unique()
                    if int(value.split("_")[0][1:]) == n
                ),
                key=lambda value: int(value.split("_")[1][1:]),
            )
            for low, high in zip(strata, strata[1:], strict=False):
                low_frame = selected[selected.stratum.eq(low)].set_index("task_id")
                high_frame = selected[selected.stratum.eq(high)].set_index("task_id")
                task_ids = sorted(set(low_frame.index) & set(high_frame.index))
                if not task_ids:
                    continue
                low_frame = low_frame.loc[task_ids]
                high_frame = high_frame.loc[task_ids]
                draws = rng.integers(0, len(task_ids), size=(replicates, len(task_ids)))
                low_den = low_frame.denominator.to_numpy()[draws].sum(axis=1)
                high_den = high_frame.denominator.to_numpy()[draws].sum(axis=1)
                valid = (low_den > 0) & (high_den > 0)
                low_rates = (
                    low_frame.numerator.to_numpy()[draws].sum(axis=1)[valid]
                    / low_den[valid]
                )
                high_rates = (
                    high_frame.numerator.to_numpy()[draws].sum(axis=1)[valid]
                    / high_den[valid]
                )
                values = high_rates - low_rates
                if len(values) == 0:
                    continue
                if low_frame.denominator.sum() == 0 or high_frame.denominator.sum() == 0:
                    continue
                low_point = low_frame.numerator.sum() / low_frame.denominator.sum()
                high_point = high_frame.numerator.sum() / high_frame.denominator.sum()
                rows.append(
                    {
                        "regime": regime,
                        "n": n,
                        "metric": metric,
                        "low_stratum": low,
                        "high_stratum": high,
                        "difference": float(high_point - low_point),
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "ci_excludes_zero": bool(
                            np.quantile(values, 0.025) > 0
                            or np.quantile(values, 0.975) < 0
                        ),
                        "tasks": len(task_ids),
                    }
                )
    return pd.DataFrame(rows)


def transition_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "regime",
                "stratum",
                "previous_attack_state",
                "current_attack_state",
            ],
            as_index=False,
        )
        .agg(updates=("task_id", "size"), tasks=("task_id", "nunique"))
        .assign(
            row_total=lambda x: x.groupby(
                ["regime", "stratum", "previous_attack_state"]
            ).updates.transform("sum"),
            transition_probability=lambda x: x.updates / x.row_total,
        )
    )


def exposure_reach(frame: pd.DataFrame) -> pd.DataFrame:
    condition_keys = ["regime", "stratum", "task_id", "graph_id", "attack_node"]
    receiver_keys = [*condition_keys, "receiver_node"]
    receivers = (
        frame.groupby(receiver_keys, as_index=False)
        .agg(
            ever_target_exposed=("received_target", "max"),
            ever_induced_target_exposed=("received_induced_target", "max"),
        )
    )
    conditions = (
        receivers.groupby(condition_keys, as_index=False)
        .agg(
            unique_eligible_receivers=("receiver_node", "size"),
            unique_target_exposed_receivers=("ever_target_exposed", "sum"),
            unique_induced_target_exposed_receivers=(
                "ever_induced_target_exposed",
                "sum",
            ),
        )
    )
    return (
        conditions.groupby(["regime", "stratum"], as_index=False)
        .agg(
            attack_conditions=("task_id", "size"),
            mean_unique_eligible_receivers=("unique_eligible_receivers", "mean"),
            mean_unique_target_exposed_receivers=(
                "unique_target_exposed_receivers",
                "mean",
            ),
            mean_unique_induced_target_exposed_receivers=(
                "unique_induced_target_exposed_receivers",
                "mean",
            ),
        )
    )


def by_round(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in metrics():
        eligible = metric.eligible(frame).astype(bool)
        success = metric.success(frame).astype(bool) & eligible
        selected = frame.loc[eligible, ["regime", "stratum", "round_index"]].copy()
        selected["success"] = success.loc[eligible].astype(int).to_numpy()
        rows.append(
            selected.groupby(["regime", "stratum", "round_index"], as_index=False)
            .agg(numerator=("success", "sum"), denominator=("success", "size"))
            .assign(metric=metric.name)
        )
    result = pd.concat(rows, ignore_index=True)
    result["estimate"] = result.numerator / result.denominator
    return result


def combine_transition_matrices(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine graph/task chunks before normalizing transition rows."""
    result = (
        pd.concat(parts, ignore_index=True)
        .groupby(
            [
                "regime",
                "stratum",
                "previous_attack_state",
                "current_attack_state",
            ],
            as_index=False,
        )
        .agg(updates=("updates", "sum"), task_graphs=("tasks", "sum"))
    )
    result["row_total"] = result.groupby(
        ["regime", "stratum", "previous_attack_state"]
    ).updates.transform("sum")
    result["transition_probability"] = result.updates / result.row_total
    return result


def combine_exposure_reach(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine chunk means using attack-condition counts as weights."""
    frame = pd.concat(parts, ignore_index=True)
    value_columns = [
        "mean_unique_eligible_receivers",
        "mean_unique_target_exposed_receivers",
        "mean_unique_induced_target_exposed_receivers",
    ]
    for column in value_columns:
        frame[f"weighted_{column}"] = frame[column] * frame.attack_conditions
    aggregations: dict[str, tuple[str, str]] = {
        "attack_conditions": ("attack_conditions", "sum")
    }
    aggregations.update(
        {
            f"weighted_{column}": (f"weighted_{column}", "sum")
            for column in value_columns
        }
    )
    result = frame.groupby(["regime", "stratum"], as_index=False).agg(
        **aggregations
    )
    for column in value_columns:
        result[column] = result[f"weighted_{column}"] / result.attack_conditions
        result = result.drop(columns=f"weighted_{column}")
    result["induced_target_reach_fraction"] = (
        result.mean_unique_induced_target_exposed_receivers
        / result.mean_unique_eligible_receivers
    )
    return result


def combine_by_round(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine round-level count tables before computing rates."""
    result = (
        pd.concat(parts, ignore_index=True)
        .groupby(
            ["regime", "stratum", "round_index", "metric"], as_index=False
        )[["numerator", "denominator"]]
        .sum()
    )
    result["estimate"] = result.numerator / result.denominator
    return result


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    status = read_json(args.run_root / "orchestrator_status.json")
    stat_parts = []
    matrix_parts = []
    reach_parts = []
    round_parts = []
    audits = []
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        graph_records = read_jsonl(
            args.run_root / "strata" / stratum / "selected_graphs.jsonl"
        )
        for graph_record in graph_records:
            graph_id = str(graph_record["graph_id"])
            task_records = read_jsonl(
                args.run_root / "strata" / stratum / "batch/inputs/tasks.jsonl"
            )
            task_values = [str(item["task_id"]) for item in task_records]
            graph_audits = []
            for start in range(0, len(task_values), TASK_CHUNK_SIZE):
                chunk_ids = set(task_values[start : start + TASK_CHUNK_SIZE])
                raw, chunk_audit = extract_updates(
                    args.run_root,
                    {"strata": [descriptor]},
                    graph_ids={graph_id},
                    task_ids=chunk_ids,
                )
                if not chunk_audit["passed"]:
                    raise RuntimeError(
                        "integrity audit failed: "
                        + "; ".join(chunk_audit["errors"][:10])
                    )
                frame = add_regimes(raw)
                stat_parts.append(sufficient_statistics(frame))
                matrix_parts.append(transition_matrix(frame))
                reach_parts.append(exposure_reach(frame))
                round_parts.append(by_round(frame))
                graph_audits.append(
                    {
                        **chunk_audit,
                        "fixed_t3_updates": int(frame.regime.eq("fixed_t3").sum()),
                        "graph_depth_updates": int(
                            frame.regime.eq("graph_depth").sum()
                        ),
                        "unparsed_current_fixed_t3": int(
                            (
                                frame.regime.eq("fixed_t3")
                                & frame.current_attack_state.eq("unparsed")
                            ).sum()
                        ),
                    }
                )
                del raw, frame
            audits.append(
                {
                    "stratum": stratum,
                    "graph_id": graph_id,
                    "passed": all(item["passed"] for item in graph_audits),
                    "paired_conditions": sum(
                        item["paired_conditions"] for item in graph_audits
                    ),
                    "eligible_updates": sum(
                        item["eligible_updates"] for item in graph_audits
                    ),
                    "new_induced_adoptions": sum(
                        item["new_induced_adoptions"] for item in graph_audits
                    ),
                    "updates_receiving_target": sum(
                        item["updates_receiving_target"] for item in graph_audits
                    ),
                    "updates_receiving_induced_target": sum(
                        item["updates_receiving_induced_target"]
                        for item in graph_audits
                    ),
                    "fixed_t3_updates": sum(
                        item["fixed_t3_updates"] for item in graph_audits
                    ),
                    "graph_depth_updates": sum(
                        item["graph_depth_updates"] for item in graph_audits
                    ),
                    "unparsed_current_fixed_t3": sum(
                        item["unparsed_current_fixed_t3"] for item in graph_audits
                    ),
                    "task_chunks": len(graph_audits),
                }
            )
    stats = (
        pd.concat(stat_parts, ignore_index=True)
        .groupby(
            ["regime", "stratum", "task_id", "metric"], as_index=False
        )[["numerator", "denominator"]]
        .sum()
    )
    rng = np.random.default_rng(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output / "task_sufficient_statistics.csv", index=False)
    (output / "extraction_complete.json").write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "task_chunk_size": TASK_CHUNK_SIZE,
                "audited_graph_chunks": len(audits),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rates = summarize_rates(stats, rng, args.bootstrap_replicates)
    contrasts = density_contrasts(stats, rng, args.bootstrap_replicates)
    rates.to_csv(output / "transition_rates.csv", index=False)
    contrasts.to_csv(output / "adjacent_density_contrasts.csv", index=False)
    combine_transition_matrices(matrix_parts).to_csv(
        output / "transition_matrix.csv", index=False
    )
    combine_exposure_reach(reach_parts).to_csv(
        output / "exposure_reach.csv", index=False
    )
    combine_by_round(round_parts).to_csv(
        output / "transition_rates_by_round.csv", index=False
    )
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "passed": all(item["passed"] for item in audits),
        "strata": audits,
        "paired_conditions": sum(item["paired_conditions"] for item in audits),
        "eligible_updates": sum(item["eligible_updates"] for item in audits),
        "new_induced_adoptions": sum(
            item["new_induced_adoptions"] for item in audits
        ),
        "updates_receiving_target": sum(
            item["updates_receiving_target"] for item in audits
        ),
        "updates_receiving_induced_target": sum(
            item["updates_receiving_induced_target"] for item in audits
        ),
        "fixed_t3_updates": sum(item["fixed_t3_updates"] for item in audits),
        "graph_depth_updates": sum(
            item["graph_depth_updates"] for item in audits
        ),
        "unparsed_current_fixed_t3": sum(
            item["unparsed_current_fixed_t3"] for item in audits
        ),
    }
    (output / "integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(args.run_root.resolve()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "regimes": ["fixed_t3", "graph_depth"],
        "primary_adoption_metric": "induced_c_to_t_given_induced_target_exposed",
        "primary_recovery_metric": "induced_t_to_c",
        "claim_limits": [
            "task bootstrap is conditional on selected graphs",
            "transition states are post-treatment intermediate outcomes",
            "unparsed outputs are not folded into other",
            "graph-depth horizon changes normal correction and attack exposure",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "integrity": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
