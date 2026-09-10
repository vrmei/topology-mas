#!/usr/bin/env python3
"""Analyze prefix-horizon utility, attack accuracy, and readout alignment.

This analysis uses already-recorded T=3 traces.  Round 1 and Round 2 are
reported as prefix endpoints rather than silently relabeled as independently
executed shorter-horizon experiments.  Only attack cells with complete real
traces at every round are used for longitudinal comparisons.  A separate
official-T3 reference may include the frozen technical-failure mapping policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LONG_TO_SHORT = {"correct": "C", "target": "T", "other": "O", "unparsed": "U"}
SHORT_STATES = ("C", "T", "O", "U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-updates", type=Path, required=True)
    parser.add_argument("--official-endpoints", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def finite_p_two_sided(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    nonpositive = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    nonnegative = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return min(1.0, 2.0 * min(nonpositive, nonnegative))


def q025(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 0.025)) if len(values) else np.nan


def q975(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 0.975)) if len(values) else np.nan


def build_endpoint_panel(updates: pd.DataFrame) -> pd.DataFrame:
    readout = updates[updates.receiver_scope.eq("readout")].copy()
    required = {1, 2, 3}
    observed = readout.groupby(["system", "run_key"]).round_index.agg(set)
    complete_keys = observed[observed.map(lambda x: required.issubset(x))].index
    key_frame = complete_keys.to_frame(index=False)
    readout = readout.merge(key_frame, on=["system", "run_key"], how="inner")

    identity = ["system", "run_key", "task_id", "graph_id", "m", "attack_node"]
    rows: list[dict] = []
    for item in readout.itertuples(index=False):
        common = {name: getattr(item, name) for name in identity}
        if int(item.round_index) == 1:
            rows.append(
                {
                    **common,
                    "round": 0,
                    "clean_state": LONG_TO_SHORT[item.previous_clean_state],
                    "attack_state": LONG_TO_SHORT[item.previous_attack_state],
                }
            )
        rows.append(
            {
                **common,
                "round": int(item.round_index),
                "clean_state": LONG_TO_SHORT[item.current_clean_state],
                "attack_state": LONG_TO_SHORT[item.current_attack_state],
            }
        )
    panel = pd.DataFrame(rows).drop_duplicates(identity + ["round"])
    expected = len(complete_keys) * 4
    if len(panel) != expected:
        raise RuntimeError(f"expected {expected} endpoint rows, got {len(panel)}")
    panel["clean_correct"] = panel.clean_state.eq("C").astype(int)
    panel["attack_correct"] = panel.attack_state.eq("C").astype(int)
    panel["attack_loss"] = panel.clean_correct - panel.attack_correct
    return panel.sort_values(identity + ["round"]).reset_index(drop=True)


def graph_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    clean = (
        panel.drop_duplicates(["system", "graph_id", "task_id", "round"])
        .groupby(["system", "graph_id", "m", "round"], as_index=False)
        .agg(U=("clean_correct", "mean"), tasks=("task_id", "nunique"))
    )
    attack = (
        panel.groupby(["system", "graph_id", "m", "round"], as_index=False)
        .agg(R=("attack_correct", "mean"), attack_cells=("run_key", "nunique"))
    )
    result = clean.merge(attack, on=["system", "graph_id", "m", "round"], validate="one_to_one")
    result["L"] = result.U - result.R
    round_zero_loss = result[result["round"].eq(0)].set_index(
        ["system", "graph_id"]
    )["L"]
    result["L_adjusted"] = [
        loss - round_zero_loss.loc[(system, graph_id)]
        for system, graph_id, loss in zip(
            result.system, result.graph_id, result.L, strict=True
        )
    ]
    result["retention_ratio"] = np.where(result.U > 0, result.R / result.U, np.nan)
    return result


def density_metrics(graphs: pd.DataFrame) -> pd.DataFrame:
    return (
        graphs.groupby(["system", "m", "round"], as_index=False)
        .agg(
            U=("U", "mean"),
            U_sd=("U", "std"),
            R=("R", "mean"),
            R_sd=("R", "std"),
            L=("L", "mean"),
            L_sd=("L", "std"),
            L_adjusted=("L_adjusted", "mean"),
            L_adjusted_sd=("L_adjusted", "std"),
            graph_count=("graph_id", "nunique"),
        )
    )


def task_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    clean = (
        panel.drop_duplicates(["system", "graph_id", "task_id", "round"])
        .groupby(["system", "task_id", "round"], as_index=False)
        .agg(U=("clean_correct", "mean"), clean_graphs=("graph_id", "nunique"))
    )
    attack = (
        panel.groupby(["system", "task_id", "round"], as_index=False)
        .agg(R=("attack_correct", "mean"), attack_cells=("run_key", "nunique"))
    )
    result = clean.merge(attack, on=["system", "task_id", "round"], validate="one_to_one")
    result["L"] = result.U - result.R
    round_zero_loss = result[result["round"].eq(0)].set_index(
        ["system", "task_id"]
    )["L"]
    result["L_adjusted"] = [
        loss - round_zero_loss.loc[(system, task_id)]
        for system, task_id, loss in zip(
            result.system, result.task_id, result.L, strict=True
        )
    ]
    return result


def rank_stability(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for system, source in frame.groupby("system"):
        for metric in ("U", "R", "L", "L_adjusted"):
            wide = source.pivot(index=unit, columns="round", values=metric)
            for early_round in (0, 1, 2):
                if wide[early_round].nunique() <= 1 or wide[3].nunique() <= 1:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(wide[early_round], wide[3])
                rows.append(
                    {
                        "system": system, "unit": unit, "metric": metric,
                        "early_round": early_round, "reference_round": 3,
                        "spearman_rho": rho, "p_value": p_value,
                        "n_units": len(wide),
                    }
                )
    return pd.DataFrame(rows)


def best_round_counts(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for system, source in frame.groupby("system"):
        for metric in ("U", "R"):
            wide = source.pivot(index=unit, columns="round", values=metric)
            maximum = wide.max(axis=1)
            for round_index in wide.columns:
                other_max = wide.drop(columns=round_index).max(axis=1)
                rows.append(
                    {
                        "system": system, "unit": unit, "metric": metric,
                        "round": int(round_index),
                        "best_including_ties": int(wide[round_index].eq(maximum).sum()),
                        "strictly_best": int(wide[round_index].gt(other_max).sum()),
                        "n_units": len(wide),
                    }
                )
    return pd.DataFrame(rows)


def edge_associations(graphs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, round_index), source in graphs.groupby(["system", "round"]):
        for metric in ("U", "R", "L", "L_adjusted"):
            if source.m.nunique() <= 1 or source[metric].nunique() <= 1:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(source.m, source[metric])
            rows.append(
                {
                    "system": system, "round": int(round_index), "metric": metric,
                    "spearman_rho": rho, "p_value": p_value,
                    "graphs": source.graph_id.nunique(), "edge_levels": source.m.nunique(),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ur(
    panel: pd.DataFrame, reps: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    increments: list[dict] = []
    for system, frame in panel.groupby("system"):
        tasks = np.array(sorted(frame.task_id.unique()))
        per_task = task_metrics(frame)
        rounds = np.array([0, 1, 2, 3])
        arrays = {
            metric: per_task.pivot(index="task_id", columns="round", values=metric)
            .reindex(index=tasks, columns=rounds)
            .to_numpy(float)
            for metric in ("U", "R", "L", "L_adjusted")
        }
        weights = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=reps)
        draws = {metric: (weights @ values) / len(tasks) for metric, values in arrays.items()}
        for metric, values in arrays.items():
            point = np.nanmean(values, axis=0)
            for j, round_index in enumerate(rounds):
                estimates.append(
                    {
                        "system": system,
                        "metric": metric,
                        "round": int(round_index),
                        "estimate": float(point[j]),
                        "ci95_low": q025(draws[metric][:, j]),
                        "ci95_high": q975(draws[metric][:, j]),
                        "tasks": len(tasks),
                        "attack_cells": frame.run_key.nunique(),
                    }
                )
            for j in range(1, len(rounds)):
                diff = draws[metric][:, j] - draws[metric][:, j - 1]
                increments.append(
                    {
                        "system": system,
                        "metric": metric,
                        "from_round": int(rounds[j - 1]),
                        "to_round": int(rounds[j]),
                        "estimate": float(point[j] - point[j - 1]),
                        "ci95_low": q025(diff),
                        "ci95_high": q975(diff),
                        "p_boot_two_sided": finite_p_two_sided(diff),
                        "tasks": len(tasks),
                        "bootstrap_reps": reps,
                    }
                )
    return pd.DataFrame(estimates), pd.DataFrame(increments)


def unique_plurality(row: pd.Series, prefix: str) -> str | None:
    counts = {state: int(row[f"in_{state}"]) for state in SHORT_STATES}
    maximum = max(counts.values())
    if maximum <= 0 or list(counts.values()).count(maximum) != 1:
        return None
    return max(counts, key=counts.get)


def build_alignment_events(updates: pd.DataFrame) -> pd.DataFrame:
    readout = updates[updates.receiver_scope.eq("readout")].copy()
    readout["peer_plurality"] = readout.apply(lambda row: unique_plurality(row, "in_"), axis=1)
    readout["has_unique_plurality"] = readout.peer_plurality.notna()
    readout["previous_equals_plurality"] = readout.prev_state.eq(readout.peer_plurality)
    readout["next_equals_plurality"] = readout.next_state.eq(readout.peer_plurality)
    readout["switched_to_plurality"] = (
        readout.has_unique_plurality
        & ~readout.previous_equals_plurality
        & readout.next_equals_plurality
    )
    readout["beneficial_alignment"] = (
        readout.prev_state.ne("C")
        & readout.peer_plurality.eq("C")
        & readout.next_state.eq("C")
    )
    readout["harmful_alignment"] = (
        readout.prev_state.eq("C")
        & readout.peer_plurality.notna()
        & readout.peer_plurality.ne("C")
        & readout.next_state.ne("C")
    )
    readout["resisted_harmful_plurality"] = (
        readout.prev_state.eq("C")
        & readout.peer_plurality.notna()
        & readout.peer_plurality.ne("C")
        & readout.next_state.eq("C")
    )
    readout["missed_correction"] = (
        readout.prev_state.ne("C")
        & readout.peer_plurality.eq("C")
        & readout.next_state.ne("C")
    )
    readout["target_alignment"] = (
        readout.prev_state.ne("T")
        & readout.peer_plurality.eq("T")
        & readout.next_state.eq("T")
    )
    readout["alignment_opportunity"] = (
        readout.has_unique_plurality & ~readout.previous_equals_plurality
    )
    readout["beneficial_opportunity"] = (
        readout.prev_state.ne("C") & readout.peer_plurality.eq("C")
    )
    readout["harmful_opportunity"] = (
        readout.prev_state.eq("C")
        & readout.peer_plurality.notna()
        & readout.peer_plurality.ne("C")
    )
    readout["target_opportunity"] = (
        readout.prev_state.ne("T") & readout.peer_plurality.eq("T")
    )
    keep = [
        "system", "run_key", "task_id", "graph_id", "m", "attack_node",
        "round_index", "prev_state", "next_state", "peer_plurality", "degree",
        *[f"in_{state}" for state in SHORT_STATES], "has_unique_plurality",
        "previous_equals_plurality", "next_equals_plurality", "switched_to_plurality",
        "beneficial_alignment", "harmful_alignment", "resisted_harmful_plurality",
        "missed_correction", "target_alignment", "alignment_opportunity",
        "beneficial_opportunity", "harmful_opportunity", "target_opportunity",
    ]
    return readout[keep].rename(columns={"round_index": "round"})


def alignment_summary(events: pd.DataFrame) -> pd.DataFrame:
    event_cols = [
        "has_unique_plurality", "next_equals_plurality", "switched_to_plurality",
        "beneficial_alignment", "harmful_alignment", "resisted_harmful_plurality",
        "missed_correction", "target_alignment",
    ]
    rows = []
    for (system, round_index), frame in events.groupby(["system", "round"]):
        def conditional(event: str, eligible: str) -> tuple[float, int]:
            mask = frame[eligible].astype(bool)
            return (
                float(frame.loc[mask, event].mean()) if mask.any() else np.nan,
                int(mask.sum()),
            )

        switch_rate, switch_n = conditional("switched_to_plurality", "alignment_opportunity")
        correction_rate, correction_n = conditional("beneficial_alignment", "beneficial_opportunity")
        corruption_rate, corruption_n = conditional("harmful_alignment", "harmful_opportunity")
        resist_rate, _ = conditional("resisted_harmful_plurality", "harmful_opportunity")
        target_rate, target_n = conditional("target_alignment", "target_opportunity")
        rows.append(
            {
                "system": system,
                "round": int(round_index),
                **{name: float(frame[name].mean()) for name in event_cols},
                "switch_to_plurality_given_opportunity": switch_rate,
                "switch_opportunities": switch_n,
                "correction_given_correct_plurality": correction_rate,
                "correct_plurality_opportunities": correction_n,
                "corruption_given_noncorrect_plurality": corruption_rate,
                "resistance_given_noncorrect_plurality": resist_rate,
                "noncorrect_plurality_opportunities": corruption_n,
                "target_adoption_given_target_plurality": target_rate,
                "target_plurality_opportunities": target_n,
                "observations": len(frame),
            }
        )
    return pd.DataFrame(rows)


def alignment_by_density(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, m, round_index), frame in events.groupby(["system", "m", "round"]):
        opportunity = frame.alignment_opportunity.astype(bool)
        beneficial = frame.beneficial_opportunity.astype(bool)
        harmful = frame.harmful_opportunity.astype(bool)
        target = frame.target_opportunity.astype(bool)
        rows.append(
            {
                "system": system,
                "m": int(m),
                "round": int(round_index),
                "switch_to_plurality_given_opportunity": frame.loc[opportunity, "switched_to_plurality"].mean(),
                "switch_opportunities": int(opportunity.sum()),
                "correction_given_correct_plurality": frame.loc[beneficial, "beneficial_alignment"].mean(),
                "correct_plurality_opportunities": int(beneficial.sum()),
                "corruption_given_noncorrect_plurality": frame.loc[harmful, "harmful_alignment"].mean(),
                "resistance_given_noncorrect_plurality": frame.loc[harmful, "resisted_harmful_plurality"].mean(),
                "noncorrect_plurality_opportunities": int(harmful.sum()),
                "target_adoption_given_target_plurality": frame.loc[target, "target_alignment"].mean(),
                "target_plurality_opportunities": int(target.sum()),
                "observations": len(frame),
            }
        )
    return pd.DataFrame(rows)


def round_gain_loss(panel: pd.DataFrame) -> pd.DataFrame:
    identity = ["system", "run_key", "task_id", "graph_id", "m", "attack_node"]
    previous = panel[identity + ["round", "clean_state", "attack_state"]].copy()
    previous["round"] += 1
    previous = previous.rename(columns={"clean_state": "clean_prev", "attack_state": "attack_prev"})
    current = panel[panel["round"].gt(0)].merge(
        previous, on=identity + ["round"], validate="one_to_one"
    )
    rows = []
    for condition in ("clean", "attack"):
        src, dst = f"{condition}_prev", f"{condition}_state"
        for keys, frame in current.groupby(["system", "round"]):
            corrections = (frame[src].ne("C") & frame[dst].eq("C")).mean()
            corruptions = (frame[src].eq("C") & frame[dst].ne("C")).mean()
            target_adoptions = (frame[src].ne("T") & frame[dst].eq("T")).mean()
            target_recoveries = (frame[src].eq("T") & frame[dst].ne("T")).mean()
            rows.append(
                {
                    "system": keys[0], "round": int(keys[1]), "condition": condition,
                    "correction_mass": corrections, "corruption_mass": corruptions,
                    "net_accuracy_change": corrections - corruptions,
                    "new_target_mass": target_adoptions, "target_recovery_mass": target_recoveries,
                    "net_target_change": target_adoptions - target_recoveries,
                    "observations": len(frame),
                }
            )
    return pd.DataFrame(rows)


def round_gain_loss_by_density(panel: pd.DataFrame) -> pd.DataFrame:
    identity = ["system", "run_key", "task_id", "graph_id", "m", "attack_node"]
    previous = panel[identity + ["round", "clean_state", "attack_state"]].copy()
    previous["round"] += 1
    previous = previous.rename(columns={"clean_state": "clean_prev", "attack_state": "attack_prev"})
    current = panel[panel["round"].gt(0)].merge(previous, on=identity + ["round"], validate="one_to_one")
    rows = []
    for condition in ("clean", "attack"):
        src, dst = f"{condition}_prev", f"{condition}_state"
        for keys, frame in current.groupby(["system", "m", "round"]):
            corrections = (frame[src].ne("C") & frame[dst].eq("C")).mean()
            corruptions = (frame[src].eq("C") & frame[dst].ne("C")).mean()
            rows.append(
                {
                    "system": keys[0], "m": int(keys[1]), "round": int(keys[2]),
                    "condition": condition, "correction_mass": corrections,
                    "corruption_mass": corruptions,
                    "net_accuracy_change": corrections - corruptions,
                    "observations": len(frame),
                }
            )
    return pd.DataFrame(rows)


def transition_summary(panel: pd.DataFrame) -> pd.DataFrame:
    identity = ["system", "run_key", "task_id", "graph_id", "m", "attack_node"]
    previous = panel[identity + ["round", "clean_state", "attack_state"]].copy()
    previous["round"] += 1
    previous = previous.rename(columns={"clean_state": "clean_prev", "attack_state": "attack_prev"})
    current = panel[panel["round"].gt(0)].merge(
        previous, on=identity + ["round"], validate="one_to_one"
    )
    rows = []
    for condition in ("clean", "attack"):
        src, dst = f"{condition}_prev", f"{condition}_state"
        table = (
            current.groupby(["system", "round", src, dst]).size().rename("count").reset_index()
        )
        table["denominator"] = table.groupby(["system", "round", src])["count"].transform("sum")
        table["rate"] = table["count"] / table["denominator"]
        table["condition"] = condition
        table = table.rename(columns={src: "previous_state", dst: "next_state"})
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def official_t3_reference(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    rows = []
    for system, frame in data.groupby("system"):
        clean = frame.drop_duplicates(["graph_id", "task_id"])
        U = clean.clean_state.eq("C").mean()
        R = frame.attack_state.eq("C").mean()
        rows.append(
            {
                "system": system,
                "round": 3,
                "U": U,
                "R": R,
                "L": U - R,
                "attack_cells": len(frame),
                "policy": "official_all_cells",
            }
        )
    return pd.DataFrame(rows)


def plot_roundwise(estimates: pd.DataFrame, out: Path) -> None:
    systems = list(estimates.system.unique())
    colors = {systems[0]: "#2878c8", systems[-1]: "#e36f0a"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for ax, metric, title in zip(
        axes, ("U", "R", "L_adjusted"),
        ("Clean utility", "Attack accuracy", "Adjusted attack loss"), strict=True
    ):
        for system in systems:
            x = estimates[(estimates.system.eq(system)) & (estimates.metric.eq(metric))].sort_values("round")
            ax.plot(x["round"], x.estimate, marker="o", linewidth=2.2, label=system, color=colors[system])
            ax.fill_between(x["round"], x.ci95_low, x.ci95_high, alpha=.15, color=colors[system])
        ax.axhline(0, color="0.45", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("Prefix round T"); ax.grid(alpha=.25)
        ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("Accuracy / difference")
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def plot_density(density: pd.DataFrame, out: Path) -> None:
    systems = list(density.system.unique())
    fig, axes = plt.subplots(len(systems), 3, figsize=(13.5, 4.0 * len(systems)), squeeze=False)
    for row, system in enumerate(systems):
        source = density[density.system.eq(system)]
        for col, metric in enumerate(("U", "R", "L_adjusted")):
            ax = axes[row, col]
            for round_index, frame in source.groupby("round"):
                frame = frame.sort_values("m")
                ax.plot(frame.m, frame[metric], marker="o", label=f"T={round_index}")
            ax.set_title(f"{system}: {metric}"); ax.set_xlabel("Edges m"); ax.grid(alpha=.25)
            if col == 0: ax.set_ylabel("Accuracy / difference")
            if col == 2: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def write_report(
    out: Path,
    estimates: pd.DataFrame,
    increments: pd.DataFrame,
    alignment: pd.DataFrame,
    official: pd.DataFrame,
    audit: dict,
) -> None:
    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Round-wise U–R and alignment report",
        "",
        "Round 1/2 are prefix endpoints extracted from the same T=3 traces. They are not labeled as independently rerun shorter-horizon experiments.",
        "",
        "## Completeness",
        "",
    ]
    for system, info in audit["systems"].items():
        lines.append(f"- {system}: {info['attack_cells']} complete attack cells, {info['tasks']} tasks, {info['graphs']} graphs.")
    lines += ["", "## Endpoint metrics", ""]
    for system in estimates.system.unique():
        lines.append(f"### {system}")
        lines.append("")
        pivot = estimates[estimates.system.eq(system)].pivot(index="round", columns="metric", values="estimate")
        lines.append("| Prefix round | U | R | U-R |")
        lines.append("|---:|---:|---:|---:|")
        for round_index, row in pivot.iterrows():
            lines.append(f"| {round_index} | {pct(row.U)} | {pct(row.R)} | {pct(row.L)} |")
        lines.append("")
        lines.append("Incremental round effects (task bootstrap):")
        lines.append("")
        lines.append("| Metric | Transition | Estimate | 95% CI | p_boot |")
        lines.append("|---|---|---:|---:|---:|")
        for item in increments[increments.system.eq(system)].itertuples(index=False):
            lines.append(
                f"| {item.metric} | {item.from_round}→{item.to_round} | {pct(item.estimate)} | "
                f"[{pct(item.ci95_low)}, {pct(item.ci95_high)}] | {item.p_boot_two_sided:.4g} |"
            )
        lines.append("")
    lines += ["## Readout alignment diagnostics", ""]
    lines.append("Peer plurality is computed from incoming C/T/O/U states only and must not be confused with exact-answer consensus; O collapses multiple wrong answers.")
    lines.append("")
    lines.append("| System | Round | unique plurality | switch→plurality | correction given C plurality | corruption given non-C plurality | resistance |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in alignment.itertuples(index=False):
        lines.append(
            f"| {item.system} | {item.round} | {pct(item.has_unique_plurality)} | "
            f"{pct(item.switch_to_plurality_given_opportunity)} | "
            f"{pct(item.correction_given_correct_plurality)} | "
            f"{pct(item.corruption_given_noncorrect_plurality)} | "
            f"{pct(item.resistance_given_noncorrect_plurality)} |"
        )
    if not official.empty:
        lines += ["", "## Official T=3 all-cell sensitivity reference", ""]
        lines.append("| System | U | R | U-R | N attack cells |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in official.itertuples(index=False):
            lines.append(f"| {item.system} | {pct(item.U)} | {pct(item.R)} | {pct(item.L)} | {item.attack_cells} |")
    lines += [
        "",
        "## Claim boundary",
        "",
        "- U is clean endpoint accuracy; R is endpoint accuracy under attack; U-R is attack loss.",
        "- Longitudinal claims use the same complete-pair population at every round.",
        "- A change in alignment and a simultaneous change in accuracy is descriptive, not by itself causal evidence that alignment produced the accuracy change.",
        "- Cross-system differences are descriptive because model, dataset, temperature, and message representation differ.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    updates = pd.read_parquet(args.local_updates)
    panel = build_endpoint_panel(updates)
    graphs = graph_metrics(panel)
    density = density_metrics(graphs)
    tasks = task_metrics(panel)
    graph_rank = rank_stability(graphs, "graph_id")
    task_rank = rank_stability(tasks, "task_id")
    best_rounds = pd.concat(
        [best_round_counts(graphs, "graph_id"), best_round_counts(tasks, "task_id")],
        ignore_index=True,
    )
    edge_rho = edge_associations(graphs)
    rng = np.random.default_rng(args.seed)
    estimates, increments = bootstrap_ur(panel, args.bootstrap_reps, rng)
    events = build_alignment_events(updates)
    alignment = alignment_summary(events)
    alignment_density = alignment_by_density(events)
    transitions = transition_summary(panel)
    gain_loss = round_gain_loss(panel)
    gain_loss_density = round_gain_loss_by_density(panel)
    official = official_t3_reference(args.official_endpoints)

    panel.to_parquet(args.out / "endpoint_roundwise_cells.parquet", index=False)
    panel.to_csv(args.out / "endpoint_roundwise_cells.csv", index=False)
    graphs.to_csv(args.out / "graph_roundwise_ur.csv", index=False)
    density.to_csv(args.out / "density_roundwise_ur.csv", index=False)
    tasks.to_csv(args.out / "task_roundwise_ur.csv", index=False)
    graph_rank.to_csv(args.out / "graph_rank_stability_vs_t3.csv", index=False)
    task_rank.to_csv(args.out / "task_rank_stability_vs_t3.csv", index=False)
    best_rounds.to_csv(args.out / "best_round_counts.csv", index=False)
    edge_rho.to_csv(args.out / "edge_metric_spearman_by_round.csv", index=False)
    estimates.to_csv(args.out / "system_roundwise_ur_task_bootstrap.csv", index=False)
    increments.to_csv(args.out / "incremental_round_effects_task_bootstrap.csv", index=False)
    events.to_parquet(args.out / "readout_alignment_events.parquet", index=False)
    alignment.to_csv(args.out / "readout_alignment_summary.csv", index=False)
    alignment_density.to_csv(args.out / "readout_alignment_by_density.csv", index=False)
    transitions.to_csv(args.out / "readout_state_transitions.csv", index=False)
    gain_loss.to_csv(args.out / "round_gain_loss_decomposition.csv", index=False)
    gain_loss_density.to_csv(args.out / "round_gain_loss_by_density.csv", index=False)
    official.to_csv(args.out / "official_t3_all_cell_reference.csv", index=False)

    plot_roundwise(estimates, args.out / "roundwise_UR_loss.png")
    plot_density(density, args.out / "density_by_round_UR_loss.png")

    audit = {
        "analysis": "roundwise-prefix-ur-alignment-v1",
        "source": str(args.local_updates),
        "official_endpoint_source": str(args.official_endpoints) if args.official_endpoints else None,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "prefix_rounds": [0, 1, 2, 3],
        "systems": {
            system: {
                "attack_cells": int(frame.run_key.nunique()),
                "tasks": int(frame.task_id.nunique()),
                "graphs": int(frame.graph_id.nunique()),
            }
            for system, frame in panel.groupby("system")
        },
        "interpretation": "Round 1/2 are endpoints of the recorded T=3 prefix, not separate reruns.",
    }
    (args.out / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_report(args.out, estimates, increments, alignment, official, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
