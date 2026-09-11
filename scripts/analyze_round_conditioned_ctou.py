#!/usr/bin/env python3
"""Separate round-composition shifts from local CTOU response-law shifts.

The input is a normalized local-update parquet with one row per normal receiver
update.  T=1 and T=2 are treated as prefixes of the recorded T=3 trajectory;
the script never labels them as independently rerun horizons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRANSITIONS = {
    "C_to_T": (("C",), "T"),
    "C_to_O": (("C",), "O"),
    "C_to_notC": (("C",), "notC"),
    "O_to_C": (("O",), "C"),
    "U_to_C": (("U",), "C"),
    "OU_to_C": (("O", "U"), "C"),
    "T_to_C": (("T",), "C"),
    "T_to_T": (("T",), "T"),
}
ROUND_PAIRS = ((1, 2), (2, 3))
COUNT_COLUMNS = ("in_C", "in_T", "in_O", "in_U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--minimum-cell-round-rows", type=int, default=10)
    return parser.parse_args()


def bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna()
    if not valid.any():
        return result
    ordered = numeric[valid].sort_values()
    count = len(ordered)
    adjusted = (ordered * count / np.arange(1, count + 1)).clip(upper=1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]
    result.loc[adjusted.index] = adjusted
    return result


def event_indicator(frame: pd.DataFrame, next_state: str) -> np.ndarray:
    if next_state == "notC":
        return frame.next_state.ne("C").to_numpy(np.int8)
    return frame.next_state.eq(next_state).to_numpy(np.int8)


def selected_cells(
    frame: pd.DataFrame,
    cell_columns: list[str],
    left_round: int,
    right_round: int,
    minimum_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        frame.groupby([*cell_columns, "round_index"], dropna=False)
        .agg(rows=("event", "size"), events=("event", "sum"))
        .reset_index()
    )
    n_wide = counts.pivot(index=cell_columns, columns="round_index", values="rows").fillna(0)
    e_wide = counts.pivot(index=cell_columns, columns="round_index", values="events").fillna(0)
    if left_round not in n_wide or right_round not in n_wide:
        return pd.DataFrame(), pd.DataFrame()
    keep = n_wide[left_round].ge(minimum_rows) & n_wide[right_round].ge(minimum_rows)
    n_wide = n_wide.loc[keep, [left_round, right_round]]
    e_wide = e_wide.loc[keep, [left_round, right_round]]
    if n_wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    result = n_wide.reset_index()
    result["left_rows"] = n_wide[left_round].to_numpy(float)
    result["right_rows"] = n_wide[right_round].to_numpy(float)
    result["left_events"] = e_wide[left_round].to_numpy(float)
    result["right_events"] = e_wide[right_round].to_numpy(float)
    result["left_rate"] = result.left_events / result.left_rows
    result["right_rate"] = result.right_events / result.right_rows
    result["cell_weight"] = np.minimum(result.left_rows, result.right_rows)
    result["risk_difference"] = result.right_rate - result.left_rate
    lookup = result[cell_columns].copy()
    lookup["cell_id"] = np.arange(len(result), dtype=int)
    return result, lookup


def cluster_bootstrap(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    cell_columns: list[str],
    left_round: int,
    right_round: int,
    cluster_column: str,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    working = frame.merge(lookup, on=cell_columns, how="inner", validate="many_to_one")
    working = working[working.round_index.isin([left_round, right_round])].copy()
    clusters = sorted(working[cluster_column].astype(str).unique())
    cluster_index = {value: index for index, value in enumerate(clusters)}
    working["cluster_index"] = working[cluster_column].astype(str).map(cluster_index)
    working["round_slot"] = working.round_index.map({left_round: 0, right_round: 1})
    grouped = (
        working.groupby(["cluster_index", "cell_id", "round_slot"], as_index=False)
        .agg(rows=("event", "size"), events=("event", "sum"))
    )
    cluster_count = len(clusters)
    cell_count = len(lookup)
    rows = np.zeros((cluster_count, cell_count, 2), dtype=np.float64)
    events = np.zeros_like(rows)
    index = (
        grouped.cluster_index.to_numpy(int),
        grouped.cell_id.to_numpy(int),
        grouped.round_slot.to_numpy(int),
    )
    rows[index] = grouped.rows.to_numpy(float)
    events[index] = grouped.events.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    remaining = replicates
    while remaining:
        batch = min(100, remaining)
        weights = rng.multinomial(
            cluster_count, np.full(cluster_count, 1 / cluster_count), size=batch
        ).astype(float)
        draw_rows = np.einsum("bc,cjr->bjr", weights, rows, optimize=True)
        draw_events = np.einsum("bc,cjr->bjr", weights, events, optimize=True)
        rates = np.divide(
            draw_events,
            draw_rows,
            out=np.full_like(draw_events, np.nan),
            where=draw_rows > 0,
        )
        cell_weights = np.minimum(draw_rows[:, :, 0], draw_rows[:, :, 1])
        valid = np.isfinite(rates).all(axis=2) & (cell_weights > 0)
        numerators = np.where(
            valid,
            cell_weights * (rates[:, :, 1] - rates[:, :, 0]),
            0,
        ).sum(axis=1)
        denominators = np.where(valid, cell_weights, 0).sum(axis=1)
        estimates.append(
            np.divide(
                numerators,
                denominators,
                out=np.full(batch, np.nan),
                where=denominators > 0,
            )
        )
        remaining -= batch
    return np.concatenate(estimates)


def bootstrap_summary(draws: np.ndarray) -> tuple[float, float, float, int]:
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return np.nan, np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [0.025, 0.975])
    lower_count = int((finite <= 0).sum())
    upper_count = int((finite >= 0).sum())
    p = min(1.0, 2 * (min(lower_count, upper_count) + 1) / (len(finite) + 1))
    return float(lo), float(hi), float(p), int(len(finite))


def matched_round_effects(
    source: pd.DataFrame,
    *,
    matching: str,
    minimum_rows: int,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    cells_out: list[pd.DataFrame] = []
    for scope in ("readout", "internal"):
        scoped = source[source.receiver_scope.eq(scope)]
        for n in sorted(scoped.n.unique()):
            by_size = scoped[scoped.n.eq(n)]
            for transition_index, (transition, (previous, following)) in enumerate(TRANSITIONS.items()):
                selected = by_size[by_size.prev_state.isin(previous)].copy()
                selected["event"] = event_indicator(selected, following)
                cell_columns = list(COUNT_COLUMNS)
                if matching == "task_ctou":
                    cell_columns = ["task_id", *cell_columns]
                for pair_index, (left_round, right_round) in enumerate(ROUND_PAIRS):
                    cells, lookup = selected_cells(
                        selected,
                        cell_columns,
                        left_round,
                        right_round,
                        minimum_rows,
                    )
                    if cells.empty:
                        continue
                    estimate = float(np.average(cells.risk_difference, weights=cells.cell_weight))
                    task_draws = cluster_bootstrap(
                        selected,
                        lookup,
                        cell_columns,
                        left_round,
                        right_round,
                        "task_id",
                        replicates=replicates,
                        seed=seed + 100_000 * int(n) + 1_000 * transition_index + pair_index,
                    )
                    graph_draws = cluster_bootstrap(
                        selected,
                        lookup,
                        cell_columns,
                        left_round,
                        right_round,
                        "graph_id",
                        replicates=replicates,
                        seed=seed + 1_000_000 + 100_000 * int(n) + 1_000 * transition_index + pair_index,
                    )
                    task_lo, task_hi, task_p, task_success = bootstrap_summary(task_draws)
                    graph_lo, graph_hi, graph_p, graph_success = bootstrap_summary(graph_draws)
                    matched_keys = selected.merge(lookup, on=cell_columns, how="inner")
                    coverage = {}
                    for round_index in (left_round, right_round):
                        denominator = int(selected.round_index.eq(round_index).sum())
                        numerator = int(matched_keys.round_index.eq(round_index).sum())
                        coverage[round_index] = numerator / denominator if denominator else np.nan
                    rows.append({
                        "matching": matching,
                        "receiver_scope": scope,
                        "n": int(n),
                        "transition": transition,
                        "from_round": left_round,
                        "to_round": right_round,
                        "matched_cells": len(cells),
                        "matched_weight": int(cells.cell_weight.sum()),
                        "from_rate": float(np.average(cells.left_rate, weights=cells.cell_weight)),
                        "to_rate": float(np.average(cells.right_rate, weights=cells.cell_weight)),
                        "risk_difference": estimate,
                        "from_coverage": coverage[left_round],
                        "to_coverage": coverage[right_round],
                        "task_ci95_low": task_lo,
                        "task_ci95_high": task_hi,
                        "task_p_two_sided": task_p,
                        "task_bootstrap_success": task_success,
                        "graph_ci95_low": graph_lo,
                        "graph_ci95_high": graph_hi,
                        "graph_p_two_sided": graph_p,
                        "graph_bootstrap_success": graph_success,
                    })
                    cells = cells.assign(
                        matching=matching,
                        receiver_scope=scope,
                        n=int(n),
                        transition=transition,
                        from_round=left_round,
                        to_round=right_round,
                    )
                    cells_out.append(cells)
    result = pd.DataFrame(rows)
    if not result.empty:
        family = ["matching", "receiver_scope", "transition", "from_round", "to_round"]
        result["task_q_bh"] = result.groupby(family, group_keys=False)["task_p_two_sided"].apply(bh_adjust)
        result["graph_q_bh"] = result.groupby(family, group_keys=False)["graph_p_two_sided"].apply(bh_adjust)
    return result, pd.concat(cells_out, ignore_index=True) if cells_out else pd.DataFrame()


def symmetric_decomposition(source: pd.DataFrame, minimum_rows: int) -> pd.DataFrame:
    rows: list[dict] = []
    for scope in ("readout", "internal"):
        scoped = source[source.receiver_scope.eq(scope)]
        for n in sorted(scoped.n.unique()):
            by_size = scoped[scoped.n.eq(n)]
            for transition, (previous, following) in TRANSITIONS.items():
                selected = by_size[by_size.prev_state.isin(previous)].copy()
                selected["event"] = event_indicator(selected, following)
                for left_round, right_round in ROUND_PAIRS:
                    cells, _ = selected_cells(
                        selected,
                        list(COUNT_COLUMNS),
                        left_round,
                        right_round,
                        minimum_rows,
                    )
                    if cells.empty:
                        continue
                    w_left = cells.left_rows / cells.left_rows.sum()
                    w_right = cells.right_rows / cells.right_rows.sum()
                    p_left, p_right = cells.left_rate, cells.right_rate
                    composition = float((0.5 * (w_right - w_left) * (p_left + p_right)).sum())
                    law = float((0.5 * (p_right - p_left) * (w_left + w_right)).sum())
                    left_rate = float((w_left * p_left).sum())
                    right_rate = float((w_right * p_right).sum())
                    rows.append({
                        "receiver_scope": scope,
                        "n": int(n),
                        "transition": transition,
                        "from_round": left_round,
                        "to_round": right_round,
                        "matched_cells": len(cells),
                        "from_rate_common_support": left_rate,
                        "to_rate_common_support": right_rate,
                        "total_change": right_rate - left_rate,
                        "composition_component": composition,
                        "local_law_component": law,
                        "identity_error": (right_rate - left_rate) - composition - law,
                    })
    return pd.DataFrame(rows)


def plot_matched_effects(frame: pd.DataFrame, path: Path) -> None:
    data = frame[(frame.matching == "ctou") & (frame.receiver_scope == "readout")].copy()
    transitions = list(TRANSITIONS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, (left_round, right_round) in zip(axes, ROUND_PAIRS, strict=True):
        current = data[(data.from_round == left_round) & (data.to_round == right_round)]
        for offset, n in zip(np.linspace(-0.24, 0.24, 4), sorted(current.n.unique()), strict=True):
            group = current[current.n.eq(n)].set_index("transition").reindex(transitions)
            x = np.arange(len(transitions)) + offset
            y = group.risk_difference.to_numpy(float)
            low = group.task_ci95_low.to_numpy(float)
            high = group.task_ci95_high.to_numpy(float)
            ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", capsize=3, label=f"n={n}")
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
        ax.set_title(f"Readout matched local-law difference: Round {right_round} minus {left_round}")
        ax.set_ylabel("Risk difference")
        ax.yaxis.set_major_formatter(lambda value, _: f"{100*value:.0f}%")
        ax.grid(alpha=0.2)
    axes[-1].set_xticks(np.arange(len(transitions)), transitions)
    axes[-1].set_xlabel("Transition")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_decomposition(frame: pd.DataFrame, path: Path) -> None:
    data = frame[
        (frame.receiver_scope == "readout")
        & (frame.from_round == 1)
        & (frame.to_round == 2)
        & frame.transition.isin(["C_to_T", "C_to_notC", "OU_to_C", "T_to_C"])
    ].copy()
    transitions = ["C_to_T", "C_to_notC", "OU_to_C", "T_to_C"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, transition in zip(axes.flat, transitions, strict=True):
        current = data[data.transition.eq(transition)].sort_values("n")
        x = np.arange(len(current))
        ax.bar(x, current.composition_component, label="composition shift")
        ax.bar(
            x,
            current.local_law_component,
            bottom=current.composition_component,
            label="local-law shift",
        )
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
        ax.set_xticks(x, [f"n={n}" for n in current.n])
        ax.set_title(transition)
        ax.yaxis.set_major_formatter(lambda value, _: f"{100*value:.0f}%")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Round 2 - Round 1")
    axes[1, 0].set_ylabel("Round 2 - Round 1")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Readout transition change decomposition on common CTOU support")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(out: Path, effects: pd.DataFrame, decomposition: pd.DataFrame, args: argparse.Namespace) -> None:
    primary = effects[
        (effects.matching == "ctou")
        & (effects.receiver_scope == "readout")
        & (effects.from_round == 1)
        & (effects.to_round == 2)
    ].sort_values(["transition", "n"])
    lines = [
        "# Round-conditioned CTOU analysis", "",
        "T=1 and T=2 are prefix endpoints from the recorded T=3 trajectories, not independent reruns.", "",
        "## Primary readout contrasts", "",
        "Exact previous state and incoming CTOU counts are held fixed. Positive values mean the transition is more likely in Round 2 than Round 1.", "",
        "| transition | n | R1 | R2 | difference (pp) | task 95% CI (pp) | task q | graph q | matched weight |",
        "|:---|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.transition} | {row.n} | {100*row.from_rate:.2f} | {100*row.to_rate:.2f} | "
            f"{100*row.risk_difference:+.2f} | [{100*row.task_ci95_low:+.2f}, {100*row.task_ci95_high:+.2f}] | "
            f"{row.task_q_bh:.4f} | {row.graph_q_bh:.4f} | {row.matched_weight} |"
        )
    lines += ["", "## Interpretation boundary", "",
              "- A matched round contrast is observational. It can identify residual association after CTOU controls, not a causal effect of a round label.",
              "- The symmetric decomposition is restricted to CTOU cells supported in both rounds; unmatched-support coverage is reported separately in the CSV.",
              "- GPU replay is gated by the preregistered practical and uncertainty criteria in docs/round_phase_mechanism_plan.md."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "round-conditioned-ctou-v1",
        "updates": str(args.updates),
        "rows": int(len(pd.read_parquet(args.updates, columns=["task_id"]))),
        "bootstrap_reps": args.bootstrap_reps,
        "minimum_cell_round_rows": args.minimum_cell_round_rows,
        "seed": args.seed,
        "prefix_endpoint_warning": "T1/T2 are prefixes of T3 traces, not independent reruns",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.updates)
    required = {
        "task_id", "graph_id", "n", "round_index", "receiver_scope",
        "prev_state", "next_state", *COUNT_COLUMNS,
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    effect_frames = []
    cell_frames = []
    for matching in ("ctou", "task_ctou"):
        effects, cells = matched_round_effects(
            source,
            matching=matching,
            minimum_rows=args.minimum_cell_round_rows,
            replicates=args.bootstrap_reps,
            seed=args.seed + (0 if matching == "ctou" else 10_000_000),
        )
        effect_frames.append(effects)
        cell_frames.append(cells)
    effects = pd.concat(effect_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True)
    decomposition = symmetric_decomposition(source, args.minimum_cell_round_rows)
    effects.to_csv(args.out / "matched_round_effects.csv", index=False)
    cells.to_csv(args.out / "matched_cell_rates.csv", index=False)
    decomposition.to_csv(args.out / "round_change_decomposition.csv", index=False)
    plot_matched_effects(effects, args.out / "readout_matched_round_effects.png")
    plot_decomposition(decomposition, args.out / "readout_round1_to_round2_decomposition.png")
    write_report(args.out, effects, decomposition, args)
    print(json.dumps({"effects": len(effects), "cells": len(cells), "decompositions": len(decomposition)}))


if __name__ == "__main__":
    main()
