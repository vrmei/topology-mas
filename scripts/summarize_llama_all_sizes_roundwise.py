#!/usr/bin/env python3
"""Create a compact configuration-level report from all-size roundwise outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving the input index."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return result
    ordered = numeric[valid].sort_values()
    count = len(ordered)
    adjusted = (ordered * count / np.arange(1, count + 1)).clip(upper=1.0)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]
    result.loc[adjusted.index] = adjusted
    return result


def endpoint_wide(estimates: pd.DataFrame) -> pd.DataFrame:
    index = ["system", "n", "m", "graphs", "tasks", "attack_cells"]
    wide = estimates.pivot(index=index, columns=["metric", "round"], values="estimate")
    wide.columns = [f"{metric}_T{int(round_index)}" for metric, round_index in wide.columns]
    return wide.reset_index()


def increment_wide(increments: pd.DataFrame) -> pd.DataFrame:
    data = increments.copy()
    data["transition"] = data.from_round.astype(str) + "to" + data.to_round.astype(str)
    index = ["system", "n", "m", "graphs", "tasks", "attack_cells"]
    values = ["estimate", "ci95_low", "ci95_high", "p_boot_two_sided", "q_bh_within_metric_transition"]
    wide = data.pivot(index=index, columns=["metric", "transition"], values=values)
    wide.columns = [f"{value}_{metric}_{transition}" for value, metric, transition in wide.columns]
    return wide.reset_index()


def pattern_counts(increments: pd.DataFrame) -> pd.DataFrame:
    data = increments[increments.metric.isin(["U", "R", "L_adjusted"])].copy()
    data["positive"] = data.estimate.gt(0)
    data["negative"] = data.estimate.lt(0)
    data["zero"] = data.estimate.eq(0)
    data["fdr_positive"] = data.positive & data.q_bh_within_metric_transition.lt(0.05)
    data["fdr_negative"] = data.negative & data.q_bh_within_metric_transition.lt(0.05)
    return (
        data.groupby(["system", "n", "metric", "from_round", "to_round"], as_index=False)
        .agg(
            configurations=("m", "size"), positive=("positive", "sum"),
            negative=("negative", "sum"), zero=("zero", "sum"),
            fdr_positive=("fdr_positive", "sum"), fdr_negative=("fdr_negative", "sum"),
            mean_effect=("estimate", "mean"), median_effect=("estimate", "median"),
        )
    )


def best_rounds(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in wide.itertuples(index=False):
        for metric in ("U", "R"):
            values = {t: getattr(item, f"{metric}_T{t}") for t in (1, 2, 3)}
            maximum = max(values.values())
            winners = [str(t) for t, value in values.items() if np.isclose(value, maximum)]
            rows.append({
                "system": item.system, "n": item.n, "m": item.m,
                "metric": metric, "best_prefix_rounds_including_ties": ",".join(winners),
                "T1": values[1], "T2": values[2], "T3": values[3],
            })
    return pd.DataFrame(rows)


def best_round_counts(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, n, metric), group in best.groupby(["system", "n", "metric"]):
        for round_index in (1, 2, 3):
            contains = group.best_prefix_rounds_including_ties.str.split(",").apply(
                lambda values: str(round_index) in values
            )
            unique = group.best_prefix_rounds_including_ties.eq(str(round_index))
            rows.append({
                "system": system,
                "n": n,
                "metric": metric,
                "round": round_index,
                "best_including_ties": int(contains.sum()),
                "uniquely_best": int(unique.sum()),
                "configurations": int(len(group)),
            })
    return pd.DataFrame(rows)


def adjust_cross_size_contrasts(source: Path) -> pd.DataFrame | None:
    path = source / "cross_size_paired_task_contrasts_vs_n5.csv"
    if not path.exists():
        return None
    contrasts = pd.read_csv(path)
    contrasts["q_bh_within_round_metric"] = (
        contrasts.groupby(["round", "metric"], group_keys=False)["p"].apply(bh_adjust)
    )
    contrasts.to_csv(source / "cross_size_paired_task_contrasts_vs_n5_fdr.csv", index=False)
    return contrasts


def plot_increment_distributions(increments: pd.DataFrame, out: Path) -> None:
    data = increments[increments.metric.isin(["U", "R", "L_adjusted"])].copy()
    data["transition"] = data.from_round.astype(str) + "→" + data.to_round.astype(str)
    data["n_label"] = "n=" + data.n.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, metric, title in zip(
        axes, ("U", "R", "L_adjusted"),
        ("Clean utility increment", "Attack-accuracy increment", "Adjusted-loss increment"),
        strict=True,
    ):
        source = data[data.metric.eq(metric)]
        sns.boxplot(data=source, x="transition", y="estimate", hue="n_label", ax=ax, fliersize=0)
        sns.stripplot(
            data=source, x="transition", y="estimate", hue="n_label", dodge=True,
            alpha=0.38, size=3, legend=False, ax=ax,
        )
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("Round transition"); ax.set_ylabel("Change")
        ax.yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    axes[-1].legend(title="System size", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def plot_alignment(alignment: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("switch_to_plurality_given_opportunity", "Switch to peer plurality"),
        ("correction_given_correct_plurality", "Correction | correct plurality"),
        ("corruption_given_noncorrect_plurality", "Corruption | non-C plurality"),
        ("target_adoption_given_target_plurality", "Target adoption | target plurality"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for system, group in alignment.groupby("system"):
            ax.plot(group["round"], group[metric], marker="o", linewidth=2, label=system)
        ax.set_title(title); ax.set_xlabel("Update round"); ax.set_ylabel("Conditional rate")
        ax.set_xticks([1, 2, 3]); ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def write_report(
    output: Path,
    endpoints: pd.DataFrame,
    increments: pd.DataFrame,
    patterns: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    lines = [
        "# Llama/GSM8K all-size roundwise configuration report", "",
        "T=1 and T=2 are prefix endpoints of the same recorded T=3 traces, not independent reruns.", "",
    ]
    for n in sorted(endpoints.n.unique()):
        source = endpoints[endpoints.n.eq(n)].sort_values("m")
        lines += [f"## n={n}", "", "| m | graphs | U1 | R1 | U1-R1 | U2 | R2 | U2-R2 | U3 | R3 | U3-R3 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for item in source.itertuples(index=False):
            lines.append(
                f"| {item.m} | {item.graphs} | {pct(item.U_T1)} | {pct(item.R_T1)} | {pct(item.L_T1)} | "
                f"{pct(item.U_T2)} | {pct(item.R_T2)} | {pct(item.L_T2)} | "
                f"{pct(item.U_T3)} | {pct(item.R_T3)} | {pct(item.L_T3)} |"
            )
        lines += ["", "Round increments; `*` means BH-FDR q<0.05 within the same metric/transition family.", "", "| m | ΔU 0→1 | ΔU 1→2 | ΔU 2→3 | ΔR 0→1 | ΔR 1→2 | ΔR 2→3 | ΔL* 0→1 | ΔL* 1→2 | ΔL* 2→3 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        inc = increments[increments.n.eq(n)]
        for m in sorted(inc.m.unique()):
            row = [str(m)]
            for metric in ("U", "R", "L_adjusted"):
                for left, right in ((0, 1), (1, 2), (2, 3)):
                    item = inc[(inc.m.eq(m)) & (inc.metric.eq(metric)) & (inc.from_round.eq(left)) & (inc.to_round.eq(right))].iloc[0]
                    star = "*" if item.q_bh_within_metric_transition < 0.05 else ""
                    row.append(f"{100*item.estimate:+.2f}{star}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += [
        "## Interpretation boundary", "",
        "- U is clean readout accuracy; R is attacked readout accuracy; L=U-R.",
        "- L_adjusted subtracts each graph's non-zero Round-0 clean/attack mismatch before measuring added attack loss.",
        "- Per-configuration p-values use a paired task bootstrap over the same 50 tasks; stars use BH-FDR within each metric × round-transition family.",
        "- Alignment rates are conditional descriptions. Later-round opportunities are selected residual cases, so changes across rounds are not identified as a causal change in conformity.",
    ]
    (output / "CONFIGURATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_findings(
    output: Path,
    endpoints: pd.DataFrame,
    increments: pd.DataFrame,
    patterns: pd.DataFrame,
    best_counts: pd.DataFrame,
    alignment: pd.DataFrame,
    contrasts: pd.DataFrame | None,
) -> None:
    lines = [
        "# Llama/GSM8K all-size roundwise findings", "",
        "## Scope", "",
        "- 259 directed graphs across 55 (n,m) configurations: n=5 (61), n=6 (51), n=7 (76), n=8 (71).",
        "- All systems use the same 50 GSM8K tasks.",
        "- T=1 and T=2 are prefix endpoints reconstructed from the same T=3 traces; they are not independently rerun horizons.",
        "- U is clean readout accuracy, R is attacked readout accuracy, and adjusted loss removes each graph's Round-0 clean/attack mismatch.", "",
        "## Configuration-level direction counts", "",
        "The table counts signs across edge-count configurations. FDR counts use BH q<0.05 within each metric and round-transition family.", "",
        "| n | metric | transition | + | - | 0 | FDR+ | FDR- | configs |",
        "|---:|:---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in patterns.itertuples(index=False):
        lines.append(
            f"| {row.n} | {row.metric} | {row.from_round}→{row.to_round} | "
            f"{row.positive} | {row.negative} | {row.zero} | "
            f"{row.fdr_positive} | {row.fdr_negative} | {row.configurations} |"
        )
    lines += ["", "## Best observed prefix round by configuration", "",
              "Ties are retained; therefore tie-inclusive counts can sum to more than the number of configurations.", "",
              "| n | metric | T | best incl. ties | uniquely best | configs |",
              "|---:|:---|---:|---:|---:|---:|"]
    for row in best_counts.itertuples(index=False):
        lines.append(
            f"| {row.n} | {row.metric} | {row.round} | {row.best_including_ties} | "
            f"{row.uniquely_best} | {row.configurations} |"
        )
    if contrasts is not None:
        lines += ["", "## Cross-size paired task contrasts versus n=5", "",
                  "BH correction is applied to the three size contrasts within each round × metric family.", "",
                  "| T | metric | contrast | estimate (pp) | 95% CI (pp) | p | q |",
                  "|---:|:---|:---|---:|:---:|---:|---:|"]
        for row in contrasts.itertuples(index=False):
            lines.append(
                f"| {row.round} | {row.metric} | {row.contrast} | {100*row.estimate:+.2f} | "
                f"[{100*row.lo:+.2f}, {100*row.hi:+.2f}] | {row.p:.4f} | "
                f"{row.q_bh_within_round_metric:.4f} |"
            )
    lines += [
        "", "## Claim boundary", "",
        "- The most repeatable empirical pattern is a first-round attacked-accuracy decrease followed by a second-round recovery.",
        "- The third update has smaller and heterogeneous marginal effects; these data do not justify a universal claim that T=3 is better than T=2.",
        "- Higher n raises both U and R at later prefixes. Evidence that it lowers adjusted attack damage is weaker and does not survive every multiplicity correction.",
        "- Later-round alignment rates condition on selected remaining opportunities. They describe the traces but do not identify a causal increase or decrease in conformity.",
    ]
    (output / "ALL_SIZE_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.analysis_dir
    estimates = pd.read_csv(source / "configuration_roundwise_task_bootstrap.csv")
    increments = pd.read_csv(source / "configuration_incremental_task_bootstrap.csv")
    alignment = pd.read_csv(source / "readout_alignment_summary.csv")
    endpoints = endpoint_wide(estimates)
    increment_table = increment_wide(increments)
    patterns = pattern_counts(increments)
    best = best_rounds(endpoints)
    best_counts = best_round_counts(best)
    contrasts = adjust_cross_size_contrasts(source)
    endpoints.to_csv(source / "configuration_endpoint_wide.csv", index=False)
    increment_table.to_csv(source / "configuration_increment_wide.csv", index=False)
    patterns.to_csv(source / "configuration_round_pattern_counts.csv", index=False)
    best.to_csv(source / "configuration_best_prefix_rounds.csv", index=False)
    best_counts.to_csv(source / "configuration_best_prefix_round_counts.csv", index=False)
    plot_increment_distributions(increments, source / "configuration_increment_distributions.png")
    plot_alignment(alignment, source / "readout_alignment_by_size_and_round.png")
    write_report(source, endpoints, increments, patterns, alignment)
    write_findings(source, endpoints, increments, patterns, best_counts, alignment, contrasts)


if __name__ == "__main__":
    main()
