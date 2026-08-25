#!/usr/bin/env python3
"""Analyze the paired with-self versus no-self evidence-volume intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topology_mas.experiments.evidence_volume import atomic_json, read_jsonl

CONTRASTS = ((2, 1), (3, 2), (3, 1))
EQUIVALENCE_MARGIN = 0.05
PAIR_KEYS = ["task_id", "scenario", "ratio_id", "replicate", "multiplier"]
FROZEN_COLUMNS = [
    *PAIR_KEYS,
    "generation_seed",
    "previous_stimulus_id",
    "peer_set_fingerprint",
    "peer_stimulus_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-self-run-dir", type=Path, required=True)
    parser.add_argument("--no-self-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser.parse_args()


def load_condition(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path / "outcomes.jsonl"))
    if len(frame) != 7500:
        raise ValueError(f"{condition}: expected 7500 rows, found {len(frame)}")
    if frame.request_id.duplicated().any():
        raise ValueError(f"{condition}: duplicate request IDs")
    return frame.assign(self_condition=condition)


def audit_pairing(with_self: pd.DataFrame, no_self: pd.DataFrame) -> dict[str, Any]:
    left = with_self.sort_values(PAIR_KEYS).reset_index(drop=True)
    right = no_self.sort_values(PAIR_KEYS).reset_index(drop=True)
    mismatches: dict[str, int] = {}
    for column in FROZEN_COLUMNS:
        left_values = (
            left[column].map(json.dumps) if column == "peer_stimulus_ids" else left[column]
        )
        right_values = (
            right[column].map(json.dumps) if column == "peer_stimulus_ids" else right[column]
        )
        count = int((left_values != right_values).sum())
        if count:
            mismatches[column] = count
    if mismatches:
        raise ValueError(f"with/no-self pairing mismatch: {mismatches}")
    if "previous_mode" in no_self and set(no_self.previous_mode) != {"omit"}:
        raise ValueError("no-self outcomes do not declare previous_mode=omit")
    return {
        "rows_per_condition": len(left),
        "pairing_mismatches": mismatches,
        "tasks": int(left.task_id.nunique()),
        "ratios": int(left.ratio_id.nunique()),
        "replicates": int(left.replicate.nunique()),
    }


def cluster_interval(
    per_task: pd.Series,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = per_task.to_numpy(dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def task_scale_effects(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["task_id", "scenario", "ratio_id", "replicate"],
        columns="multiplier",
        values=outcome,
        aggfunc="first",
    ).reset_index()
    if pivot[[1, 2, 3]].isna().any().any():
        raise ValueError(f"incomplete scale pairs for {outcome}")
    pivot[[1, 2, 3]] = pivot[[1, 2, 3]].astype(float)
    rows: list[dict[str, Any]] = []
    for high, low in CONTRASTS:
        work = pivot.assign(effect=pivot[high] - pivot[low])
        grouped = (
            work.groupby(["task_id", "scenario", "ratio_id"], observed=True)
            .effect.mean()
            .reset_index()
        )
        grouped["contrast"] = f"{high}x-{low}x"
        rows.extend(grouped.to_dict("records"))
        pooled = (
            work.groupby(["task_id", "scenario"], observed=True).effect.mean().reset_index()
        )
        pooled["ratio_id"] = "pooled"
        pooled["contrast"] = f"{high}x-{low}x"
        rows.extend(pooled.to_dict("records"))
    return pd.DataFrame(rows)


def summarize_effects(
    effects: pd.DataFrame,
    *,
    effect_name: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["scenario", "ratio_id", "contrast"]
    for group_key, group in effects.groupby(groups, observed=True, sort=True):
        scenario, ratio_id, contrast = group_key
        per_task = group.set_index("task_id").effect
        estimate, low, high = cluster_interval(
            per_task,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + sum(ord(char) for char in ":".join(group_key)),
        )
        rows.append(
            {
                "scenario": scenario,
                "ratio_id": ratio_id,
                "contrast": contrast,
                "effect_name": effect_name,
                "tasks": len(per_task),
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "practically_equivalent": low > -EQUIVALENCE_MARGIN
                and high < EQUIVALENCE_MARGIN,
                "directional": low > 0 or high < 0,
            }
        )
    return pd.DataFrame(rows)


def interaction_effects(
    with_effects: pd.DataFrame,
    no_effects: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["task_id", "scenario", "ratio_id", "contrast"]
    merged = with_effects.merge(
        no_effects,
        on=keys,
        suffixes=("_with", "_no"),
        validate="one_to_one",
    )
    return merged[keys].assign(effect=merged.effect_with - merged.effect_no)


def decision_table(no_self: pd.DataFrame, interaction: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in sorted(no_self.scenario.unique()):
        no_row = no_self[
            no_self.scenario.eq(scenario)
            & no_self.ratio_id.eq("pooled")
            & no_self.contrast.eq("3x-1x")
        ].iloc[0]
        int_row = interaction[
            interaction.scenario.eq(scenario)
            & interaction.ratio_id.eq("pooled")
            & interaction.contrast.eq("3x-1x")
        ].iloc[0]
        if bool(no_row.practically_equivalent) and float(int_row.ci_low) > 0:
            classification = "self_dilution_dominated"
        elif float(no_row.ci_low) > EQUIVALENCE_MARGIN:
            classification = "strong_peer_volume_persistence"
        elif bool(no_row.directional) and not bool(no_row.practically_equivalent):
            classification = "peer_volume_persistence"
        else:
            classification = "mixed_or_inconclusive"
        rows.append(
            {
                "scenario": scenario,
                "classification": classification,
                "no_self_3x_minus_1x": float(no_row.estimate),
                "no_self_ci_low": float(no_row.ci_low),
                "no_self_ci_high": float(no_row.ci_high),
                "with_minus_no_interaction": float(int_row.estimate),
                "interaction_ci_low": float(int_row.ci_low),
                "interaction_ci_high": float(int_row.ci_high),
            }
        )
    return rows


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    cells = (
        frame.groupby(
            ["self_condition", "scenario", "ratio_id", "correct_share", "incoming_degree"],
            as_index=False,
            observed=True,
        )
        .is_primary_outcome.mean()
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    for row_index, scenario in enumerate(("attack_adoption", "benign_correction")):
        for column_index, self_condition in enumerate(("with_self", "no_self")):
            axis = axes[row_index, column_index]
            work = cells[
                cells.scenario.eq(scenario) & cells.self_condition.eq(self_condition)
            ]
            for _ratio_id, group in work.groupby("ratio_id", sort=True):
                group = group.sort_values("incoming_degree")
                label = f"{100 * float(group.correct_share.iloc[0]):.0f}% C"
                axis.plot(group.incoming_degree, group.is_primary_outcome, "o-", label=label)
            axis.set_title(f"{scenario.replace('_', ' ')} | {self_condition.replace('_', ' ')}")
            axis.set_xlabel("Distinct peer rationales")
            axis.set_ylabel("Primary outcome probability")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with_self = load_condition(args.with_self_run_dir.resolve(), "with_self")
    no_self = load_condition(args.no_self_run_dir.resolve(), "no_self")
    audit = audit_pairing(with_self, no_self)
    prompt_audit_path = args.no_self_run_dir.resolve() / "prompt_audit.json"
    prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))
    incidental_ids = set(prompt_audit["incidental_previous_text_match_request_ids"])

    all_rows = pd.concat([with_self, no_self], ignore_index=True)
    all_rows["is_primary_outcome"] = all_rows.is_primary_outcome.astype(float)
    all_rows["is_unparsed"] = all_rows.is_unparsed.astype(float)

    summaries: list[pd.DataFrame] = []
    task_outputs: list[pd.DataFrame] = []
    for outcome in ("is_primary_outcome", "is_unparsed"):
        with_effects = task_scale_effects(
            with_self.assign(**{outcome: with_self[outcome].astype(float)}), outcome
        )
        no_effects = task_scale_effects(
            no_self.assign(**{outcome: no_self[outcome].astype(float)}), outcome
        )
        interactions = interaction_effects(with_effects, no_effects)
        for name, effects, seed_offset in (
            ("with_self", with_effects, 0),
            ("no_self", no_effects, 1000),
            ("with_minus_no_interaction", interactions, 2000),
        ):
            label = f"{outcome}:{name}"
            summaries.append(
                summarize_effects(
                    effects,
                    effect_name=label,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed + seed_offset,
                )
            )
            task_outputs.append(effects.assign(effect_name=label))

    if incidental_ids:
        unit_keys = ["task_id", "scenario", "ratio_id", "replicate"]
        contaminated_units = {
            tuple(row)
            for row in no_self[no_self.request_id.isin(incidental_ids)][unit_keys].itertuples(
                index=False, name=None
            )
        }

        def exclude_units(frame: pd.DataFrame) -> pd.DataFrame:
            keep = [
                tuple(row) not in contaminated_units
                for row in frame[unit_keys].itertuples(index=False, name=None)
            ]
            return frame.loc[keep].copy()

        with_filtered = exclude_units(with_self)
        no_filtered = exclude_units(no_self)
        with_effects = task_scale_effects(
            with_filtered.assign(
                is_primary_outcome=with_filtered.is_primary_outcome.astype(float)
            ),
            "is_primary_outcome",
        )
        no_effects = task_scale_effects(
            no_filtered.assign(is_primary_outcome=no_filtered.is_primary_outcome.astype(float)),
            "is_primary_outcome",
        )
        interactions = interaction_effects(with_effects, no_effects)
        for name, effects, seed_offset in (
            ("with_self", with_effects, 3000),
            ("no_self", no_effects, 4000),
            ("with_minus_no_interaction", interactions, 5000),
        ):
            label = f"is_primary_outcome:{name}:exclude_incidental_text_matches"
            summaries.append(
                summarize_effects(
                    effects,
                    effect_name=label,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed + seed_offset,
                )
            )
            task_outputs.append(effects.assign(effect_name=label))
        audit["incidental_text_sensitivity"] = {
            "request_ids": sorted(incidental_ids),
            "excluded_scale_units": len(contaminated_units),
        }

    summary = pd.concat(summaries, ignore_index=True)
    task_effects = pd.concat(task_outputs, ignore_index=True)
    no_primary = summary[summary.effect_name.eq("is_primary_outcome:no_self")]
    interaction_primary = summary[
        summary.effect_name.eq("is_primary_outcome:with_minus_no_interaction")
    ]
    decisions = decision_table(no_primary, interaction_primary)

    cell_summary = (
        all_rows.groupby(
            [
                "self_condition",
                "scenario",
                "ratio_id",
                "correct_share",
                "multiplier",
                "incoming_degree",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            requests=("request_id", "size"),
            primary_rate=("is_primary_outcome", "mean"),
            correct_rate=("is_correct", "mean"),
            target_rate=("is_target", "mean"),
            other_rate=("is_other", "mean"),
            unparsed_rate=("is_unparsed", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
    )
    cell_summary.to_csv(output / "cell_summary.csv", index=False)
    summary.to_csv(output / "paired_contrasts_and_interactions.csv", index=False)
    task_effects.to_csv(output / "task_effects.csv", index=False)
    plot_comparison(all_rows, output / "self_ablation_response.png")
    manifest = {
        "analysis": "evidence-volume-self-ablation-v1",
        "audit": audit,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "decisions": decisions,
        "interaction_sign": "with-self volume contrast minus no-self volume contrast",
        "claim_limits": [
            "no-self scenario labels denote target/correct selection, not state transitions",
            "message count and total input-token volume remain coupled",
            "the ablation changes both previous text and the minimal update instruction "
            "referring to it",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
