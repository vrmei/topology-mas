#!/usr/bin/env python3
"""Analyze paired evidence-volume interventions with task-cluster uncertainty."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser.parse_args()


def validate_outcomes(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "request_id",
        "task_id",
        "scenario",
        "ratio_id",
        "multiplier",
        "replicate",
        "generation_seed",
        "previous_stimulus_id",
        "peer_stimulus_ids",
        "next_state",
        "is_primary_outcome",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"outcomes missing columns: {sorted(missing)}")
    if frame.request_id.duplicated().any():
        raise ValueError("duplicate request IDs")
    expected = 50 * 2 * 5 * 3 * 5
    if len(frame) != expected:
        raise ValueError(f"expected {expected} outcomes, found {len(frame)}")

    nested_failures = 0
    pairing_failures = 0
    keys = ["task_id", "scenario", "ratio_id", "replicate"]
    for _, group in frame.groupby(keys, sort=False):
        if set(group.multiplier) != {1, 2, 3}:
            pairing_failures += 1
            continue
        if group.previous_stimulus_id.nunique() != 1 or group.generation_seed.nunique() != 1:
            pairing_failures += 1
        sets = {
            int(row.multiplier): set(row.peer_stimulus_ids) for row in group.itertuples(index=False)
        }
        if not sets[1].issubset(sets[2]) or not sets[2].issubset(sets[3]):
            nested_failures += 1
    if pairing_failures or nested_failures:
        raise ValueError(
            f"paired-plan audit failed: pairing={pairing_failures}, nested={nested_failures}"
        )
    return {
        "rows": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "scenarios": sorted(frame.scenario.unique()),
        "ratios": sorted(frame.ratio_id.unique()),
        "multipliers": sorted(int(value) for value in frame.multiplier.unique()),
        "pairing_failures": pairing_failures,
        "nested_set_failures": nested_failures,
    }


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(
        parsed_primary=np.where(frame.is_unparsed, np.nan, frame.is_primary_outcome)
    )
    columns = ["scenario", "ratio_id", "correct_share", "multiplier"]
    return (
        frame.groupby(columns, as_index=False, observed=True)
        .agg(
            requests=("request_id", "size"),
            tasks=("task_id", "nunique"),
            incoming_degree=("incoming_degree", "first"),
            primary_rate=("is_primary_outcome", "mean"),
            primary_rate_given_parsed=("parsed_primary", "mean"),
            correct_rate=("is_correct", "mean"),
            target_rate=("is_target", "mean"),
            other_rate=("is_other", "mean"),
            unparsed_rate=("is_unparsed", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            degroot_primary_mass=("degroot_primary_mass", "mean"),
            peer_only_primary_mass=("peer_only_primary_mass", "mean"),
        )
        .sort_values(columns)
    )


def task_cluster_interval(
    task_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, float]:
    values = np.asarray(task_values, dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return estimate, float(low), float(high), p_value


def paired_contrasts(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    outcome_name: str = "primary_unconditional",
    drop_incomplete: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = frame.pivot_table(
        index=["task_id", "scenario", "ratio_id", "replicate"],
        columns="multiplier",
        values="is_primary_outcome",
        aggfunc="first",
    ).reset_index()
    incomplete = pivot[[1, 2, 3]].isna().any(axis=1)
    if incomplete.any() and not drop_incomplete:
        raise ValueError("paired outcome pivot is incomplete")
    pivot = pivot.loc[~incomplete].copy()
    pivot[[1, 2, 3]] = pivot[[1, 2, 3]].astype(float)

    task_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for scenario in sorted(pivot.scenario.unique()):
        scenario_frame = pivot[pivot.scenario.eq(scenario)]
        ratio_levels = [*sorted(scenario_frame.ratio_id.unique()), "pooled"]
        for ratio_id in ratio_levels:
            selected = (
                scenario_frame
                if ratio_id == "pooled"
                else scenario_frame[scenario_frame.ratio_id.eq(ratio_id)]
            )
            for high, low in CONTRASTS:
                work = selected.assign(difference=selected[high] - selected[low])
                per_task = work.groupby("task_id", observed=True).difference.mean()
                estimate, ci_low, ci_high, p_value = task_cluster_interval(
                    per_task.to_numpy(),
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed
                    + sum(ord(char) for char in f"{scenario}:{ratio_id}:{high}:{low}"),
                )
                equivalence = ci_low > -EQUIVALENCE_MARGIN and ci_high < EQUIVALENCE_MARGIN
                directional = ci_low > 0 or ci_high < 0
                result_rows.append(
                    {
                        "scenario": scenario,
                        "outcome": outcome_name,
                        "ratio_id": ratio_id,
                        "contrast": f"{high}x-{low}x",
                        "tasks": len(per_task),
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_p_two_sided": p_value,
                        "equivalence_margin": EQUIVALENCE_MARGIN,
                        "practically_equivalent": equivalence,
                        "directional": directional,
                        "classification": (
                            "practical_equivalence"
                            if equivalence
                            else "directional_effect"
                            if directional
                            else "inconclusive"
                        ),
                    }
                )
                for task_id, value in per_task.items():
                    task_rows.append(
                        {
                            "task_id": task_id,
                            "scenario": scenario,
                            "outcome": outcome_name,
                            "ratio_id": ratio_id,
                            "contrast": f"{high}x-{low}x",
                            "task_mean_difference": value,
                        }
                    )
    return pd.DataFrame(result_rows), pd.DataFrame(task_rows)


def disposition(contrasts: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario in sorted(contrasts.scenario.unique()):
        work = contrasts[(contrasts.scenario == scenario) & (contrasts.ratio_id == "pooled")]
        main = work[work.contrast == "3x-1x"].iloc[0]
        early = work[work.contrast == "2x-1x"].iloc[0]
        late = work[work.contrast == "3x-2x"].iloc[0]
        if bool(main.practically_equivalent):
            label = "ratio_dominant_within_5pp"
        elif bool(early.directional) and bool(late.practically_equivalent):
            label = "saturation_candidate"
        elif bool(main.directional):
            label = "absolute_volume_effect"
        else:
            label = "inconclusive"
        result[scenario] = {
            "classification": label,
            "primary_3x_minus_1x": {
                "estimate": float(main.estimate),
                "ci_low": float(main.ci_low),
                "ci_high": float(main.ci_high),
            },
            "decision_note": (
                "classification uses the frozen +/-5 percentage-point equivalence margin; "
                "ratio-specific heterogeneity remains diagnostic"
            ),
        }
    return result


def plot_response(cells: pd.DataFrame, output: Path) -> None:
    scenarios = ["attack_adoption", "benign_correction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, cells.ratio_id.nunique()))
    for axis, scenario in zip(axes, scenarios, strict=True):
        work = cells[cells.scenario.eq(scenario)]
        for color, (_ratio_id, group) in zip(
            colors, work.groupby("ratio_id", sort=True), strict=True
        ):
            group = group.sort_values("incoming_degree")
            label = f"{100 * float(group.correct_share.iloc[0]):.0f}% C"
            axis.plot(
                group.incoming_degree,
                group.primary_rate,
                "o-",
                color=color,
                label=label,
            )
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_xlabel("Number of distinct peer rationales")
        axis.set_ylabel("Primary transition probability")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("One-step response at fixed evidence ratio and manipulated absolute volume")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(run_dir / "outcomes.jsonl")
    frame = pd.DataFrame(rows)
    audit = validate_outcomes(frame)
    # Two deliberately simple classical references. Peer-only equal-weight mixing is
    # invariant to evidence-volume scaling at a fixed composition. A DeGroot-style
    # update that gives the receiver's previous state one equal-weight vote is not:
    # increasing peer volume dilutes that self vote. These are exposure scores, not
    # calibrated categorical transition probabilities.
    frame = frame.assign(
        degroot_primary_mass=np.where(
            frame.scenario.eq("attack_adoption"),
            frame.error_count / (1.0 + frame.incoming_degree),
            frame.correct_count / (1.0 + frame.incoming_degree),
        ),
        peer_only_primary_mass=np.where(
            frame.scenario.eq("attack_adoption"),
            1.0 - frame.correct_share,
            frame.correct_share,
        ),
    )
    cells = summarize_cells(frame)
    contrasts, task_effects = paired_contrasts(
        frame,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    unparsed_contrasts, unparsed_task_effects = paired_contrasts(
        frame.assign(is_primary_outcome=frame.is_unparsed),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1000,
        outcome_name="unparsed",
    )
    parsed_diagnostic = frame.assign(
        is_primary_outcome=np.where(
            frame.is_unparsed, np.nan, frame.is_primary_outcome.astype(float)
        )
    )
    parsed_contrasts, parsed_task_effects = paired_contrasts(
        parsed_diagnostic,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 2000,
        outcome_name="primary_given_parsed_post_treatment",
        drop_incomplete=True,
    )
    degroot_contrasts, degroot_task_effects = paired_contrasts(
        frame.assign(is_primary_outcome=frame.degroot_primary_mass),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 3000,
        outcome_name="degroot_self_plus_peers_exposure",
    )
    peer_only_contrasts, peer_only_task_effects = paired_contrasts(
        frame.assign(is_primary_outcome=frame.peer_only_primary_mass),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 4000,
        outcome_name="peer_only_ratio_exposure",
    )
    all_contrasts = pd.concat(
        [contrasts, unparsed_contrasts, parsed_contrasts], ignore_index=True
    )
    all_task_effects = pd.concat(
        [task_effects, unparsed_task_effects, parsed_task_effects],
        ignore_index=True,
    )
    classical_exposure_contrasts = pd.concat(
        [degroot_contrasts, peer_only_contrasts], ignore_index=True
    )
    classical_exposure_task_effects = pd.concat(
        [degroot_task_effects, peer_only_task_effects], ignore_index=True
    )
    classical_comparison = contrasts[
        ["scenario", "ratio_id", "contrast", "estimate", "ci_low", "ci_high"]
    ].rename(
        columns={
            "estimate": "observed_transition_contrast",
            "ci_low": "observed_transition_ci_low",
            "ci_high": "observed_transition_ci_high",
        }
    ).merge(
        degroot_contrasts[
            ["scenario", "ratio_id", "contrast", "estimate"]
        ].rename(columns={"estimate": "degroot_exposure_contrast"}),
        on=["scenario", "ratio_id", "contrast"],
        validate="one_to_one",
    ).merge(
        peer_only_contrasts[
            ["scenario", "ratio_id", "contrast", "estimate"]
        ].rename(columns={"estimate": "peer_only_exposure_contrast"}),
        on=["scenario", "ratio_id", "contrast"],
        validate="one_to_one",
    )
    decisions = disposition(contrasts)
    cells.to_csv(output / "cell_summary.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    task_effects.to_csv(output / "task_contrast_effects.csv", index=False)
    all_contrasts.to_csv(output / "paired_contrasts_all_outcomes.csv", index=False)
    all_task_effects.to_csv(output / "task_contrast_effects_all_outcomes.csv", index=False)
    classical_exposure_contrasts.to_csv(
        output / "classical_exposure_contrasts.csv", index=False
    )
    classical_exposure_task_effects.to_csv(
        output / "classical_exposure_task_effects.csv", index=False
    )
    classical_comparison.to_csv(output / "classical_exposure_comparison.csv", index=False)
    frame.drop(columns=["raw_output"], errors="ignore").to_csv(
        output / "outcomes_compact.csv.gz", index=False, compression="gzip"
    )
    plot_response(cells, output / "evidence_volume_response.png")
    manifest = {
        "analysis": "evidence-volume-intervention-analysis-v1",
        "run_dir": str(run_dir),
        "audit": audit,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "disposition": decisions,
        "sensitivity_outputs": {
            "unparsed": "paired_contrasts_all_outcomes.csv",
            "primary_given_parsed": (
                "diagnostic only because parsed status is measured after intervention"
            ),
        },
        "classical_baselines": {
            "peer_only_ratio": (
                "scale-invariant equal-weight peer exposure; predicts zero contrast"
            ),
            "degroot_self_plus_peers": (
                "one equal-weight self-state plus all peers; increasing peer volume "
                "dilutes self inertia"
            ),
            "interpretation_limit": (
                "both are continuous exposure scores, not calibrated categorical "
                "transition probabilities; they are shown side by side and must not "
                "be subtracted as if they shared a response scale"
            ),
        },
        "claim_limits": [
            "the intervention changes both message count and total input-token volume",
            "the five replicates combine message-set and generation randomness",
            "results are conditional on Llama-3.1-8B, GSM8K, and the frozen node prompt",
            "practical equivalence within 5pp is not exact equality or arbitrary-scale invariance",
            "a fixed peer ratio does not hold receiver self-weight constant under "
            "equal-weight DeGroot mixing",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["disposition"], indent=2))


if __name__ == "__main__":
    main()
