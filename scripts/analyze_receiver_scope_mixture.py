"""Separate readout/internal transitions and analyze incoming answer mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_node_round_adoption import extract_updates, read_json, read_jsonl
from analyze_node_round_transitions import (
    DEFAULT_BOOTSTRAPS,
    DEFAULT_SEED,
    TASK_CHUNK_SIZE,
    add_regimes,
    ratio_interval,
)

ANALYSIS_VERSION = "receiver-scope-mixture-v1"
SCOPE_ORDER = ("internal", "readout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def add_scope_and_mixture(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["receiver_scope"] = np.where(
        result.receiver_is_readout.eq(1), "readout", "internal"
    )
    result["parsed_incoming_count"] = (
        result.incoming_correct_count
        + result.incoming_target_count
        + result.incoming_other_count
    )
    result["incoming_unparsed_count"] = result.incoming_unparsed_count.astype(int)
    result["target_share_parsed"] = np.where(
        result.parsed_incoming_count.gt(0),
        result.incoming_target_count / result.parsed_incoming_count,
        np.nan,
    )
    result["target_share_bin"] = result.target_share_parsed.map(target_share_bin)
    return result


def target_share_bin(value: float) -> str:
    if pd.isna(value):
        return "undefined"
    if value <= 0:
        return "0"
    if value <= 0.25:
        return "(0,.25]"
    if value < 0.5:
        return "(.25,.5)"
    if np.isclose(value, 0.5):
        return ".5"
    if value < 0.75:
        return "(.5,.75)"
    if value < 1:
        return "[.75,1)"
    return "1"


def scope_metric_masks(frame: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series, str]]:
    previous_induced = frame.previous_induced_target_state.eq(1)
    previous_correct = frame.previous_attack_state.eq("correct")
    received_induced = frame.received_induced_target.eq(1)
    current_induced = frame.current_attack_state.eq("target") & ~frame.current_clean_state.eq(
        "target"
    )
    return {
        "received_induced_target_per_update": (
            pd.Series(True, index=frame.index),
            received_induced,
            "P(attack-induced target exposure) among eligible updates",
        ),
        "induced_c_to_t_given_induced_target_exposed": (
            previous_correct & received_induced,
            current_induced,
            "P(attack-attributed C->T | attack-induced target exposed)",
        ),
        "induced_t_to_c": (
            previous_induced,
            frame.current_attack_state.eq("correct"),
            "P(T->C | previous attack-induced T)",
        ),
        "induced_t_to_t": (
            previous_induced,
            current_induced,
            "P(current attack-induced T | previous attack-induced T)",
        ),
        "raw_c_to_o_given_target_exposed": (
            previous_correct & frame.received_target.eq(1),
            frame.current_attack_state.eq("other"),
            "P(C->O | any target exposed); U is separate",
        ),
    }


def scope_sufficient_statistics(
    frame: pd.DataFrame, *, include_round: bool = False
) -> pd.DataFrame:
    rows = []
    keys = ["regime", "stratum", "receiver_scope", "task_id"]
    if include_round:
        keys.append("round_index")
    all_tasks = frame[keys].drop_duplicates()
    for name, (eligible, success, _) in scope_metric_masks(frame).items():
        selected = frame.loc[eligible, keys].copy()
        selected["numerator"] = (eligible & success).loc[eligible].astype(int).to_numpy()
        selected["denominator"] = 1
        grouped = selected.groupby(keys, as_index=False)[["numerator", "denominator"]].sum()
        grouped = all_tasks.merge(grouped, on=keys, how="left", validate="one_to_one")
        grouped[["numerator", "denominator"]] = grouped[
            ["numerator", "denominator"]
        ].fillna(0)
        grouped["metric"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_scope_rates(
    stats: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    descriptions = {
        name: description
        for name, (_, _, description) in scope_metric_masks(
            pd.DataFrame(
                {
                    "previous_induced_target_state": [0],
                    "previous_attack_state": ["correct"],
                    "received_induced_target": [0],
                    "current_attack_state": ["correct"],
                    "current_clean_state": ["correct"],
                    "received_target": [0],
                }
            )
        ).items()
    }
    rows = []
    group_columns = ["regime", "stratum", "receiver_scope"]
    if "round_index" in stats.columns:
        group_columns.append("round_index")
    group_columns.append("metric")
    for key_values, group in stats.groupby(group_columns, sort=True):
        key_map = dict(zip(group_columns, key_values, strict=True))
        regime = str(key_map["regime"])
        stratum = str(key_map["stratum"])
        scope = str(key_map["receiver_scope"])
        metric = str(key_map["metric"])
        point, low, high = ratio_interval(group, rng, replicates)
        row = {
                "regime": regime,
                "stratum": stratum,
                "receiver_scope": scope,
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
        if "round_index" in key_map:
            row["round_index"] = int(key_map["round_index"])
        rows.append(row)
    return pd.DataFrame(rows)


def scope_density_contrasts(
    stats: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    rows = []
    for (regime, scope, metric), selected in stats.groupby(
        ["regime", "receiver_scope", "metric"], sort=True
    ):
        for n in (5, 8):
            strata = sorted(
                [s for s in selected.stratum.unique() if int(s.split("_")[0][1:]) == n],
                key=lambda s: int(s.split("_")[1][1:]),
            )
            for low, high in zip(strata, strata[1:], strict=False):
                low_frame = selected[selected.stratum.eq(low)].set_index("task_id")
                high_frame = selected[selected.stratum.eq(high)].set_index("task_id")
                task_ids = sorted(set(low_frame.index) & set(high_frame.index))
                low_frame = low_frame.loc[task_ids]
                high_frame = high_frame.loc[task_ids]
                if low_frame.denominator.sum() == 0 or high_frame.denominator.sum() == 0:
                    continue
                draws = rng.integers(0, len(task_ids), size=(replicates, len(task_ids)))
                low_den = low_frame.denominator.to_numpy()[draws].sum(axis=1)
                high_den = high_frame.denominator.to_numpy()[draws].sum(axis=1)
                valid = (low_den > 0) & (high_den > 0)
                deltas = (
                    high_frame.numerator.to_numpy()[draws].sum(axis=1)[valid]
                    / high_den[valid]
                    - low_frame.numerator.to_numpy()[draws].sum(axis=1)[valid]
                    / low_den[valid]
                )
                low_point = low_frame.numerator.sum() / low_frame.denominator.sum()
                high_point = high_frame.numerator.sum() / high_frame.denominator.sum()
                rows.append(
                    {
                        "regime": regime,
                        "n": n,
                        "receiver_scope": scope,
                        "metric": metric,
                        "low_stratum": low,
                        "high_stratum": high,
                        "difference": float(high_point - low_point),
                        "ci95_low": float(np.quantile(deltas, 0.025)),
                        "ci95_high": float(np.quantile(deltas, 0.975)),
                        "ci_excludes_zero": bool(
                            np.quantile(deltas, 0.025) > 0
                            or np.quantile(deltas, 0.975) < 0
                        ),
                        "tasks": len(task_ids),
                    }
                )
    return pd.DataFrame(rows)


def mixture_rows(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = (
        frame.previous_attack_state.eq("correct")
        & frame.received_induced_target.eq(1)
    )
    columns = [
        "regime",
        "stratum",
        "receiver_scope",
        "task_id",
        "graph_id",
        "round_index",
        "incoming_correct_count",
        "incoming_target_count",
        "incoming_other_count",
        "incoming_unparsed_count",
        "parsed_incoming_count",
        "target_share_parsed",
        "target_share_bin",
    ]
    result = frame.loc[eligible, columns].copy()
    result["adopted"] = (
        frame.loc[eligible, "current_attack_state"].eq("target")
        & ~frame.loc[eligible, "current_clean_state"].eq("target")
    ).astype(int)
    return result


def mixture_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "mean_incoming_correct": "incoming_correct_count",
        "mean_incoming_target": "incoming_target_count",
        "mean_incoming_other": "incoming_other_count",
        "mean_incoming_unparsed": "incoming_unparsed_count",
        "mean_parsed_incoming": "parsed_incoming_count",
        "mean_target_share": "target_share_parsed",
    }
    keys = ["regime", "stratum", "receiver_scope", "task_id"]
    rows = []
    all_tasks = frame[keys].drop_duplicates()
    for metric, column in definitions.items():
        selected = frame.loc[frame[column].notna(), [*keys, column]].copy()
        selected["numerator"] = selected[column].astype(float)
        selected["denominator"] = 1
        grouped = selected.groupby(keys, as_index=False)[
            ["numerator", "denominator"]
        ].sum()
        grouped = all_tasks.merge(grouped, on=keys, how="left", validate="one_to_one")
        grouped[["numerator", "denominator"]] = grouped[
            ["numerator", "denominator"]
        ].fillna(0)
        grouped["metric"] = metric
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_ratio_metrics(
    stats: pd.DataFrame, rng: np.random.Generator, replicates: int
) -> pd.DataFrame:
    rows = []
    for keys, group in stats.groupby(
        ["regime", "stratum", "receiver_scope", "metric"], sort=True
    ):
        regime, stratum, scope, metric = keys
        point, low, high = ratio_interval(group, rng, replicates)
        rows.append(
            {
                "regime": regime,
                "stratum": stratum,
                "receiver_scope": scope,
                "metric": metric,
                "numerator": float(group.numerator.sum()),
                "denominator": int(group.denominator.sum()),
                "estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "tasks": int(group.task_id.nunique()),
                "ci_scope": "task bootstrap conditional on selected graphs",
            }
        )
    return pd.DataFrame(rows)


def summarize_mixture(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["regime", "stratum", "receiver_scope"]
    return frame.groupby(keys, as_index=False).agg(
        updates=("adopted", "size"),
        adoptions=("adopted", "sum"),
        adoption_rate=("adopted", "mean"),
        mean_incoming_correct=("incoming_correct_count", "mean"),
        mean_incoming_target=("incoming_target_count", "mean"),
        mean_incoming_other=("incoming_other_count", "mean"),
        mean_incoming_unparsed=("incoming_unparsed_count", "mean"),
        mean_parsed_incoming=("parsed_incoming_count", "mean"),
        mean_target_share=("target_share_parsed", "mean"),
    )


def adoption_by_composition(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "regime",
        "stratum",
        "receiver_scope",
        "round_index",
        "incoming_target_count",
        "incoming_correct_count",
        "incoming_other_count",
        "incoming_unparsed_count",
    ]
    return frame.groupby(keys, as_index=False).agg(
        updates=("adopted", "size"),
        adoptions=("adopted", "sum"),
        adoption_rate=("adopted", "mean"),
        tasks=("task_id", "nunique"),
        graphs=("graph_id", "nunique"),
        mean_target_share=("target_share_parsed", "mean"),
    )


def adoption_by_share_bin(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "regime",
        "stratum",
        "receiver_scope",
        "round_index",
        "target_share_bin",
    ]
    return frame.groupby(keys, as_index=False).agg(
        updates=("adopted", "size"),
        adoptions=("adopted", "sum"),
        adoption_rate=("adopted", "mean"),
        tasks=("task_id", "nunique"),
        mean_target_share=("target_share_parsed", "mean"),
        mean_parsed_incoming=("parsed_incoming_count", "mean"),
    )


def mixture_decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reweight a pooled exact-composition transition law by each stratum's mixture."""
    rows = []
    contrasts = []
    cell_keys = [
        "round_index",
        "incoming_target_count",
        "incoming_correct_count",
        "incoming_other_count",
        "incoming_unparsed_count",
    ]
    for (regime, scope, n), group in frame.assign(
        n=frame.stratum.str.extract(r"n(\d+)_", expand=False).astype(int)
    ).groupby(["regime", "receiver_scope", "n"], sort=True):
        law = group.groupby(cell_keys, as_index=False).agg(
            pooled_adoptions=("adopted", "sum"), pooled_updates=("adopted", "size")
        )
        law["pooled_rate"] = law.pooled_adoptions / law.pooled_updates
        for stratum, selected in group.groupby("stratum", sort=True):
            cells = selected.groupby(cell_keys, as_index=False).agg(
                updates=("adopted", "size"), adoptions=("adopted", "sum")
            )
            cells = cells.merge(law, on=cell_keys, how="left", validate="one_to_one")
            observed = cells.adoptions.sum() / cells.updates.sum()
            predicted = float(
                (cells.updates * cells.pooled_rate).sum() / cells.updates.sum()
            )
            rows.append(
                {
                    "regime": regime,
                    "receiver_scope": scope,
                    "n": n,
                    "stratum": stratum,
                    "updates": int(cells.updates.sum()),
                    "observed_adoption": float(observed),
                    "mixture_predicted_adoption": predicted,
                    "within_mixture_residual": float(observed - predicted),
                    "law": "pooled within n/scope over round x exact (T,C,O,U)",
                }
            )
        result = pd.DataFrame(
            [
                row
                for row in rows
                if row["regime"] == regime
                and row["receiver_scope"] == scope
                and row["n"] == n
            ]
        )
        ordered = result.sort_values(
            "stratum",
            key=lambda x: x.str.extract(r"m(\d+)", expand=False).astype(int),
        )
        for low, high in zip(
            ordered.itertuples(), ordered.iloc[1:].itertuples(), strict=False
        ):
            observed_delta = high.observed_adoption - low.observed_adoption
            predicted_delta = (
                high.mixture_predicted_adoption - low.mixture_predicted_adoption
            )
            contrasts.append(
                {
                    "regime": regime,
                    "receiver_scope": scope,
                    "n": n,
                    "low_stratum": low.stratum,
                    "high_stratum": high.stratum,
                    "observed_delta": observed_delta,
                    "mixture_predicted_delta": predicted_delta,
                    "residual_delta": observed_delta - predicted_delta,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    status = read_json(args.run_root / "orchestrator_status.json")
    stat_parts: list[pd.DataFrame] = []
    round_stat_parts: list[pd.DataFrame] = []
    mixture_parts: list[pd.DataFrame] = []
    mixture_stat_parts: list[pd.DataFrame] = []
    audits = []
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        root = args.run_root / "strata" / stratum
        graph_path = root / "selected_graphs.jsonl"
        if not graph_path.exists():
            graph_path = root / "batch" / "inputs" / "graphs.jsonl"
        graph_records = read_jsonl(graph_path)
        task_values = [
            str(item["task_id"])
            for item in read_jsonl(root / "batch/inputs/tasks.jsonl")
        ]
        for graph_record in graph_records:
            graph_id = str(graph_record["graph_id"])
            for start in range(0, len(task_values), TASK_CHUNK_SIZE):
                chunk_ids = set(task_values[start : start + TASK_CHUNK_SIZE])
                raw, audit = extract_updates(
                    args.run_root,
                    {"strata": [descriptor]},
                    graph_ids={graph_id},
                    task_ids=chunk_ids,
                )
                if not audit["passed"]:
                    raise RuntimeError("integrity audit failed: " + "; ".join(audit["errors"][:10]))
                frame = add_scope_and_mixture(add_regimes(raw))
                stat_parts.append(scope_sufficient_statistics(frame))
                round_stat_parts.append(
                    scope_sufficient_statistics(frame, include_round=True)
                )
                chunk_mixture = mixture_rows(frame)
                mixture_parts.append(chunk_mixture)
                mixture_stat_parts.append(
                    mixture_sufficient_statistics(chunk_mixture)
                )
                audits.append(
                    {
                        "stratum": stratum,
                        "graph_id": graph_id,
                        "task_chunk_start": start,
                        "paired_conditions": audit["paired_conditions"],
                        "eligible_updates": audit["eligible_updates"],
                        "readout_updates_fixed_t3": int(
                            (frame.regime.eq("fixed_t3") & frame.receiver_scope.eq("readout")).sum()
                        ),
                        "internal_updates_fixed_t3": int(
                            (
                                frame.regime.eq("fixed_t3")
                                & frame.receiver_scope.eq("internal")
                            ).sum()
                        ),
                    }
                )
                del raw, frame
    stats = (
        pd.concat(stat_parts, ignore_index=True)
        .groupby(
            ["regime", "stratum", "receiver_scope", "task_id", "metric"],
            as_index=False,
        )[["numerator", "denominator"]]
        .sum()
    )
    round_stats = (
        pd.concat(round_stat_parts, ignore_index=True)
        .groupby(
            [
                "regime",
                "stratum",
                "receiver_scope",
                "task_id",
                "round_index",
                "metric",
            ],
            as_index=False,
        )[["numerator", "denominator"]]
        .sum()
    )
    mixture_stats = (
        pd.concat(mixture_stat_parts, ignore_index=True)
        .groupby(
            ["regime", "stratum", "receiver_scope", "task_id", "metric"],
            as_index=False,
        )[["numerator", "denominator"]]
        .sum()
    )
    mixture = pd.concat(mixture_parts, ignore_index=True)
    rng = np.random.default_rng(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output / "task_scope_sufficient_statistics.csv", index=False)
    round_stats.to_csv(
        output / "task_scope_round_sufficient_statistics.csv", index=False
    )
    mixture_stats.to_csv(
        output / "task_mixture_sufficient_statistics.csv", index=False
    )
    summarize_scope_rates(stats, rng, args.bootstrap_replicates).to_csv(
        output / "receiver_scope_rates.csv", index=False
    )
    scope_density_contrasts(stats, rng, args.bootstrap_replicates).to_csv(
        output / "receiver_scope_density_contrasts.csv", index=False
    )
    summarize_scope_rates(round_stats, rng, args.bootstrap_replicates).to_csv(
        output / "receiver_scope_rates_by_round.csv", index=False
    )
    summarize_ratio_metrics(mixture_stats, rng, args.bootstrap_replicates).to_csv(
        output / "incoming_mixture_metric_rates.csv", index=False
    )
    scope_density_contrasts(
        mixture_stats, rng, args.bootstrap_replicates
    ).to_csv(output / "incoming_mixture_density_contrasts.csv", index=False)
    summarize_mixture(mixture).to_csv(output / "incoming_mixture_summary.csv", index=False)
    adoption_by_composition(mixture).to_csv(
        output / "adoption_by_exact_composition.csv", index=False
    )
    adoption_by_share_bin(mixture).to_csv(
        output / "adoption_by_target_share_bin.csv", index=False
    )
    decomposition, decomposition_contrasts = mixture_decomposition(mixture)
    decomposition.to_csv(output / "mixture_decomposition.csv", index=False)
    decomposition_contrasts.to_csv(
        output / "mixture_decomposition_contrasts.csv", index=False
    )
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "passed": True,
        "graph_task_chunks": len(audits),
        "paired_conditions": sum(item["paired_conditions"] for item in audits),
        "eligible_updates": sum(item["eligible_updates"] for item in audits),
        "readout_updates_fixed_t3": sum(
            item["readout_updates_fixed_t3"] for item in audits
        ),
        "internal_updates_fixed_t3": sum(
            item["internal_updates_fixed_t3"] for item in audits
        ),
        "chunks": audits,
    }
    (output / "integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(args.run_root.resolve()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "primary_regime": "fixed_t3",
        "receiver_scopes": list(SCOPE_ORDER),
        "mixture_definition": "#T / (#C + #T + #O); U retained separately",
        "claim_limits": [
            "task bootstrap is conditional on selected graphs",
            "incoming mixture is post-treatment",
            "mixture decomposition is descriptive rather than causal mediation",
            "residual density association is not proof of an LLM-specific mechanism",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "integrity": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
