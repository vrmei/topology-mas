"""Recursively validate frozen n=5 local laws at real n=6,7,8,10 endpoints."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_recursive_rollout import dense_lookup, mean_field_rollout, query_frame
from analyze_ctou_scale_transfer import attach_endpoint_losses, graph_and_curve_outputs

from topology_mas.simulation.ctou_scale import CTOU_STATES, ctou_design_matrix

MODELS = (
    "proportions",
    "absolute_counts",
    "counts_plus_proportions",
    "proportions_saturating_volume_k2",
)
SIZES = (6, 7, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5-n8-cache", type=Path, required=True)
    parser.add_argument("--n6-n7-cache", type=Path, required=True)
    parser.add_argument("--n10-cache", type=Path, required=True)
    parser.add_argument("--frozen-laws", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_selected(path: Path, sizes: set[int]) -> dict[int, dict[str, Any]]:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    result: dict[int, dict[str, Any]] = {}
    for size in sizes:
        result[size] = {}
        for condition in ("attack", "clean"):
            cases = bundle[f"{condition}_cases"]
            result[size][f"{condition}_cases"] = cases.loc[cases.n.eq(size)].copy()
            graph_ids = set(result[size][f"{condition}_cases"].graph_id.astype(str))
            result[size][f"{condition}_graphs"] = {
                str(key): value
                for key, value in bundle[f"{condition}_graphs"].items()
                if str(key) in graph_ids
            }
    return result


def aligned_probability(model: Any, frame: pd.DataFrame, variant: str) -> np.ndarray:
    raw = model.predict_proba(ctou_design_matrix(frame, variant))
    result = np.zeros((len(frame), len(CTOU_STATES)), dtype=np.float32)
    result[:, model.classes_.astype(int)] = raw
    return result


def build_lookup(model: Any, variant: str, maximum_neighbors: int, horizon: int) -> np.ndarray:
    query = query_frame(maximum_neighbors, horizon)
    return dense_lookup(
        query,
        aligned_probability(model, query, variant),
        horizon,
        maximum_neighbors,
    )


def evaluate(
    *,
    sources: dict[int, dict[str, Any]],
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for size in SIZES:
        for condition in ("attack", "clean"):
            cases = sources[size][f"{condition}_cases"]
            graphs = sources[size][f"{condition}_graphs"]
            for fold in range(5):
                selected = cases.loc[cases.task_fold.eq(fold)]
                fold_models = frozen["condition_fold_models"][condition][fold]
                lookups = {
                    variant: build_lookup(
                        fold_models[variant],
                        variant,
                        maximum_neighbors=size - 1,
                        horizon=3,
                    )
                    for variant in MODELS
                }
                cache: dict[tuple[object, ...], np.ndarray] = {}
                for case in selected.itertuples(index=False):
                    graph = graphs[str(case.graph_id)]
                    attack_node = int(case.attack_node) if condition == "attack" else None
                    readout = int(graph["readout_node"])
                    round0_correct = int(tuple(case.initial_states)[readout] == 0)
                    base = {
                        "condition": condition,
                        "task_id": str(case.task_id),
                        "graph_id": str(case.graph_id),
                        "attack_node": int(case.attack_node),
                        "n": int(case.n),
                        "m": int(case.m),
                        "rho": int(case.m) / ((int(case.n) - 1) ** 2),
                        "task_fold": fold,
                        "actual_state": str(case.actual_state),
                        "actual_state_index": int(case.actual_state_index),
                        "actual_target": int(case.actual_target),
                        "actual_correct": int(case.actual_correct),
                        "round0_correct": round0_correct,
                    }
                    for variant, lookup in lookups.items():
                        key = (
                            variant,
                            str(case.graph_id),
                            attack_node,
                            tuple(case.initial_states),
                        )
                        probability = cache.get(key)
                        if probability is None:
                            probability = mean_field_rollout(
                                graph=graph,
                                initial_states=tuple(case.initial_states),
                                attack_node=attack_node,
                                model=variant,
                                lookup=lookup,
                            )
                            cache[key] = probability
                        rows.append(
                            {
                                **base,
                                "model": variant,
                                **{
                                    f"p_{state}": float(probability[index])
                                    for index, state in enumerate(CTOU_STATES)
                                },
                            }
                        )
                audits.append(
                    {
                        "n": size,
                        "condition": condition,
                        "task_fold": fold,
                        "cases": len(selected),
                        "tasks": selected.task_id.nunique(),
                        "graphs": selected.graph_id.nunique(),
                    }
                )
    return attach_endpoint_losses(pd.DataFrame(rows)), pd.DataFrame(audits)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.frozen_laws.open("rb") as handle:
        frozen = pickle.load(handle)
    sources: dict[int, dict[str, Any]] = {}
    for path, sizes in (
        (args.n5_n8_cache, {8}),
        (args.n6_n7_cache, {6, 7}),
        (args.n10_cache, {10}),
    ):
        sources.update(load_selected(path, sizes))
    print("stage=recursive_anchor_validation", flush=True)
    endpoints, audits = evaluate(sources=sources, frozen=frozen)
    graph, curves, metrics = graph_and_curve_outputs(endpoints)
    endpoints.to_csv(args.output_dir / "endpoint_anchor_predictions.csv.gz", index=False)
    audits.to_csv(args.output_dir / "endpoint_anchor_audit.csv", index=False)
    graph.to_csv(args.output_dir / "endpoint_anchor_graphs.csv", index=False)
    curves.to_csv(args.output_dir / "endpoint_anchor_curves.csv", index=False)
    metrics.to_csv(args.output_dir / "endpoint_anchor_metrics.csv", index=False)

    primary = metrics.loc[
        metrics.level.eq("graph")
        & metrics.quantity.isin(["utility", "robustness"])
        & metrics.model.isin(["proportions", "proportions_saturating_volume_k2"])
    ]
    average_mae = primary.groupby("model").mae.mean()
    selected = "proportions_saturating_volume_k2"
    mae_ratio = float(average_mae.loc[selected] / average_mae.loc["proportions"])
    selected_rows = primary.loc[primary.model.eq(selected)]
    gate = {
        "selected_model": selected,
        "mean_utility_robustness_graph_mae_ratio_vs_proportions": mae_ratio,
        "minimum_utility_robustness_graph_spearman": float(selected_rows.spearman.min()),
        "passed": bool(mae_ratio <= 1.10 and selected_rows.spearman.min() >= 0),
        "thresholds": {
            "maximum_mae_ratio": 1.10,
            "minimum_spearman": 0.0,
        },
    }
    atomic_json(args.output_dir / "endpoint_anchor_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
