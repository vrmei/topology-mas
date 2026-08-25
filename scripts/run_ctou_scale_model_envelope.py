"""Run a fixed-graph CTOU local-law envelope across n=5..50."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from topology_mas.simulation.ctou_scale import (
    COUNT_COLUMNS,
    ctou_design_matrix,
    extract_round_zero_groups,
    fit_hierarchical_round_zero,
)
from topology_mas.simulation.graph_sampling import sample_backbone_augmented_graph
from topology_mas.simulation.rollout import expected_composition_rollout

VARIANTS = (
    "proportions",
    "absolute_counts",
    "counts_plus_proportions",
    "proportions_saturating_volume_k2",
)
DENSITIES = (0.0, 0.5, 1.0)
REAL_SIZES = (5, 6, 7, 8, 10)
SEED = 20_260_825
UPDATE_COLUMNS = (
    "task_id",
    "n",
    "previous_attack_state",
    "round_index",
    *COUNT_COLUMNS,
    "current_state_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5-n8-cache", type=Path, required=True)
    parser.add_argument("--n6-n7-cache", type=Path, required=True)
    parser.add_argument("--n10-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swap-steps", type=int, default=200)
    return parser.parse_args()


def load_sources(paths: tuple[Path, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    updates: dict[str, list[pd.DataFrame]] = {"attack": [], "clean": []}
    round0: list[pd.DataFrame] = []
    for path in paths:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        for condition in updates:
            updates[condition].append(bundle[f"{condition}_updates"][list(UPDATE_COLUMNS)])
        round0.append(extract_round_zero_groups(bundle["clean_cases"]))
    return (
        {key: pd.concat(parts, ignore_index=True) for key, parts in updates.items()},
        pd.concat(round0, ignore_index=True).drop_duplicates(["task_id", "graph_id", "n"]),
    )


def fit_predictor(
    frame: pd.DataFrame,
    *,
    variant: str,
    sizes: set[int],
    balance: bool,
):
    selected = frame.loc[frame.n.isin(sizes)].reset_index(drop=True)
    sample_weight = None
    if balance:
        count = selected.groupby(["n", "task_id"]).size()
        keys = pd.MultiIndex.from_frame(selected[["n", "task_id"]])
        sample_weight = 1.0 / count.reindex(keys).to_numpy(float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=0)
    model.fit(
        ctou_design_matrix(selected, variant),
        selected.current_state_index.to_numpy(int),
        sample_weight=sample_weight,
    )

    def predict(input_frame: pd.DataFrame) -> np.ndarray:
        raw = model.predict_proba(ctou_design_matrix(input_frame, variant))
        output = np.zeros((len(input_frame), 4), dtype=float)
        output[:, model.classes_.astype(int)] = raw
        return output

    return predict


def initial_marginals(initializer: Any, task_ids: list[str], n: int) -> np.ndarray:
    output = np.zeros((len(task_ids), n, 4), dtype=float)
    for index, task_id in enumerate(task_ids):
        correct, other, unparsed = initializer.mean_for_task(task_id)
        output[index, :, 0] = correct
        output[index, :, 2] = other
        output[index, :, 3] = unparsed
    return output


def edge_count(n: int, density: float) -> int:
    return int(round((n - 1) + density * ((n - 1) ** 2 - (n - 1))))


def simulate(
    *,
    graph: Any,
    task_ids: list[str],
    initializer: Any,
    clean_predictor: Any,
    attack_predictor: Any,
) -> dict[str, float]:
    initial = initial_marginals(initializer, task_ids, graph.node_count)
    clean = expected_composition_rollout(
        graph=graph,
        initial_marginals=initial,
        attack_nodes=np.full(len(task_ids), -1, dtype=int),
        predictor=clean_predictor,
    )
    positions = np.arange(graph.node_count - 1, dtype=int)
    attack_initial = np.repeat(initial[:, None], len(positions), axis=1).reshape(
        len(task_ids) * len(positions), graph.node_count, 4
    )
    attack = expected_composition_rollout(
        graph=graph,
        initial_marginals=attack_initial,
        attack_nodes=np.tile(positions, len(task_ids)),
        predictor=attack_predictor,
    ).reshape(len(task_ids), len(positions), 4)
    utility = float(clean[:, 0].mean())
    robustness = float(attack[:, :, 0].mean())
    return {
        "utility": utility,
        "robustness": robustness,
        "target_risk": float(attack[:, :, 1].mean()),
        "attack_penalty": utility - robustness,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates, round0 = load_sources(
        (args.n5_n8_cache, args.n6_n7_cache, args.n10_cache)
    )
    versions: dict[str, dict[str, Any]] = {}
    for version, sizes, balance in (
        ("strict_n5", {5}, False),
        ("calibrated_n5_n6_n7_n8_n10", set(REAL_SIZES), True),
    ):
        versions[version] = {
            "initializer": fit_hierarchical_round_zero(
                round0.loc[round0.n.isin(sizes)], required_sizes=sizes
            ),
            "models": {
                variant: {
                    condition: fit_predictor(
                        updates[condition],
                        variant=variant,
                        sizes=sizes,
                        balance=balance,
                    )
                    for condition in ("clean", "attack")
                }
                for variant in VARIANTS
            },
        }
    task_ids = sorted(versions["strict_n5"]["initializer"].task_means)
    rows: list[dict[str, object]] = []
    for n in range(5, 51):
        for density in DENSITIES:
            m = edge_count(n, density)
            graph, audit = sample_backbone_augmented_graph(
                node_count=n,
                edge_count=m,
                horizon=3,
                seed=SEED,
                sample_index=0,
                swap_steps=args.swap_steps,
            )
            for version, fitted in versions.items():
                for variant, models in fitted["models"].items():
                    metrics = simulate(
                        graph=graph,
                        task_ids=task_ids,
                        initializer=fitted["initializer"],
                        clean_predictor=models["clean"],
                        attack_predictor=models["attack"],
                    )
                    rows.append(
                        {
                            "version": version,
                            "variant": variant,
                            "n": n,
                            "m": m,
                            "density": density,
                            "average_degree": m / (n - 1),
                            "graph_id": graph.graph_id,
                            "swap_acceptance_rate": audit.acceptance_rate,
                            **metrics,
                        }
                    )
        print(f"stage=envelope n={n}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "model_envelope_curves.csv", index=False)


if __name__ == "__main__":
    main()
