"""Compare CTOU mean-field and discrete particle rollouts on a frozen grid."""

from __future__ import annotations

import argparse
import json
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
from topology_mas.simulation.rollout import (
    expected_composition_rollout,
    particle_composition_rollout,
    sample_round_zero_states,
)

VARIANT = "proportions_saturating_volume_k2"
SIZES = (5, 10, 15, 20, 30, 40, 50)
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
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--task-chunk-size", type=int, default=5)
    parser.add_argument("--swap-steps", type=int, default=200)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--densities", type=float, nargs="+", default=list(DENSITIES))
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


def fit_law(frame: pd.DataFrame, sizes: set[int], balance: bool) -> LogisticRegression:
    selected = frame.loc[frame.n.isin(sizes)].reset_index(drop=True)
    sample_weight = None
    if balance:
        group_count = selected.groupby(["n", "task_id"]).size()
        keys = pd.MultiIndex.from_frame(selected[["n", "task_id"]])
        sample_weight = 1.0 / group_count.reindex(keys).to_numpy(float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=0)
    model.fit(
        ctou_design_matrix(selected, VARIANT),
        selected.current_state_index.to_numpy(int),
        sample_weight=sample_weight,
    )
    return model


def predictor(model: LogisticRegression):
    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = model.predict_proba(ctou_design_matrix(frame, VARIANT))
        result = np.zeros((len(frame), 4), dtype=float)
        result[:, model.classes_.astype(int)] = raw
        return result

    return predict


def edge_count(n: int, density: float) -> int:
    minimum = n - 1
    maximum = (n - 1) ** 2
    return int(round(minimum + density * (maximum - minimum)))


def attacker_positions(n: int) -> np.ndarray:
    count = n - 1 if n <= 10 else 5
    return np.unique(np.linspace(0, n - 2, count, dtype=int))


def initial_marginals(initializer: Any, task_ids: list[str], n: int) -> np.ndarray:
    result = np.zeros((len(task_ids), n, 4), dtype=float)
    for index, task_id in enumerate(task_ids):
        correct, other, unparsed = initializer.mean_for_task(task_id)
        result[index, :, 0] = correct
        result[index, :, 2] = other
        result[index, :, 3] = unparsed
    return result


def mean_field_cell(
    *,
    graph: Any,
    task_ids: list[str],
    initializer: Any,
    clean_predictor: Any,
    attack_predictor: Any,
    positions: np.ndarray,
) -> pd.DataFrame:
    initial = initial_marginals(initializer, task_ids, graph.node_count)
    clean = expected_composition_rollout(
        graph=graph,
        initial_marginals=initial,
        attack_nodes=np.full(len(task_ids), -1, dtype=int),
        predictor=clean_predictor,
    )
    attack = np.repeat(initial[:, None], len(positions), axis=1).reshape(
        len(task_ids) * len(positions), graph.node_count, 4
    )
    endpoint = expected_composition_rollout(
        graph=graph,
        initial_marginals=attack,
        attack_nodes=np.tile(positions, len(task_ids)),
        predictor=attack_predictor,
    ).reshape(len(task_ids), len(positions), 4)
    return pd.DataFrame(
        {
            "task_id": task_ids,
            "meanfield_utility": clean[:, 0],
            "meanfield_robustness": endpoint[:, :, 0].mean(axis=1),
            "meanfield_target_risk": endpoint[:, :, 1].mean(axis=1),
        }
    )


def particle_cell(
    *,
    graph: Any,
    task_ids: list[str],
    initializer: Any,
    clean_predictor: Any,
    attack_predictor: Any,
    positions: np.ndarray,
    particles: int,
    task_chunk_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chunk_start in range(0, len(task_ids), task_chunk_size):
        chunk = task_ids[chunk_start : chunk_start + task_chunk_size]
        rng = np.random.default_rng(seed + chunk_start)
        initial = sample_round_zero_states(
            initializer=initializer,
            task_ids=chunk,
            node_count=graph.node_count,
            particles=particles,
            rng=rng,
        )
        clean = particle_composition_rollout(
            graph=graph,
            initial_states=initial,
            attack_nodes=np.full(len(initial), -1, dtype=int),
            predictor=clean_predictor,
            rng=rng,
        ).reshape(len(chunk), particles)
        correct = np.empty((len(chunk), len(positions)), dtype=float)
        target = np.empty_like(correct)
        for position_index, position in enumerate(positions):
            endpoint = particle_composition_rollout(
                graph=graph,
                initial_states=initial,
                attack_nodes=np.full(len(initial), position, dtype=int),
                predictor=attack_predictor,
                rng=rng,
            ).reshape(len(chunk), particles)
            correct[:, position_index] = np.mean(endpoint == 0, axis=1)
            target[:, position_index] = np.mean(endpoint == 1, axis=1)
        for task_index, task_id in enumerate(chunk):
            rows.append(
                {
                    "task_id": task_id,
                    "particle_utility": float(np.mean(clean[task_index] == 0)),
                    "particle_robustness": float(correct[task_index].mean()),
                    "particle_target_risk": float(target[task_index].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.particles < 1 or args.task_chunk_size < 1:
        raise ValueError("particles and task chunk size must be positive")
    if not set(args.sizes) <= set(SIZES):
        raise ValueError(f"sizes must be a subset of {SIZES}")
    if not set(args.densities) <= set(DENSITIES):
        raise ValueError(f"densities must be a subset of {DENSITIES}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates, round0 = load_sources(
        (args.n5_n8_cache, args.n6_n7_cache, args.n10_cache)
    )
    fitted: dict[str, dict[str, Any]] = {}
    for version, sizes, balance in (
        ("strict_n5", {5}, False),
        ("calibrated_n5_n6_n7_n8_n10", set(REAL_SIZES), True),
    ):
        fitted[version] = {
            "clean": predictor(fit_law(updates["clean"], sizes, balance)),
            "attack": predictor(fit_law(updates["attack"], sizes, balance)),
            "initializer": fit_hierarchical_round_zero(
                round0.loc[round0.n.isin(sizes)], required_sizes=sizes
            ),
        }
    task_ids = sorted(fitted["strict_n5"]["initializer"].task_means)
    predictions: list[pd.DataFrame] = []
    for version_index, (version, model) in enumerate(fitted.items()):
        for n in args.sizes:
            for density in args.densities:
                m = edge_count(n, density)
                graph, audit = sample_backbone_augmented_graph(
                    node_count=n,
                    edge_count=m,
                    horizon=3,
                    seed=SEED,
                    sample_index=0,
                    swap_steps=args.swap_steps,
                )
                positions = attacker_positions(n)
                meanfield = mean_field_cell(
                    graph=graph,
                    task_ids=task_ids,
                    initializer=model["initializer"],
                    clean_predictor=model["clean"],
                    attack_predictor=model["attack"],
                    positions=positions,
                )
                particles = particle_cell(
                    graph=graph,
                    task_ids=task_ids,
                    initializer=model["initializer"],
                    clean_predictor=model["clean"],
                    attack_predictor=model["attack"],
                    positions=positions,
                    particles=args.particles,
                    task_chunk_size=args.task_chunk_size,
                    seed=SEED + version_index * 100_000 + n * 100 + int(density * 10),
                )
                merged = meanfield.merge(particles, on="task_id", validate="one_to_one")
                merged = merged.assign(
                    version=version,
                    n=n,
                    m=m,
                    density=density,
                    graph_id=graph.graph_id,
                    attacker_positions=len(positions),
                    swap_acceptance_rate=audit.acceptance_rate,
                )
                predictions.append(merged)
                print(f"stage=particle version={version} n={n} density={density}", flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(args.output_dir / "particle_predictions.csv.gz", index=False)
    rows: list[dict[str, object]] = []
    for keys, frame in prediction.groupby(["version", "n", "density"], sort=True):
        version, n, density = keys
        row: dict[str, object] = {"version": version, "n": n, "density": density}
        for metric in ("utility", "robustness", "target_risk"):
            difference = frame[f"meanfield_{metric}"] - frame[f"particle_{metric}"]
            row[f"{metric}_task_mae"] = float(np.mean(np.abs(difference)))
            row[f"{metric}_aggregate_error"] = float(abs(difference.mean()))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "particle_metrics.csv", index=False)
    gate = {
        "particles": args.particles,
        "utility_task_mae": float(metrics.utility_task_mae.mean()),
        "robustness_task_mae": float(metrics.robustness_task_mae.mean()),
        "max_utility_aggregate_error": float(metrics.utility_aggregate_error.max()),
        "max_robustness_aggregate_error": float(
            metrics.robustness_aggregate_error.max()
        ),
    }
    gate["passed"] = bool(
        gate["utility_task_mae"] <= 0.03
        and gate["robustness_task_mae"] <= 0.03
        and gate["max_utility_aggregate_error"] <= 0.05
        and gate["max_robustness_aggregate_error"] <= 0.05
    )
    (args.output_dir / "particle_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
