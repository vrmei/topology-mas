"""Run strict and calibrated CTOU model-based scale simulations."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
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
from topology_mas.simulation.graph_sampling import (
    normalized_density_edge_levels,
    sample_backbone_augmented_graph,
)
from topology_mas.simulation.rollout import expected_composition_rollout

VARIANT = "proportions_saturating_volume_k2"
REAL_SIZES = (5, 6, 7, 8, 10)
DELTAS = tuple([index / 20 for index in range(19)] + [1.0])
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
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--graphs-per-level", type=int, default=10)
    parser.add_argument("--swap-steps", type=int, default=200)
    parser.add_argument(
        "--attacker-samples",
        type=int,
        default=0,
        help="0 traverses all non-readout attacker positions.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_training_sources(paths: tuple[Path, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    update_parts: dict[str, list[pd.DataFrame]] = {"attack": [], "clean": []}
    round0_parts: list[pd.DataFrame] = []
    for path in paths:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        for condition in ("attack", "clean"):
            update_parts[condition].append(
                bundle[f"{condition}_updates"][list(UPDATE_COLUMNS)].copy()
            )
        round0_parts.append(extract_round_zero_groups(bundle["clean_cases"]))
        del bundle
    return (
        {
            condition: pd.concat(frames, ignore_index=True)
            for condition, frames in update_parts.items()
        },
        pd.concat(round0_parts, ignore_index=True).drop_duplicates(["task_id", "graph_id", "n"]),
    )


def fit_law(frame: pd.DataFrame, sizes: set[int], *, balance_sizes: bool) -> LogisticRegression:
    selected = frame.loc[frame.n.isin(sizes)].reset_index(drop=True)
    sample_weight = None
    if balance_sizes:
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


def task_initial_marginals(initializer: Any, task_ids: list[str], n: int) -> np.ndarray:
    result = np.zeros((len(task_ids), n, 4), dtype=float)
    for task_index, task_id in enumerate(task_ids):
        correct, other, unparsed = initializer.mean_for_task(task_id)
        result[task_index, :, 0] = correct
        result[task_index, :, 2] = other
        result[task_index, :, 3] = unparsed
    return result


def attacker_positions(n: int, samples: int, *, seed: int) -> np.ndarray:
    positions = np.arange(n - 1, dtype=int)
    if samples <= 0 or samples >= len(positions):
        return positions
    return np.sort(np.random.default_rng(seed).choice(positions, size=samples, replace=False))


def simulate_graph(
    *,
    graph: Any,
    task_ids: list[str],
    initializer: Any,
    clean_predictor: Any,
    attack_predictor: Any,
    attacker_sample_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n = graph.node_count
    clean_initial = task_initial_marginals(initializer, task_ids, n)
    clean_endpoint = expected_composition_rollout(
        graph=graph,
        initial_marginals=clean_initial,
        attack_nodes=np.full(len(task_ids), -1, dtype=int),
        predictor=clean_predictor,
    )
    positions = attacker_positions(n, attacker_sample_count, seed=seed)
    attack_initial = np.repeat(clean_initial[:, None, :, :], len(positions), axis=1).reshape(
        len(task_ids) * len(positions), n, 4
    )
    attacks = np.tile(positions, len(task_ids))
    attack_endpoint = expected_composition_rollout(
        graph=graph,
        initial_marginals=attack_initial,
        attack_nodes=attacks,
        predictor=attack_predictor,
    ).reshape(len(task_ids), len(positions), 4)
    task = pd.DataFrame(
        {
            "task_id": task_ids,
            "utility": clean_endpoint[:, 0],
            "robustness": attack_endpoint[:, :, 0].mean(axis=1),
            "target_risk": attack_endpoint[:, :, 1].mean(axis=1),
            "u0": clean_initial[:, graph.readout_node, 0],
        }
    )
    task["attack_penalty"] = task.utility - task.robustness
    task["delta_utility"] = task.utility - task.u0
    summary = {
        column: float(task[column].mean())
        for column in (
            "utility",
            "robustness",
            "target_risk",
            "u0",
            "attack_penalty",
            "delta_utility",
        )
    }
    summary["attacker_positions"] = len(positions)
    return task, summary


def main() -> None:
    args = parse_args()
    if not 5 <= args.n_min <= args.n_max <= 50:
        raise ValueError("simulation range must lie within n=5..50")
    if args.graphs_per_level < 1 or args.swap_steps < 0:
        raise ValueError("graph count must be positive and swap steps nonnegative")
    paths = (args.n5_n8_cache, args.n6_n7_cache, args.n10_cache)
    print("stage=load_and_fit", flush=True)
    updates, round0 = load_training_sources(paths)
    versions: dict[str, dict[str, Any]] = {}
    for version, sizes, balance in (
        ("strict_n5", {5}, False),
        ("calibrated_n5_n6_n7_n8_n10", set(REAL_SIZES), True),
    ):
        versions[version] = {
            "clean": fit_law(updates["clean"], sizes, balance_sizes=balance),
            "attack": fit_law(updates["attack"], sizes, balance_sizes=balance),
            "initializer": fit_hierarchical_round_zero(
                round0.loc[round0.n.isin(sizes)], required_sizes=sizes
            ),
            "sizes": sorted(sizes),
        }
    task_ids = sorted(versions["strict_n5"]["initializer"].task_means)
    checkpoint_root = args.output_dir / "simulated_curves" / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for n in range(args.n_min, args.n_max + 1):
        checkpoint = checkpoint_root / f"n{n}.csv.gz"
        task_checkpoint = checkpoint_root / f"n{n}_tasks.csv.gz"
        if checkpoint.is_file() and task_checkpoint.is_file():
            print(f"stage=skip_n n={n}", flush=True)
            continue
        print(f"stage=simulate_n n={n}", flush=True)
        graph_rows: list[dict[str, object]] = []
        task_rows: list[pd.DataFrame] = []
        for edge_count, requested_delta in normalized_density_edge_levels(n, DELTAS):
            maximum = (n - 1) ** 2
            graph_count = 1 if edge_count == maximum else args.graphs_per_level
            for graph_index in range(graph_count):
                graph, mixing = sample_backbone_augmented_graph(
                    node_count=n,
                    edge_count=edge_count,
                    horizon=3,
                    seed=args.seed,
                    sample_index=graph_index,
                    swap_steps=args.swap_steps,
                )
                for version, fitted in versions.items():
                    task, summary = simulate_graph(
                        graph=graph,
                        task_ids=task_ids,
                        initializer=fitted["initializer"],
                        clean_predictor=predictor(fitted["clean"]),
                        attack_predictor=predictor(fitted["attack"]),
                        attacker_sample_count=args.attacker_samples,
                        seed=args.seed + graph_index + edge_count,
                    )
                    common = {
                        "version": version,
                        "model": VARIANT,
                        "n": n,
                        "m": edge_count,
                        "delta_requested": requested_delta,
                        "delta_realized": (edge_count - (n - 1)) / ((n - 1) ** 2 - (n - 1))
                        if n > 2
                        else 0.0,
                        "average_degree": edge_count / (n - 1),
                        "graph_id": graph.graph_id,
                        "graph_index": graph_index,
                        "swap_acceptance_rate": mixing.acceptance_rate,
                    }
                    graph_rows.append({**common, **summary})
                    task_rows.append(task.assign(**common))
        pd.DataFrame(graph_rows).to_csv(checkpoint, index=False)
        pd.concat(task_rows, ignore_index=True).to_csv(task_checkpoint, index=False)
    graphs = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(checkpoint_root.glob("n[0-9]*.csv.gz"))
            if "_tasks" not in path.name
        ],
        ignore_index=True,
    )
    curves = graphs.groupby(
        ["version", "model", "n", "m", "delta_realized", "average_degree"],
        as_index=False,
    ).agg(
        graphs=("graph_id", "nunique"),
        utility=("utility", "mean"),
        utility_sd=("utility", "std"),
        robustness=("robustness", "mean"),
        robustness_sd=("robustness", "std"),
        target_risk=("target_risk", "mean"),
        attack_penalty=("attack_penalty", "mean"),
        delta_utility=("delta_utility", "mean"),
        u0=("u0", "mean"),
    )
    curves.to_csv(args.output_dir / "simulated_curves" / "primary_curves.csv", index=False)
    manifest = {
        "analysis_version": "ctou-scale-simulation-n5-to-n50-v1-phase2-primary",
        "simulation_only_beyond_n": 10,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "graphs_per_level": args.graphs_per_level,
        "swap_steps": args.swap_steps,
        "attacker_samples": args.attacker_samples,
        "density_levels": list(DELTAS),
        "local_law": VARIANT,
        "versions": {key: value["sizes"] for key, value in versions.items()},
        "mean_field": "expected incoming CTOU composition plug-in",
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "model-based simulation; n>10 is not real LLM measurement",
    }
    atomic_json(args.output_dir / "phase2_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
