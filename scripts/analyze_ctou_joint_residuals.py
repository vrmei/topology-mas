"""Measure joint node-transition residuals left by CTOU marginal laws."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from analyze_ctou_transition_prediction import (
    COUNT_COLUMNS,
    STATE_INDEX,
    STATES,
    design_matrix,
    table_predictions,
)
from analyze_provenance_recursive_rollout import (
    P_COUNT_COLUMNS,
    P_STATE_INDEX,
    P_STATES,
    collapse_provenance,
    load_provenance_updates,
    provenance_design_matrix,
    provenance_table_predictions,
)
from sklearn.linear_model import LogisticRegression


MODELS = ("ctou_table", "ctou_logit", "provenance_table", "provenance_logit")
OUTCOMES = ("correct", "target")
FEATURES = (
    "immediate_jaccard",
    "causal_jaccard",
    "normal_causal_jaccard",
    "immediate_overlap_any",
    "causal_overlap_any",
    "normal_causal_overlap_any",
)
PAIR_SCOPES = ("all", "internal_internal", "readout_internal")
DEFAULT_SEED = 20_260_816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-updates", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fit_aligned_logistic(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    test_matrix: np.ndarray,
    classes: int,
) -> np.ndarray:
    model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0)
    model.fit(train_matrix, train_labels)
    raw = model.predict_proba(test_matrix)
    aligned = np.zeros((len(test_matrix), classes), dtype=np.float32)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def crossed_marginal_predictions(
    updates: pd.DataFrame,
    *,
    folds: int,
    prior_strength: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    predictions = {
        model: np.full((len(updates), len(STATES)), np.nan, dtype=np.float32)
        for model in MODELS
    }
    ctou_matrix = design_matrix(updates)
    provenance_matrix = provenance_design_matrix(updates)
    ctou_labels = updates.current_state_index.to_numpy(int)
    provenance_labels = updates.current_provenance_index.to_numpy(int)
    audit: list[dict[str, Any]] = []

    for graph_fold in range(folds):
        for task_fold in range(folds):
            test_mask = updates.graph_fold.eq(graph_fold) & updates.task_fold.eq(task_fold)
            train_mask = updates.graph_fold.ne(graph_fold) & updates.task_fold.ne(task_fold)
            test_indices = np.flatnonzero(test_mask.to_numpy())
            train_indices = np.flatnonzero(train_mask.to_numpy())
            if len(test_indices) == 0:
                continue
            print(
                f"marginal fold graph={graph_fold} task={task_fold} "
                f"train={len(train_indices)} test={len(test_indices)}",
                flush=True,
            )
            train = updates.iloc[train_indices]
            test = updates.iloc[test_indices]
            predictions["ctou_table"][test_indices] = table_predictions(
                train, test, prior_strength
            )
            predictions["ctou_logit"][test_indices] = fit_aligned_logistic(
                ctou_matrix[train_indices],
                ctou_labels[train_indices],
                ctou_matrix[test_indices],
                len(STATES),
            )
            provenance_table = provenance_table_predictions(train, test, prior_strength)
            provenance_logit = fit_aligned_logistic(
                provenance_matrix[train_indices],
                provenance_labels[train_indices],
                provenance_matrix[test_indices],
                len(P_STATES),
            )
            predictions["provenance_table"][test_indices] = collapse_provenance(
                provenance_table
            )
            predictions["provenance_logit"][test_indices] = collapse_provenance(
                provenance_logit
            )
            audit.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "train_updates": len(train),
                    "test_updates": len(test),
                    "graph_overlap": len(set(train.graph_id) & set(test.graph_id)),
                    "task_overlap": len(set(train.task_id) & set(test.task_id)),
                }
            )
    for model, probability in predictions.items():
        if np.isnan(probability).any():
            raise RuntimeError(f"incomplete out-of-fold probabilities for {model}")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"invalid probability rows for {model}")
    return predictions, pd.DataFrame(audit)


def cell_ids(frame: pd.DataFrame, provenance: bool) -> np.ndarray:
    columns: Iterable[str]
    if provenance:
        columns = ("previous_provenance_state", "round_index", *P_COUNT_COLUMNS)
    else:
        columns = ("previous_state", "round_index", *COUNT_COLUMNS)
    index = pd.MultiIndex.from_frame(frame[list(columns)])
    return pd.factorize(index, sort=True)[0]


def cross_graph_adjusted_residual(
    frame: pd.DataFrame,
    residual: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    work = frame[["task_id", "round_index", "graph_id"]].copy()
    work["cell_id"] = cells
    work["residual"] = residual
    group = ["task_id", "round_index", "cell_id"]
    graph_group = [*group, "graph_id"]
    total_sum = work.groupby(group).residual.transform("sum").to_numpy(float)
    total_count = work.groupby(group).residual.transform("size").to_numpy(int)
    graph_sum = work.groupby(graph_group).residual.transform("sum").to_numpy(float)
    graph_count = work.groupby(graph_group).residual.transform("size").to_numpy(int)
    support = total_count - graph_count
    baseline = np.divide(
        total_sum - graph_sum,
        support,
        out=np.full(len(work), np.nan, dtype=float),
        where=support > 0,
    )
    return residual - baseline, support


def graph_incoming(graph: dict[str, Any]) -> list[set[int]]:
    incoming = [set() for _ in range(int(graph["node_count"]))]
    for edge in graph["edges"]:
        incoming[int(edge["target"])].add(int(edge["source"]))
    return incoming


def causal_cone(incoming: list[set[int]], node: int, round_index: int) -> set[int]:
    reached = {node}
    frontier = {node}
    for _ in range(round_index):
        frontier = set().union(*(incoming[current] for current in frontier)) if frontier else set()
        frontier -= reached
        reached |= frontier
    return reached


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def pair_topology_features(
    graph: dict[str, Any],
    attack_node: int,
    round_index: int,
    left: int,
    right: int,
) -> dict[str, Any]:
    incoming = graph_incoming(graph)
    left_immediate = incoming[left]
    right_immediate = incoming[right]
    left_cone = causal_cone(incoming, left, round_index)
    right_cone = causal_cone(incoming, right, round_index)
    left_normal = left_cone - {attack_node}
    right_normal = right_cone - {attack_node}
    return {
        "immediate_shared_count": len(left_immediate & right_immediate),
        "immediate_jaccard": jaccard(left_immediate, right_immediate),
        "causal_shared_count": len(left_cone & right_cone),
        "causal_jaccard": jaccard(left_cone, right_cone),
        "normal_causal_shared_count": len(left_normal & right_normal),
        "normal_causal_jaccard": jaccard(left_normal, right_normal),
        "attacker_in_both_cones": int(
            attack_node in left_cone and attack_node in right_cone
        ),
        "direct_attacker_to_both": int(
            attack_node in left_immediate and attack_node in right_immediate
        ),
        "pair_connected": int(
            left in right_immediate or right in left_immediate
        ),
    }


def build_pairs(
    updates: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    ctou_cells: np.ndarray,
    provenance_cells: np.ndarray,
) -> pd.DataFrame:
    frame = updates.reset_index(drop=True).copy()
    frame["update_id"] = np.arange(len(frame), dtype=np.int64)
    frame["ctou_cell_id"] = ctou_cells
    frame["provenance_cell_id"] = provenance_cells
    keys = [
        "stratum",
        "task_id",
        "graph_id",
        "run_spec_id",
        "attack_node",
        "round_index",
    ]
    rows: list[dict[str, Any]] = []
    feature_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event_index, (event, group) in enumerate(frame.groupby(keys, sort=False)):
        stratum, task_id, graph_id, run_spec_id, attack_node, round_index = event
        records = list(group.itertuples(index=False))
        for left, right in itertools.combinations(records, 2):
            nodes = tuple(sorted((int(left.receiver_node), int(right.receiver_node))))
            cache_key = (graph_id, int(attack_node), int(round_index), *nodes)
            features = feature_cache.get(cache_key)
            if features is None:
                features = pair_topology_features(
                    graphs[str(graph_id)],
                    int(attack_node),
                    int(round_index),
                    nodes[0],
                    nodes[1],
                )
                feature_cache[cache_key] = features
            scopes = {str(left.receiver_scope), str(right.receiver_scope)}
            pair_scope = (
                "internal_internal"
                if scopes == {"internal"}
                else "readout_internal"
                if scopes == {"internal", "readout"}
                else "readout_readout"
            )
            rows.append(
                {
                    "event_id": event_index,
                    "stratum": stratum,
                    "task_id": task_id,
                    "graph_id": graph_id,
                    "run_spec_id": run_spec_id,
                    "attack_node": int(attack_node),
                    "round_index": int(round_index),
                    "n": int(left.n),
                    "m": int(left.m),
                    "left_id": int(left.update_id),
                    "right_id": int(right.update_id),
                    "left_receiver": int(left.receiver_node),
                    "right_receiver": int(right.receiver_node),
                    "pair_scope": pair_scope,
                    "same_ctou_cell": int(left.ctou_cell_id == right.ctou_cell_id),
                    "same_provenance_cell": int(
                        left.provenance_cell_id == right.provenance_cell_id
                    ),
                    **features,
                }
            )
        if event_index and event_index % 20_000 == 0:
            print(f"paired events={event_index} pairs={len(rows)}", flush=True)
    return pd.DataFrame(rows)


def cluster_ratio_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    sufficient = pd.DataFrame(
        {"cluster": clusters, "numerator": numerator, "denominator": denominator}
    ).groupby("cluster", as_index=False).sum()
    numerators = sufficient.numerator.to_numpy(float)
    denominators = sufficient.denominator.to_numpy(float)
    indices = rng.integers(
        0, len(sufficient), size=(replicates, len(sufficient)), endpoint=False
    )
    draw_num = numerators[indices].sum(axis=1)
    draw_den = denominators[indices].sum(axis=1)
    draws = np.divide(
        draw_num,
        draw_den,
        out=np.full(replicates, np.nan),
        where=draw_den > 0,
    )
    return tuple(float(value) for value in np.nanquantile(draws, [0.025, 0.975]))


def mean_product_summary(
    pairs: pd.DataFrame,
    product: np.ndarray,
    valid: np.ndarray,
    *,
    model: str,
    outcome: str,
    residual_variant: str,
    metric: str,
    subset: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_scope in PAIR_SCOPES:
        mask = valid.copy()
        if pair_scope != "all":
            mask &= pairs.pair_scope.eq(pair_scope).to_numpy()
        values = product[mask]
        if len(values) == 0:
            continue
        task_ci = cluster_ratio_bootstrap(
            values,
            np.ones(len(values)),
            pairs.task_id.to_numpy()[mask],
            replicates=replicates,
            rng=np.random.default_rng(seed),
        )
        graph_ci = cluster_ratio_bootstrap(
            values,
            np.ones(len(values)),
            pairs.graph_id.to_numpy()[mask],
            replicates=replicates,
            rng=np.random.default_rng(seed + 1),
        )
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "residual_variant": residual_variant,
                "metric": metric,
                "subset": subset,
                "pair_scope": pair_scope,
                "pairs": len(values),
                "tasks": int(pairs.loc[mask, "task_id"].nunique()),
                "graphs": int(pairs.loc[mask, "graph_id"].nunique()),
                "estimate": float(values.mean()),
                "task_ci95_low": task_ci[0],
                "task_ci95_high": task_ci[1],
                "graph_ci95_low": graph_ci[0],
                "graph_ci95_high": graph_ci[1],
            }
        )
    return rows


def fixed_effect_slope(
    pairs: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    selected = pairs.loc[valid, ["event_id", "task_id", "graph_id"]].copy()
    selected["x"] = x[valid]
    selected["y"] = y[valid]
    if selected.empty:
        return None
    selected["dx"] = selected.x - selected.groupby("event_id").x.transform("mean")
    selected["dy"] = selected.y - selected.groupby("event_id").y.transform("mean")
    selected["numerator"] = selected.dx * selected.dy
    selected["denominator"] = selected.dx**2
    denominator = float(selected.denominator.sum())
    if denominator <= 0:
        return None
    task_ci = cluster_ratio_bootstrap(
        selected.numerator.to_numpy(float),
        selected.denominator.to_numpy(float),
        selected.task_id.to_numpy(),
        replicates=replicates,
        rng=np.random.default_rng(seed),
    )
    graph_ci = cluster_ratio_bootstrap(
        selected.numerator.to_numpy(float),
        selected.denominator.to_numpy(float),
        selected.graph_id.to_numpy(),
        replicates=replicates,
        rng=np.random.default_rng(seed + 1),
    )
    varying_events = selected.groupby("event_id").denominator.sum().gt(0).sum()
    return {
        "pairs": len(selected),
        "events": int(selected.event_id.nunique()),
        "varying_events": int(varying_events),
        "tasks": int(selected.task_id.nunique()),
        "graphs": int(selected.graph_id.nunique()),
        "slope": float(selected.numerator.sum() / denominator),
        "task_ci95_low": task_ci[0],
        "task_ci95_high": task_ci[1],
        "graph_ci95_low": graph_ci[0],
        "graph_ci95_high": graph_ci[1],
    }


def overlap_group_rows(
    pairs: pd.DataFrame,
    product: np.ndarray,
    valid: np.ndarray,
    *,
    model: str,
    outcome: str,
    residual_variant: str,
    feature: str,
    subset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = pairs[feature].to_numpy(float)
    groups = {
        "zero": values == 0,
        "positive": values > 0,
    }
    for name, group_mask in groups.items():
        mask = valid & group_mask
        if not mask.any():
            continue
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "residual_variant": residual_variant,
                "feature": feature,
                "subset": subset,
                "overlap_group": name,
                "pairs": int(mask.sum()),
                "events": int(pairs.loc[mask, "event_id"].nunique()),
                "mean_product": float(product[mask].mean()),
            }
        )
    return rows


def analyze_pairs(
    updates: pd.DataFrame,
    pairs: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    ctou_cells: np.ndarray,
    provenance_cells: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    slopes: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    residual_products: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    left = pairs.left_id.to_numpy(int)
    right = pairs.right_id.to_numpy(int)

    for model_index, model in enumerate(MODELS):
        provenance = model.startswith("provenance")
        cells = provenance_cells if provenance else ctou_cells
        same_cell = pairs[
            "same_provenance_cell" if provenance else "same_ctou_cell"
        ].eq(1).to_numpy()
        for outcome_index, outcome in enumerate(OUTCOMES):
            state_index = STATE_INDEX[outcome]
            label = updates.next_state.eq(outcome).to_numpy(float)
            probability = predictions[model][:, state_index].astype(float)
            raw = label - probability
            adjusted, support = cross_graph_adjusted_residual(updates, raw, cells)
            residuals = {"raw": raw, "task_cell_adjusted": adjusted}
            for variant_index, (variant, residual) in enumerate(residuals.items()):
                product = residual[left] * residual[right]
                valid = np.isfinite(product)
                residual_products[(model, outcome, variant)] = (product, valid)
                subsets = {"all": np.ones(len(pairs), dtype=bool), "same_cell": same_cell}
                for subset_index, (subset, subset_mask) in enumerate(subsets.items()):
                    eligible_base = valid & subset_mask
                    summaries.extend(
                        mean_product_summary(
                            pairs,
                            product,
                            eligible_base,
                            model=model,
                            outcome=outcome,
                            residual_variant=variant,
                            metric="residual_product",
                            subset=subset,
                            replicates=replicates,
                            seed=seed
                            + 10_000 * model_index
                            + 1_000 * outcome_index
                            + 100 * variant_index
                            + subset_index,
                        )
                    )
                    analysis_groups = [(scope, "all") for scope in PAIR_SCOPES]
                    if subset == "all":
                        analysis_groups.extend((("all", "5"), ("all", "8")))
                    for group_index, (pair_scope, n_group) in enumerate(analysis_groups):
                        scope_mask = (
                            np.ones(len(pairs), dtype=bool)
                            if pair_scope == "all"
                            else pairs.pair_scope.eq(pair_scope).to_numpy()
                        )
                        n_mask = (
                            np.ones(len(pairs), dtype=bool)
                            if n_group == "all"
                            else pairs.n.eq(int(n_group)).to_numpy()
                        )
                        eligible = eligible_base & scope_mask & n_mask
                        for feature_index, feature in enumerate(FEATURES):
                            result = fixed_effect_slope(
                                pairs,
                                pairs[feature].to_numpy(float),
                                product,
                                eligible,
                                replicates=replicates,
                                seed=seed
                                + 1_000_000 * model_index
                                + 100_000 * outcome_index
                                + 10_000 * variant_index
                                + 1_000 * subset_index
                                + 10 * group_index
                                + feature_index,
                            )
                            if result is not None:
                                slopes.append(
                                    {
                                        "model": model,
                                        "outcome": outcome,
                                        "residual_variant": variant,
                                        "subset": subset,
                                        "pair_scope": pair_scope,
                                        "n_group": n_group,
                                        "feature": feature,
                                        **result,
                                    }
                                )
                    for feature in FEATURES:
                        overlap_rows.extend(
                            overlap_group_rows(
                                pairs,
                                product,
                                eligible_base,
                                model=model,
                                outcome=outcome,
                                residual_variant=variant,
                                feature=feature,
                                subset=subset,
                            )
                        )

            clipped = np.clip(probability, 0.01, 0.99)
            standardized = raw / np.sqrt(clipped * (1 - clipped))
            standardized_product = standardized[left] * standardized[right]
            summaries.extend(
                mean_product_summary(
                    pairs,
                    standardized_product,
                    np.isfinite(standardized_product),
                    model=model,
                    outcome=outcome,
                    residual_variant="raw",
                    metric="pearson_residual_product",
                    subset="all",
                    replicates=replicates,
                    seed=seed + 1_000_000 + model_index * 100 + outcome_index,
                )
            )

    comparisons: list[dict[str, Any]] = []
    for provenance, ctou in (
        ("provenance_table", "ctou_table"),
        ("provenance_logit", "ctou_logit"),
    ):
        for outcome in OUTCOMES:
            for variant in ("raw", "task_cell_adjusted"):
                candidate, candidate_valid = residual_products[(provenance, outcome, variant)]
                reference, reference_valid = residual_products[(ctou, outcome, variant)]
                valid = candidate_valid & reference_valid
                difference = candidate[valid] - reference[valid]
                task_ci = cluster_ratio_bootstrap(
                    difference,
                    np.ones(len(difference)),
                    pairs.task_id.to_numpy()[valid],
                    replicates=replicates,
                    rng=np.random.default_rng(seed + 2_000_000),
                )
                graph_ci = cluster_ratio_bootstrap(
                    difference,
                    np.ones(len(difference)),
                    pairs.graph_id.to_numpy()[valid],
                    replicates=replicates,
                    rng=np.random.default_rng(seed + 2_000_001),
                )
                comparisons.append(
                    {
                        "candidate": provenance,
                        "reference": ctou,
                        "outcome": outcome,
                        "residual_variant": variant,
                        "pairs": len(difference),
                        "mean_product_difference": float(difference.mean()),
                        "task_ci95_low": task_ci[0],
                        "task_ci95_high": task_ci[1],
                        "graph_ci95_low": graph_ci[0],
                        "graph_ci95_high": graph_ci[1],
                        "negative_favors_provenance": True,
                    }
                )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(slopes),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(comparisons),
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates, update_audit = load_provenance_updates(args.provenance_updates, args.folds)
    graphs = json.loads(args.graphs.read_text(encoding="utf-8"))
    missing_graphs = sorted(set(updates.graph_id) - set(graphs))
    if missing_graphs:
        raise ValueError(f"missing graph definitions: {missing_graphs[:5]}")
    ctou_cells = cell_ids(updates, provenance=False)
    provenance_cells = cell_ids(updates, provenance=True)
    probability_path = args.output_dir / "oof_marginal_probabilities.npz"
    fold_path = args.output_dir / "fold_audit.csv"
    if probability_path.exists() and fold_path.exists() and not args.force:
        saved = np.load(probability_path)
        predictions = {model: saved[model] for model in MODELS}
        fold_audit = pd.read_csv(fold_path)
        print("loaded cached OOF marginal probabilities", flush=True)
    else:
        predictions, fold_audit = crossed_marginal_predictions(
            updates,
            folds=args.folds,
            prior_strength=args.table_prior_strength,
        )
        np.savez_compressed(probability_path, **predictions)
        fold_audit.to_csv(fold_path, index=False)
    if (fold_audit.graph_overlap != 0).any() or (fold_audit.task_overlap != 0).any():
        raise RuntimeError("crossed holdout leakage detected")
    pair_path = args.output_dir / "pair_index_and_features.pkl.gz"
    if pair_path.exists() and not args.force:
        pairs = pd.read_pickle(pair_path, compression="gzip")
        print(f"loaded cached receiver pairs={len(pairs)}", flush=True)
    else:
        pairs = build_pairs(updates, graphs, ctou_cells, provenance_cells)
        pairs.to_pickle(pair_path, compression="gzip")
    pairs["immediate_overlap_any"] = pairs.immediate_shared_count.gt(0).astype(int)
    pairs["causal_overlap_any"] = pairs.causal_shared_count.gt(0).astype(int)
    pairs["normal_causal_overlap_any"] = pairs.normal_causal_shared_count.gt(0).astype(int)
    summaries, slopes, overlap, comparisons = analyze_pairs(
        updates,
        pairs,
        predictions,
        ctou_cells,
        provenance_cells,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    fold_audit.to_csv(fold_path, index=False)
    summaries.to_csv(args.output_dir / "comovement_summary.csv", index=False)
    slopes.to_csv(args.output_dir / "within_event_topology_slopes.csv", index=False)
    overlap.to_csv(args.output_dir / "overlap_group_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "provenance_pair_comparisons.csv", index=False)
    pair_support = (
        pairs.groupby(["n", "m", "pair_scope"], as_index=False)
        .agg(
            pairs=("event_id", "size"),
            events=("event_id", "nunique"),
            tasks=("task_id", "nunique"),
            graphs=("graph_id", "nunique"),
            same_ctou_cell=("same_ctou_cell", "sum"),
            same_provenance_cell=("same_provenance_cell", "sum"),
            mean_immediate_jaccard=("immediate_jaccard", "mean"),
            mean_causal_jaccard=("causal_jaccard", "mean"),
        )
    )
    pair_support.to_csv(args.output_dir / "pair_support.csv", index=False)
    manifest = {
        "analysis_version": "ctou-joint-residual-v1",
        "models": list(MODELS),
        "outcomes": list(OUTCOMES),
        "features": list(FEATURES),
        "folds": args.folds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "updates": update_audit,
        "pairs": len(pairs),
        "events": int(pairs.event_id.nunique()),
        "tasks": int(pairs.task_id.nunique()),
        "graphs": int(pairs.graph_id.nunique()),
        "information_boundary": (
            "marginal probabilities use crossed graph/task holdout; topology features "
            "are used only in post-hoc residual analysis"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
