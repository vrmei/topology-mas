"""Measure out-of-graph structural predictability of node attack vulnerability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.switch_backend("Agg")

ANALYSIS_VERSION = "classical-structure-explainability-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

LOCAL_FEATURES = (
    "distance_to_readout",
    "direct_to_readout",
    "in_degree",
    "out_degree",
    "is_source",
    "descendant_count",
    "ancestor_count",
)
ROUTING_FEATURES = (
    "distance_to_readout",
    "direct_to_readout",
    "simple_path_count",
    "shortest_path_count",
    "walk_count_to_readout",
    "edge_connectivity_to_readout",
    "node_connectivity_to_readout",
    "postdominated_upstream_count",
)
FULL_FEATURES = tuple(
    dict.fromkeys(
        LOCAL_FEATURES
        + ROUTING_FEATURES
        + (
            "betweenness",
            "outward_closeness",
            "scc_size",
            "in_nontrivial_cycle",
            "readout_in_degree",
            "source_count",
            "max_scc_size",
            "graph_has_cycle",
            "mean_distance_to_readout",
            "std_distance_to_readout",
            "message_opportunities",
        )
    )
)
FEATURE_SETS = {
    "ridge_distance": ("distance_to_readout", "direct_to_readout"),
    "ridge_local": LOCAL_FEATURES,
    "ridge_routing": ROUTING_FEATURES,
    "ridge_full": FULL_FEATURES,
}
OUTCOMES = ("paired_accuracy_drop", "induced_readout_target_rate")


@dataclass(frozen=True)
class GraphRecord:
    stratum: str
    graph: dict[str, Any]
    node_outcomes: dict[int, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(run_root: Path, status: dict[str, Any]) -> list[GraphRecord]:
    records: list[GraphRecord] = []
    for descriptor in status["strata"]:
        stratum = descriptor["key"]
        root = run_root / "strata" / stratum
        graphs = read_jsonl(root / "selected_graphs.jsonl")
        metrics = read_json(root / "analysis-v1" / "graph_metrics.json")
        metric_by_graph = {item["graph_id"]: item for item in metrics}
        for graph in graphs:
            node_outcomes = {
                int(item["node_id"]): item
                for item in metric_by_graph[graph["graph_id"]]["node_metrics"]
            }
            records.append(GraphRecord(stratum, graph, node_outcomes))
    return records


def adjacency_graph(graph: dict[str, Any]) -> nx.DiGraph:
    value = nx.DiGraph()
    value.add_nodes_from(range(int(graph["node_count"])))
    value.add_edges_from((int(edge["source"]), int(edge["target"])) for edge in graph["edges"])
    return value


def walk_count_to_readout(graph: nx.DiGraph, node: int, readout: int, horizon: int) -> int:
    nodes = list(range(graph.number_of_nodes()))
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.int64)
    power = np.eye(len(nodes), dtype=np.int64)
    total = 0
    for _ in range(horizon):
        power = power @ adjacency
        total += int(power[node, readout])
    return total


def postdominated_upstream_count(graph: nx.DiGraph, node: int, readout: int) -> int:
    reduced = graph.copy()
    reduced.remove_node(node)
    count = 0
    for upstream in graph.nodes:
        if upstream in {node, readout}:
            continue
        if not nx.has_path(reduced, upstream, readout):
            count += 1
    return count


def shortest_path_count(graph: nx.DiGraph, node: int, readout: int) -> int:
    return sum(1 for _ in nx.all_shortest_paths(graph, node, readout))


def graph_feature_rows(record: GraphRecord) -> list[dict[str, Any]]:
    spec = record.graph
    graph = adjacency_graph(spec)
    readout = int(spec["readout_node"])
    horizon = int(spec["max_rounds"])
    distances = {
        node: int(nx.shortest_path_length(graph, node, readout)) for node in graph.nodes
    }
    components = list(nx.strongly_connected_components(graph))
    component_size = {node: len(component) for component in components for node in component}
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    outward_closeness = nx.closeness_centrality(graph.reverse(copy=False))
    source_count = sum(graph.in_degree(node) == 0 for node in graph.nodes)
    max_scc_size = max(len(component) for component in components)
    graph_has_cycle = int(not nx.is_directed_acyclic_graph(graph))
    nonreadout_distances = [value for node, value in distances.items() if node != readout]
    metadata = spec.get("metadata", {})
    message_opportunities = int(
        metadata.get(
            "message_opportunities",
            sum(metadata.get("active_edge_count_by_round", [])),
        )
    )

    rows: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node == readout:
            continue
        simple_paths = sum(1 for _ in nx.all_simple_paths(graph, node, readout))
        row = {
            "stratum": record.stratum,
            "graph_id": spec["graph_id"],
            "node_id": node,
            "node_count": int(spec["node_count"]),
            "edge_count": graph.number_of_edges(),
            "distance_to_readout": distances[node],
            "direct_to_readout": int(graph.has_edge(node, readout)),
            "in_degree": int(graph.in_degree(node)),
            "out_degree": int(graph.out_degree(node)),
            "is_source": int(graph.in_degree(node) == 0),
            "descendant_count": len(nx.descendants(graph, node)),
            "ancestor_count": len(nx.ancestors(graph, node)),
            "simple_path_count": simple_paths,
            "shortest_path_count": shortest_path_count(graph, node, readout),
            "walk_count_to_readout": walk_count_to_readout(graph, node, readout, horizon),
            "edge_connectivity_to_readout": nx.edge_connectivity(graph, node, readout),
            "node_connectivity_to_readout": nx.node_connectivity(graph, node, readout),
            "postdominated_upstream_count": postdominated_upstream_count(
                graph, node, readout
            ),
            "betweenness": float(betweenness[node]),
            "outward_closeness": float(outward_closeness[node]),
            "scc_size": int(component_size[node]),
            "in_nontrivial_cycle": int(component_size[node] > 1),
            "readout_in_degree": int(graph.in_degree(readout)),
            "source_count": int(source_count),
            "max_scc_size": int(max_scc_size),
            "graph_has_cycle": graph_has_cycle,
            "mean_distance_to_readout": float(np.mean(nonreadout_distances)),
            "std_distance_to_readout": float(np.std(nonreadout_distances)),
            "message_opportunities": message_opportunities,
        }
        row.update(record.node_outcomes[node])
        rows.append(row)
    return rows


def audit_feature_frame(frame: pd.DataFrame, status: dict[str, Any]) -> dict[str, Any]:
    expected_rows = sum(
        int(item["selected_graphs"]) * (int(item["n"]) - 1) for item in status["strata"]
    )
    errors: list[str] = []
    if len(frame) != expected_rows:
        errors.append(f"feature rows {len(frame)} != expected {expected_rows}")
    if frame.duplicated(["graph_id", "node_id"]).any():
        errors.append("duplicate graph-node feature rows")
    if bool((frame["node_id"] == frame["node_count"] - 1).any()):
        errors.append("readout node appears as an attack position")
    if frame[list(FULL_FEATURES)].isna().any().any():
        errors.append("static feature matrix contains missing values")
    if not frame["paired_accuracy_drop"].between(-1.0, 1.0).all():
        errors.append("paired accuracy drop lies outside [-1,1]")
    if not frame["induced_readout_target_rate"].between(0.0, 1.0).all():
        errors.append("target induction rate lies outside [0,1]")
    return {
        "passed": not errors,
        "errors": errors,
        "expected_rows": expected_rows,
        "observed_rows": len(frame),
        "graph_count": int(frame["graph_id"].nunique()),
        "stratum_count": int(frame["stratum"].nunique()),
        "feature_count": len(FULL_FEATURES),
    }


def fallback_mean(
    train: pd.DataFrame,
    test_row: pd.Series,
    outcome: str,
    include_distance: bool,
) -> float:
    conditions = train["stratum"] == test_row["stratum"]
    if include_distance:
        matched = train.loc[
            conditions & (train["distance_to_readout"] == test_row["distance_to_readout"])
        ]
        if not matched.empty:
            return float(matched[outcome].mean())
    matched = train.loc[conditions]
    if not matched.empty:
        return float(matched[outcome].mean())
    matched = train.loc[train["node_count"] == test_row["node_count"]]
    if not matched.empty:
        return float(matched[outcome].mean())
    return float(train[outcome].mean())


def make_ridge(feature_names: tuple[str, ...]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(feature_names)),
            (
                "stratum",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["stratum"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("features", preprocessor), ("ridge", Ridge())])


def ridge_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, float]:
    model = make_ridge(feature_names)
    groups = train["graph_id"].to_numpy()
    unique_groups = np.unique(groups)
    inner_splits = min(5, len(unique_groups))
    if inner_splits < 2:
        raise ValueError("ridge tuning requires at least two training graphs")
    search = GridSearchCV(
        model,
        {"ridge__alpha": list(RIDGE_ALPHAS)},
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=inner_splits),
        n_jobs=1,
    )
    search.fit(train, train[outcome], groups=groups)
    predictions = search.predict(test)
    if outcome == "induced_readout_target_rate":
        predictions = np.clip(predictions, 0.0, 1.0)
    return predictions, float(search.best_params_["ridge__alpha"])


def leave_one_graph_predictions(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    splitter = LeaveOneGroupOut()
    groups = frame["graph_id"].to_numpy()
    for train_index, test_index in splitter.split(frame, frame[outcome], groups):
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        held_graph = str(test["graph_id"].iloc[0])
        base = test[["stratum", "graph_id", "node_id", outcome]].copy()
        base = base.rename(columns={outcome: "observed"})

        global_part = base.copy()
        global_part["model"] = "global_mean"
        global_part["prediction"] = float(train[outcome].mean())
        global_part["selected_alpha"] = np.nan
        rows.append(global_part)

        for model_name, include_distance in (
            ("stratum_mean", False),
            ("distance_lookup", True),
        ):
            part = base.copy()
            part["model"] = model_name
            part["prediction"] = [
                fallback_mean(train, row, outcome, include_distance)
                for _, row in test.iterrows()
            ]
            part["selected_alpha"] = np.nan
            rows.append(part)

        for model_name, features in FEATURE_SETS.items():
            prediction, alpha = ridge_predictions(train, test, outcome, features)
            part = base.copy()
            part["model"] = model_name
            part["prediction"] = prediction
            part["selected_alpha"] = alpha
            rows.append(part)

        if any(part["graph_id"].iloc[0] != held_graph for part in rows[-7:]):
            raise AssertionError("outer-fold prediction mixed held-out graphs")
    result = pd.concat(rows, ignore_index=True)
    result.insert(0, "outcome", outcome)
    return result


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.all(observed == observed[0]) or np.all(predicted == predicted[0]):
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    centered_x = x - mean_x
    centered_y = y - mean_y
    covariance = float(np.average(centered_x * centered_y, weights=weights))
    variance_x = float(np.average(centered_x**2, weights=weights))
    variance_y = float(np.average(centered_y**2, weights=weights))
    if variance_x <= 0 or variance_y <= 0:
        return float("nan")
    return covariance / math.sqrt(variance_x * variance_y)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    within_graph: list[float] = []
    top1: list[float] = []
    for _, group in frame.groupby("graph_id", sort=False):
        correlation = safe_spearman(
            group["observed"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
        )
        if math.isfinite(correlation):
            within_graph.append(correlation)
        observed_max = float(group["observed"].max())
        predicted_max = float(group["prediction"].max())
        observed_top = set(group.loc[np.isclose(group["observed"], observed_max), "node_id"])
        predicted_top = set(
            group.loc[np.isclose(group["prediction"], predicted_max), "node_id"]
        )
        top1.append(len(observed_top & predicted_top) / len(predicted_top))
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
        "spearman": safe_spearman(observed, predicted),
        "mean_within_graph_spearman": (
            float(np.mean(within_graph)) if within_graph else float("nan")
        ),
        "top1_vulnerable_accuracy": float(np.mean(top1)),
    }


def bootstrap_prediction_metrics(
    frame: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    graph_ids = [str(value) for value in frame["graph_id"].unique()]
    graph_index = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    row_graph_index = frame["graph_id"].map(graph_index).to_numpy(dtype=int)
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    observed_rank = pd.Series(observed).rank(method="average").to_numpy(dtype=float)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=float)
    per_graph_within: list[float] = []
    per_graph_top1: list[float] = []
    for graph_id in graph_ids:
        metrics = prediction_metrics(frame.loc[frame["graph_id"] == graph_id])
        per_graph_within.append(metrics["mean_within_graph_spearman"])
        per_graph_top1.append(metrics["top1_vulnerable_accuracy"])
    values: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled_indices = rng.integers(0, len(graph_ids), size=len(graph_ids))
        graph_weights = np.bincount(sampled_indices, minlength=len(graph_ids)).astype(float)
        row_weights = graph_weights[row_graph_index]
        active = row_weights > 0
        active_weights = row_weights[active]
        active_observed = observed[active]
        active_predicted = predicted[active]
        error = active_observed - active_predicted
        weighted_observed_mean = float(np.average(active_observed, weights=active_weights))
        residual_sum = float(np.sum(active_weights * error**2))
        total_sum = float(
            np.sum(active_weights * (active_observed - weighted_observed_mean) ** 2)
        )
        finite_within = np.isfinite(per_graph_within)
        within_weights = graph_weights[finite_within]
        metrics = {
            "mae": float(np.average(np.abs(error), weights=active_weights)),
            "rmse": float(np.average(error**2, weights=active_weights) ** 0.5),
            "r2": 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan"),
            "spearman": weighted_correlation(
                observed_rank[active], predicted_rank[active], active_weights
            ),
            "mean_within_graph_spearman": (
                float(
                    np.average(
                        np.asarray(per_graph_within)[finite_within], weights=within_weights
                    )
                )
                if within_weights.sum() > 0
                else float("nan")
            ),
            "top1_vulnerable_accuracy": float(
                np.average(per_graph_top1, weights=graph_weights)
            ),
        }
        for metric, value in metrics.items():
            if math.isfinite(value):
                values.setdefault(metric, []).append(value)
    return {
        metric: (float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975)))
        for metric, items in values.items()
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome, model), group in predictions.groupby(["outcome", "model"], sort=False):
        estimates = prediction_metrics(group)
        intervals = bootstrap_prediction_metrics(group, replicates, rng)
        alpha_values = group["selected_alpha"].dropna().to_numpy(dtype=float)
        alpha_mode = Counter(alpha_values).most_common(1)[0][0] if len(alpha_values) else np.nan
        for metric, estimate in estimates.items():
            low, high = intervals.get(metric, (np.nan, np.nan))
            rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "outer_unit": "held_out_graph",
                    "bootstrap_unit": "held_out_graph",
                    "ridge_alpha_mode": alpha_mode,
                }
            )
    return pd.DataFrame(rows)


def compare_full_model(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pair ridge_full against simpler models at the held-out-graph level."""
    rows: list[dict[str, Any]] = []
    references = ["stratum_mean", "distance_lookup", "ridge_distance", "ridge_routing"]
    for outcome in OUTCOMES:
        outcome_frame = predictions.loc[predictions["outcome"] == outcome]
        observed = outcome_frame.loc[
            outcome_frame["model"] == "ridge_full", ["graph_id", "node_id", "observed"]
        ]
        candidate = outcome_frame.loc[
            outcome_frame["model"] == "ridge_full", ["graph_id", "node_id", "prediction"]
        ].rename(columns={"prediction": "candidate_prediction"})
        for reference in references:
            reference_frame = outcome_frame.loc[
                outcome_frame["model"] == reference,
                ["graph_id", "node_id", "prediction"],
            ].rename(columns={"prediction": "reference_prediction"})
            paired = observed.merge(candidate, on=["graph_id", "node_id"], validate="one_to_one")
            paired = paired.merge(
                reference_frame, on=["graph_id", "node_id"], validate="one_to_one"
            )
            paired["candidate_absolute_error"] = (
                paired["observed"] - paired["candidate_prediction"]
            ).abs()
            paired["reference_absolute_error"] = (
                paired["observed"] - paired["reference_prediction"]
            ).abs()
            graph_errors = paired.groupby("graph_id", sort=False).agg(
                candidate_mae=("candidate_absolute_error", "mean"),
                reference_mae=("reference_absolute_error", "mean"),
            )
            graph_errors["mae_improvement"] = (
                graph_errors["reference_mae"] - graph_errors["candidate_mae"]
            )
            values = graph_errors["mae_improvement"].to_numpy(dtype=float)
            draws = np.mean(
                values[
                    rng.integers(0, len(values), size=(replicates, len(values)))
                ],
                axis=1,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "candidate": "ridge_full",
                    "reference": reference,
                    "graph_equal_mae_improvement": float(values.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "fraction_graphs_candidate_better": float(np.mean(values > 0)),
                    "bootstrap_unit": "held_out_graph",
                    "positive_means_candidate_better": True,
                }
            )
    return pd.DataFrame(rows)


def within_stratum_rank_correlation(frame: pd.DataFrame, feature: str, outcome: str) -> float:
    feature_rank = frame.groupby("stratum")[feature].rank(method="average", pct=True)
    outcome_rank = frame.groupby("stratum")[outcome].rank(method="average", pct=True)
    centered_feature = feature_rank - feature_rank.groupby(frame["stratum"]).transform("mean")
    centered_outcome = outcome_rank - outcome_rank.groupby(frame["stratum"]).transform("mean")
    if np.allclose(centered_feature, 0) or np.allclose(centered_outcome, 0):
        return float("nan")
    return float(np.corrcoef(centered_feature, centered_outcome)[0, 1])


def feature_correlations(
    frame: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    graph_ids = [str(value) for value in frame["graph_id"].unique()]
    graph_index = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    row_graph_index = frame["graph_id"].map(graph_index).to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        for feature in FULL_FEATURES:
            feature_rank = frame.groupby("stratum")[feature].rank(method="average", pct=True)
            outcome_rank = frame.groupby("stratum")[outcome].rank(method="average", pct=True)
            centered_feature = (
                feature_rank - feature_rank.groupby(frame["stratum"]).transform("mean")
            ).to_numpy(dtype=float)
            centered_outcome = (
                outcome_rank - outcome_rank.groupby(frame["stratum"]).transform("mean")
            ).to_numpy(dtype=float)
            unit_weights = np.ones(len(frame), dtype=float)
            estimate = weighted_correlation(centered_feature, centered_outcome, unit_weights)
            draws: list[float] = []
            for _ in range(replicates):
                sampled_indices = rng.integers(0, len(graph_ids), size=len(graph_ids))
                graph_weights = np.bincount(
                    sampled_indices, minlength=len(graph_ids)
                ).astype(float)
                row_weights = graph_weights[row_graph_index]
                value = weighted_correlation(
                    centered_feature, centered_outcome, row_weights
                )
                if math.isfinite(value):
                    draws.append(value)
            rows.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "within_stratum_rank_correlation": estimate,
                    "ci95_low": float(np.quantile(draws, 0.025)) if draws else np.nan,
                    "ci95_high": float(np.quantile(draws, 0.975)) if draws else np.nan,
                    "bootstrap_unit": "graph",
                }
            )
    return pd.DataFrame(rows)


def plot_model_metrics(summary: pd.DataFrame, outcome: str, path: Path) -> None:
    metrics = ["mae", "spearman", "top1_vulnerable_accuracy"]
    titles = ["LOO-graph MAE (lower is better)", "Pooled Spearman", "Top-1 vulnerable node"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        frame = summary.loc[(summary["outcome"] == outcome) & (summary["metric"] == metric)]
        frame = frame.reset_index(drop=True)
        y = np.arange(len(frame))
        lower = frame["estimate"] - frame["ci95_low"]
        upper = frame["ci95_high"] - frame["estimate"]
        axis.errorbar(
            frame["estimate"],
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
        )
        axis.set_yticks(y, frame["model"])
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle(outcome)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_observed_predicted(
    predictions: pd.DataFrame,
    outcome: str,
    model: str,
    path: Path,
) -> None:
    frame = predictions.loc[
        (predictions["outcome"] == outcome) & (predictions["model"] == model)
    ]
    figure, axis = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    for stratum, group in frame.groupby("stratum"):
        axis.scatter(group["observed"], group["prediction"], alpha=0.7, label=stratum)
    low = float(min(frame["observed"].min(), frame["prediction"].min()))
    high = float(max(frame["observed"].max(), frame["prediction"].max()))
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Observed")
    axis.set_ylabel("Held-out-graph prediction")
    axis.set_title(f"{model}: {outcome}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    audit: dict[str, Any],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    lines = [
        "# Classical structural explainability",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity",
        "",
        f"- Feature audit passed: `{audit['passed']}`",
        f"- Graphs: {audit['graph_count']}",
        f"- Node-level rows: {audit['observed_rows']}",
        f"- Declared full features: {audit['feature_count']}",
        "",
        "## Held-out-graph prediction",
        "",
        "| outcome | model | MAE | Spearman | within-graph Spearman | top-1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    pivot = summary.pivot(index=["outcome", "model"], columns="metric", values="estimate")
    for outcome in OUTCOMES:
        for model in [
            "global_mean",
            "stratum_mean",
            "distance_lookup",
            "ridge_distance",
            "ridge_local",
            "ridge_routing",
            "ridge_full",
        ]:
            row = pivot.loc[(outcome, model)]
            lines.append(
                f"| {outcome} | {model} | {row['mae']:.4f} | {row['spearman']:.3f} | "
                f"{row['mean_within_graph_spearman']:.3f} | "
                f"{row['top1_vulnerable_accuracy']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Paired MAE improvement of the full structural model",
            "",
            "Positive values mean that `ridge_full` has lower graph-equal MAE than the reference.",
            "",
            "| outcome | reference | improvement | 95% graph-bootstrap CI | graphs better |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['outcome']} | {row['reference']} | "
            f"{row['graph_equal_mae_improvement']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['fraction_graphs_candidate_better']:.3f} |"
        )
    lines.extend(["", "## Largest diagnostic rank correlations", ""])
    for outcome in OUTCOMES:
        finite = correlations.loc[
            (correlations["outcome"] == outcome)
            & correlations["within_stratum_rank_correlation"].notna()
        ].copy()
        finite["absolute"] = finite["within_stratum_rank_correlation"].abs()
        lines.append(f"### {outcome}")
        lines.append("")
        for _, row in finite.nlargest(5, "absolute").iterrows():
            lines.append(
                f"- `{row['feature']}`: {row['within_stratum_rank_correlation']:.3f} "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
            )
        lines.append("")
    lines.extend(
        [
            "## Claim guardrails",
            "",
            "- Every reported prediction is produced with an entire graph held out.",
            "- In-sample feature correlations are diagnostic, not causal evidence.",
            "- Intervals resample selected graphs only and do not cover task/model/seed variation.",
            "- Weak structural prediction would not by itself establish an LLM semantic mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_root / "orchestrator_status.json"
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError("pilot must be completed before structural analysis")
    records = load_records(run_root, status)
    frame = pd.DataFrame(
        row for record in records for row in graph_feature_rows(record)
    )
    audit = audit_feature_frame(frame, status)
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError("feature audit failed; see integrity_audit.json")

    predictions = pd.concat(
        [leave_one_graph_predictions(frame, outcome) for outcome in OUTCOMES],
        ignore_index=True,
    )
    rng = np.random.default_rng(args.seed)
    summary = summarize_predictions(predictions, args.bootstrap_replicates, rng)
    comparisons = compare_full_model(predictions, args.bootstrap_replicates, rng)
    correlations = feature_correlations(frame, args.bootstrap_replicates, rng)

    frame.to_csv(output_dir / "node_structural_features.csv", index=False, lineterminator="\n")
    predictions.to_csv(
        output_dir / "held_out_graph_predictions.csv", index=False, lineterminator="\n"
    )
    summary.to_csv(output_dir / "predictive_metrics.csv", index=False, lineterminator="\n")
    comparisons.to_csv(
        output_dir / "model_comparisons.csv", index=False, lineterminator="\n"
    )
    correlations.to_csv(
        output_dir / "feature_correlations.csv", index=False, lineterminator="\n"
    )
    for outcome in OUTCOMES:
        plot_model_metrics(summary, outcome, output_dir / f"model_metrics_{outcome}.png")
        plot_observed_predicted(
            predictions,
            outcome,
            "ridge_full",
            output_dir / f"observed_predicted_{outcome}.png",
        )
    (output_dir / "report.md").write_text(
        render_report(audit, summary, comparisons, correlations), encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "ridge_alphas": list(RIDGE_ALPHAS),
        "outcomes": list(OUTCOMES),
        "feature_sets": {name: list(values) for name, values in FEATURE_SETS.items()},
        "graph_count": audit["graph_count"],
        "node_rows": audit["observed_rows"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
