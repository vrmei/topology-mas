#!/usr/bin/env python3
"""Diagnose sawtooth attack loss across edge-count levels in Qwen/AIME.

Primary outcome is graph-level final-round attack loss L = U - R from the
authoritative 61-graph table.  The analysis is deliberately diagnostic:
edge-count levels contain independently sampled topologies, so adjacent means
are not interpreted as marginal effects of adding one edge.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def graph_features(record: dict) -> dict:
    n = int(record["node_count"])
    m = len(record["edges"])
    readout = int(record["readout_node"])
    meta = record.get("metadata", {})
    distances = list(meta["distances_to_readout"])
    non_readout_distances = [int(d) for i, d in enumerate(distances) if i != readout]
    active_nodes = [int(x) for x in meta["active_node_count_by_round"]]
    active_edges = [int(x) for x in meta["active_edge_count_by_round"]]
    shells = {d: non_readout_distances.count(d) for d in (1, 2, 3)}
    indegree = sum(int(edge["target"]) == readout for edge in record["edges"])
    return {
        "graph_id": record["graph_id"],
        "n": n,
        "m": m,
        "odd_m": m % 2,
        "readout_indegree": indegree,
        "N_d1": shells[1],
        "N_d2": shells[2],
        "N_d3": shells[3],
        "mean_attacker_distance": float(np.mean(non_readout_distances)),
        "direct_exposure_rate": indegree / (n - 1),
        "active_node_rounds": int(sum(active_nodes)),
        "message_opportunities": int(sum(active_edges)),
        "active_nodes_by_round": "/".join(map(str, active_nodes)),
        "active_edges_by_round": "/".join(map(str, active_edges)),
        "has_directed_cycle": bool(meta.get("has_directed_cycle", False)),
    }


def permutation_spearman(x: pd.Series, y: pd.Series, reps: int, seed: int) -> tuple[float, float]:
    keep = x.notna() & y.notna()
    xv = x.loc[keep].to_numpy(float)
    yv = y.loc[keep].to_numpy(float)
    xr = rankdata(xv)
    yr = rankdata(yv)
    xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
    yr = (yr - yr.mean()) / np.linalg.norm(yr - yr.mean())
    rho = float(np.dot(xr, yr))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        trial = float(np.dot(xr, rng.permutation(yr)))
        exceed += abs(trial) >= abs(rho) - 1e-15
    return rho, (exceed + 1) / (reps + 1)


def loocv_models(frame: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "smooth_density": ["m", "m2"],
        "smooth_density_plus_parity": ["m", "m2", "odd_m"],
        "distance_shells": ["m", "m2", "N_d2", "N_d3"],
        "execution_opportunities": ["m", "m2", "active_node_rounds", "message_opportunities"],
        "compact_structure": ["m", "m2", "N_d2", "N_d3", "message_opportunities"],
    }
    rows = []
    for name, columns in specs.items():
        truth: list[float] = []
        pred: list[float] = []
        for held_m in sorted(frame["m"].unique()):
            train = frame[frame["m"] != held_m]
            test = frame[frame["m"] == held_m]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[columns], train["L"])
            truth.extend(test["L"].tolist())
            pred.extend(model.predict(test[columns]).tolist())
        rows.append({
            "model": name,
            "features": ",".join(columns),
            "leave_one_m_out_MAE": mean_absolute_error(truth, pred),
            "leave_one_m_out_R2": r2_score(truth, pred),
        })
    return pd.DataFrame(rows).sort_values("leave_one_m_out_MAE")


def parity_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("L", "L_complete_trace"):
        sub = frame.dropna(subset=[outcome]).copy()
        for trend, cols in {
            "linear": ["m", "odd_m"],
            "quadratic": ["m", "m2", "odd_m"],
        }.items():
            X = sm.add_constant(sub[cols], has_constant="add")
            fit = sm.OLS(sub[outcome], X).fit(cov_type="HC3")
            rows.append({
                "outcome": outcome,
                "smooth_trend": trend,
                "N_graphs": len(sub),
                "odd_minus_even_coef": fit.params["odd_m"],
                "HC3_SE": fit.bse["odd_m"],
                "HC3_p": fit.pvalues["odd_m"],
                "R2": fit.rsquared,
                "warning": "Exploratory only: parity is deterministic in m and not separately randomized.",
            })
    return pd.DataFrame(rows)


def nested_pairs(graphs: list[dict], frame: pd.DataFrame) -> pd.DataFrame:
    edge_sets = {
        g["graph_id"]: {(int(e["source"]), int(e["target"])) for e in g["edges"]}
        for g in graphs
    }
    indexed = frame.set_index("graph_id")
    rows = []
    for low_id, low_edges in edge_sets.items():
        low_m = len(low_edges)
        for high_id, high_edges in edge_sets.items():
            if len(high_edges) != low_m + 1 or not low_edges.issubset(high_edges):
                continue
            low = indexed.loc[low_id]
            high = indexed.loc[high_id]
            added = next(iter(high_edges - low_edges))
            rows.append({
                "low_graph": low_id,
                "high_graph": high_id,
                "m_low": low_m,
                "added_edge": f"{added[0]}->{added[1]}",
                "delta_L": high["L"] - low["L"],
                "delta_U": high["U"] - low["U"],
                "delta_R": high["R"] - low["R"],
                "delta_active_node_rounds": high["active_node_rounds"] - low["active_node_rounds"],
                "delta_message_opportunities": high["message_opportunities"] - low["message_opportunities"],
                "delta_N_d1": high["N_d1"] - low["N_d1"],
                "delta_N_d2": high["N_d2"] - low["N_d2"],
                "delta_N_d3": high["N_d3"] - low["N_d3"],
            })
    return pd.DataFrame(rows)


def exact_balanced_label_test(values: np.ndarray, observed_labels: np.ndarray) -> tuple[float, float, int]:
    """Exact two-sided test over all balanced 6-vs-6 assignments."""
    observed = float(values[observed_labels].mean() - values[~observed_labels].mean())
    diffs = []
    for selected in itertools.combinations(range(len(values)), int(observed_labels.sum())):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(selected)] = True
        diffs.append(float(values[mask].mean() - values[~mask].mean()))
    p = sum(abs(x) >= abs(observed) - 1e-15 for x in diffs) / len(diffs)
    return observed, p, len(diffs)


def parity_dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("L", "L_complete_trace"):
        by_m = frame.groupby("m")[outcome].agg(["count", "mean", "std"]).reset_index()
        # m=16 has one topology and no within-density variance; compare the six
        # odd and six even levels that each contain five sampled graphs.
        by_m = by_m[by_m["count"] == 5].copy()
        odd = (by_m["m"] % 2 == 1).to_numpy()
        for metric, values in {
            "within_m_sd": by_m["std"].to_numpy(float),
            "absolute_density_mean": by_m["mean"].abs().to_numpy(float),
            "signed_density_mean": by_m["mean"].to_numpy(float),
        }.items():
            diff, p, assignments = exact_balanced_label_test(values, odd)
            rows.append({
                "outcome": outcome,
                "metric": metric,
                "odd_mean": float(values[odd].mean()),
                "even_mean": float(values[~odd].mean()),
                "odd_minus_even": diff,
                "exact_two_sided_p": p,
                "m_levels": len(by_m),
                "balanced_assignments": assignments,
                "warning": "Exploratory density-level test; parity was not randomized.",
            })
    return pd.DataFrame(rows)


def task_bootstrap_density(endpoint_path: Path, reps: int, seed: int) -> pd.DataFrame:
    data = pd.read_parquet(endpoint_path)
    data = data[(data["system"] == "Qwen-AIME-full61") & (data["round"] == 3)].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for m, group in data.groupby("m"):
        task_means = group.groupby("task_id")["attack_loss"].mean()
        values = task_means.to_numpy(float)
        draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
        rows.append({
            "m": int(m),
            "complete_trace_cells": len(group),
            "tasks": len(values),
            "complete_trace_mean_L": float(values.mean()),
            "task_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "task_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        })
    return pd.DataFrame(rows)


def within_m_permutation_associations(
    frame: pd.DataFrame, predictors: list[str], reps: int, seed: int
) -> pd.DataFrame:
    """Test topology variation after removing each density-level mean.

    Outcome labels are permuted only among the five graphs sharing the same m.
    The complete graph is excluded because it has no within-m comparison.
    """
    data = frame[frame.groupby("m")["graph_id"].transform("size") > 1].copy().reset_index(drop=True)
    groups = [group.index.to_numpy() for _, group in data.groupby("m")]
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in ("U", "R", "L"):
        y = (data[outcome] - data.groupby("m")[outcome].transform("mean")).to_numpy(float)
        y = (y - y.mean()) / np.linalg.norm(y - y.mean())
        for predictor in predictors:
            x = (data[predictor] - data.groupby("m")[predictor].transform("mean")).to_numpy(float)
            if np.linalg.norm(x - x.mean()) == 0:
                continue
            x = (x - x.mean()) / np.linalg.norm(x - x.mean())
            observed = float(np.dot(x, y))
            exceed = 0
            for _ in range(reps):
                permuted = np.empty(len(y))
                for indices in groups:
                    permuted[indices] = rng.permutation(y[indices])
                exceed += abs(float(np.dot(x, permuted))) >= abs(observed) - 1e-15
            rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "within_m_pearson_r": observed,
                "within_m_permutation_p": (exceed + 1) / (reps + 1),
                "N_graphs": len(data),
                "m_levels": len(groups),
            })
    return pd.DataFrame(rows).sort_values(["outcome", "within_m_permutation_p"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--official-ur", type=Path, required=True)
    parser.add_argument("--roundwise-ur", type=Path, required=True)
    parser.add_argument("--endpoint-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20761304)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    graphs = read_jsonl(args.graphs)
    features = pd.DataFrame([graph_features(g) for g in graphs])
    official = pd.read_csv(args.official_ur).rename(columns={"U_minus_R": "L"})
    official = official[["graph_id", "m", "U", "R", "L"]]
    roundwise = pd.read_csv(args.roundwise_ur)
    complete = roundwise[
        (roundwise["system"] == "Qwen-AIME-full61") & (roundwise["round"] == 3)
    ][["graph_id", "L"]].rename(columns={"L": "L_complete_trace"})
    frame = features.merge(official, on=["graph_id", "m"], validate="one_to_one")
    frame = frame.merge(complete, on="graph_id", how="left", validate="one_to_one")
    frame["m2"] = frame["m"] ** 2
    frame.to_csv(args.out / "graph_structural_and_loss.csv", index=False)

    density = frame.groupby("m", as_index=False).agg(
        n_graphs=("graph_id", "size"),
        mean_U=("U", "mean"),
        mean_R=("R", "mean"),
        mean_L=("L", "mean"),
        median_L=("L", "median"),
        sd_L=("L", "std"),
        min_L=("L", "min"),
        max_L=("L", "max"),
        readout_indegree=("readout_indegree", "mean"),
        N_d1=("N_d1", "mean"),
        N_d2=("N_d2", "mean"),
        N_d3=("N_d3", "mean"),
        active_node_rounds=("active_node_rounds", "mean"),
        message_opportunities=("message_opportunities", "mean"),
        mean_attacker_distance=("mean_attacker_distance", "mean"),
        direct_exposure_rate=("direct_exposure_rate", "mean"),
    )
    density["sem_L"] = density["sd_L"] / np.sqrt(density["n_graphs"])

    sensitivity_rows = []
    for m, group in frame.groupby("m"):
        mean_l = group["L"].mean()
        loo = [group.loc[group.index != i, "L"].mean() for i in group.index] if len(group) > 1 else [np.nan]
        sensitivity_rows.append({
            "m": m,
            "n_graphs": len(group),
            "mean_L": mean_l,
            "positive_graphs": int((group["L"] > 0).sum()),
            "negative_graphs": int((group["L"] < 0).sum()),
            "zero_graphs": int((group["L"] == 0).sum()),
            "loo_mean_min": np.nanmin(loo) if len(group) > 1 else np.nan,
            "loo_mean_max": np.nanmax(loo) if len(group) > 1 else np.nan,
            "max_abs_single_graph_influence": np.nanmax(np.abs(np.asarray(loo) - mean_l)) if len(group) > 1 else np.nan,
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    # Merge on identifiers only: independently computed floating-point means can
    # differ at the last bit and must not silently drop density levels.
    density = density.merge(
        sensitivity.drop(columns="mean_L"),
        on=["m", "n_graphs"],
        validate="one_to_one",
    )
    density.to_csv(args.out / "density_structure_and_dispersion.csv", index=False)

    predictors = [
        "m", "odd_m", "readout_indegree", "N_d2", "N_d3",
        "active_node_rounds", "message_opportunities", "mean_attacker_distance",
        "direct_exposure_rate",
    ]
    assoc_rows = []
    for outcome in ("U", "R", "L"):
        for predictor in predictors:
            rho, p = permutation_spearman(frame[predictor], frame[outcome], args.permutations, args.seed)
            assoc_rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "rho": rho,
                "permutation_p": p,
                "N_graphs": len(frame),
            })
    associations = pd.DataFrame(assoc_rows).sort_values("permutation_p")
    associations.to_csv(args.out / "loss_structure_associations.csv", index=False)
    within_m = within_m_permutation_associations(frame, predictors[2:], args.permutations, args.seed)
    within_m.to_csv(args.out / "within_m_structure_associations.csv", index=False)

    parity = parity_regressions(frame)
    parity.to_csv(args.out / "parity_regression_diagnostics.csv", index=False)
    dispersion = parity_dispersion(frame)
    dispersion.to_csv(args.out / "parity_dispersion_exact_tests.csv", index=False)
    bootstrap = task_bootstrap_density(args.endpoint_cells, args.permutations, args.seed)
    bootstrap.to_csv(args.out / "density_task_bootstrap_complete_trace.csv", index=False)
    cv = loocv_models(frame)
    cv.to_csv(args.out / "leave_one_m_out_model_comparison.csv", index=False)
    nested = nested_pairs(graphs, frame)
    nested.to_csv(args.out / "observed_nested_one_edge_pairs.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    blue = "#1976D2"

    fig, ax = plt.subplots(figsize=(14, 7.5))
    sns.stripplot(data=frame, x="m", y="L", jitter=0.15, size=8, alpha=0.72, color=blue, ax=ax)
    order = sorted(frame["m"].unique())
    means = frame.groupby("m")["L"].mean().reindex(order)
    ax.plot(range(len(order)), means, color="#D55E00", marker="D", linewidth=2.4, label="mean across graphs")
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.5)
    ax.set(xlabel="Edge count, m", ylabel="Final attack loss, U - R",
           title="All graph-level attack losses at each edge count")
    ax.yaxis.set_major_formatter(lambda x, _: f"{100*x:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "graph_attack_loss_strip_by_m.png", dpi=220)
    plt.close(fig)

    structural = [
        ("readout_indegree", "Readout indegree / N(d=1)"),
        ("N_d2", "N(d=2)"),
        ("N_d3", "N(d=3)"),
        ("active_node_rounds", "Active node-rounds"),
        ("message_opportunities", "Message opportunities"),
        ("mean_attacker_distance", "Mean attacker→readout distance"),
        ("direct_exposure_rate", "Direct exposure rate"),
        ("L", "Attack loss, U-R"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True)
    for ax, (column, label) in zip(axes.flat, structural):
        sns.stripplot(data=frame, x="m", y=column, jitter=0.12, alpha=0.45, size=5, color=blue, ax=ax)
        vals = frame.groupby("m")[column].mean().reindex(order)
        ax.plot(range(len(order)), vals, color="#D55E00", marker="o", linewidth=2)
        ax.set(xlabel="Edge count, m", ylabel=label)
        if column in {"L", "direct_exposure_rate"}:
            ax.yaxis.set_major_formatter(lambda x, _: f"{100*x:.1f}%")
    fig.suptitle("Structural thresholds and attack loss across edge counts", y=1.005)
    fig.tight_layout()
    fig.savefig(args.out / "structural_metrics_by_m.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    scatter_cols = ["N_d3", "active_node_rounds", "message_opportunities", "direct_exposure_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, column in zip(axes.flat, scatter_cols):
        sns.scatterplot(data=frame, x=column, y="L", hue="m", palette="viridis", s=75, ax=ax, legend=False)
        sns.regplot(data=frame, x=column, y="L", scatter=False, color="0.25", line_kws={"linestyle": "--"}, ax=ax)
        rho, p = permutation_spearman(frame[column], frame["L"], args.permutations, args.seed)
        ax.set_title(f"rho={rho:.3f}, permutation p={p:.4f}")
        ax.yaxis.set_major_formatter(lambda x, _: f"{100*x:.1f}%")
    fig.suptitle("Attack loss versus candidate structural mechanisms", y=1.01)
    fig.tight_layout()
    fig.savefig(args.out / "loss_vs_structural_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "primary_outcome": "authoritative final-round graph-level U-R",
        "n_graphs": int(len(frame)),
        "m_levels": int(frame["m"].nunique()),
        "graphs_per_m": {str(int(k)): int(v) for k, v in frame.groupby("m").size().items()},
        "largest_density_mean_losses": density.nlargest(5, "mean_L")[["m", "mean_L", "sd_L", "min_L", "max_L"]].to_dict("records"),
        "strongest_associations": associations.head(10).to_dict("records"),
        "within_m_associations": within_m.groupby("outcome", group_keys=False).head(4).to_dict("records"),
        "parity_diagnostics": parity.to_dict("records"),
        "parity_dispersion": dispersion.to_dict("records"),
        "leave_one_m_out_models": cv.to_dict("records"),
        "nested_one_edge_pairs": int(len(nested)),
        "identification_warning": "Adjacent m levels are independent topology samples, not a nested edge-addition sequence.",
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
