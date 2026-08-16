"""Task-cluster bootstrap for cross-scale CTOU graph and density-curve metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

QUANTITIES = (
    "utility",
    "robustness",
    "target_risk",
    "attack_loss",
    "u0",
    "delta_utility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_816)
    return parser.parse_args()


def task_graph_panel(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "n", "graph_id", "m", "rho", "task_id"]
    grouped = frame.groupby([*keys, "condition"], as_index=False, observed=True).agg(
        actual_correct=("actual_correct", "mean"),
        predicted_correct=("p_correct", "mean"),
        actual_target=("actual_target", "mean"),
        predicted_target=("p_target", "mean"),
        actual_u0=("round0_correct", "mean"),
    )
    clean = grouped[grouped.condition.eq("clean")].set_index(keys)
    attack = grouped[grouped.condition.eq("attack")].set_index(keys)
    common = clean.index.intersection(attack.index)
    panel = pd.DataFrame(index=common).reset_index()
    panel["actual_utility"] = clean.loc[common, "actual_correct"].to_numpy(float)
    panel["predicted_utility"] = clean.loc[common, "predicted_correct"].to_numpy(float)
    panel["actual_robustness"] = attack.loc[common, "actual_correct"].to_numpy(float)
    panel["predicted_robustness"] = attack.loc[common, "predicted_correct"].to_numpy(float)
    panel["actual_target_risk"] = attack.loc[common, "actual_target"].to_numpy(float)
    panel["predicted_target_risk"] = attack.loc[common, "predicted_target"].to_numpy(float)
    panel["actual_u0"] = clean.loc[common, "actual_u0"].to_numpy(float)
    panel["predicted_u0"] = panel.actual_u0
    panel["actual_attack_loss"] = panel.actual_utility - panel.actual_robustness
    panel["predicted_attack_loss"] = panel.predicted_utility - panel.predicted_robustness
    panel["actual_delta_utility"] = panel.actual_utility - panel.actual_u0
    panel["predicted_delta_utility"] = panel.predicted_utility - panel.predicted_u0
    return panel


def matrix_by_unit(
    frame: pd.DataFrame,
    *,
    unit: str,
    value: str,
    tasks: list[str],
) -> tuple[np.ndarray, list[object]]:
    pivot = frame.pivot_table(
        index=unit,
        columns="task_id",
        values=value,
        aggfunc="mean",
        observed=True,
    ).reindex(columns=tasks)
    if pivot.isna().any().any():
        raise RuntimeError(f"incomplete task-by-{unit} panel for {value}")
    return pivot.to_numpy(float), pivot.index.tolist()


def rowwise_spearman(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual_rank = rankdata(actual, axis=1)
    predicted_rank = rankdata(predicted, axis=1)
    actual_centered = actual_rank - actual_rank.mean(axis=1, keepdims=True)
    predicted_centered = predicted_rank - predicted_rank.mean(axis=1, keepdims=True)
    numerator = (actual_centered * predicted_centered).sum(axis=1)
    denominator = np.sqrt((actual_centered**2).sum(axis=1) * (predicted_centered**2).sum(axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(actual), np.nan, dtype=float),
        where=denominator > 0,
    )


def summarize_distribution(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(finite, [0.025, 0.975]))  # type: ignore[return-value]


def bootstrap_metrics(frame: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (model, n), group in frame.groupby(["model", "n"], observed=True):
        tasks = sorted(group.task_id.unique())
        weights = rng.multinomial(
            len(tasks), np.full(len(tasks), 1.0 / len(tasks)), size=replicates
        ).astype(float)
        for level, unit in (("graph", "graph_id"), ("m_curve", "m")):
            units = group[[unit]].drop_duplicates().shape[0]
            for quantity in QUANTITIES:
                actual, unit_labels = matrix_by_unit(
                    group,
                    unit=unit,
                    value=f"actual_{quantity}",
                    tasks=tasks,
                )
                predicted, predicted_units = matrix_by_unit(
                    group,
                    unit=unit,
                    value=f"predicted_{quantity}",
                    tasks=tasks,
                )
                if unit_labels != predicted_units:
                    raise RuntimeError("actual/predicted unit order mismatch")
                denominator = weights.sum(axis=1)
                actual_boot = (actual @ weights.T / denominator).T
                predicted_boot = (predicted @ weights.T / denominator).T
                mae = np.abs(predicted_boot - actual_boot).mean(axis=1)
                rank = rowwise_spearman(actual_boot, predicted_boot)
                observed_actual = actual.mean(axis=1)
                observed_predicted = predicted.mean(axis=1)
                point_mae = float(np.abs(observed_predicted - observed_actual).mean())
                point_rank = (
                    float(spearmanr(observed_actual, observed_predicted).statistic)
                    if np.ptp(observed_actual) > 0 and np.ptp(observed_predicted) > 0
                    else np.nan
                )
                mae_low, mae_high = summarize_distribution(mae)
                rank_low, rank_high = summarize_distribution(rank)
                rows.extend(
                    [
                        {
                            "level": level,
                            "model": model,
                            "n": int(n),
                            "quantity": quantity,
                            "metric": "mae",
                            "units": units,
                            "tasks": len(tasks),
                            "replicates": replicates,
                            "estimate": point_mae,
                            "ci_low": mae_low,
                            "ci_high": mae_high,
                        },
                        {
                            "level": level,
                            "model": model,
                            "n": int(n),
                            "quantity": quantity,
                            "metric": "spearman",
                            "units": units,
                            "tasks": len(tasks),
                            "replicates": replicates,
                            "estimate": point_rank,
                            "ci_low": rank_low,
                            "ci_high": rank_high,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.replicates < 1_000:
        raise ValueError("replicates must be at least 1000")
    frame = pd.read_csv(args.endpoint_predictions)
    panel = task_graph_panel(frame)
    result = bootstrap_metrics(panel, replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
