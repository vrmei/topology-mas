"""Plot observed and predicted C/T/O/U transition curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS = ("degroot_equal", "ctou_table", "ctou_logit")
LABELS = {
    "degroot_equal": "DeGroot (equal weight)",
    "ctou_table": "CTOU table",
    "ctou_logit": "CTOU logistic",
}
COLORS = {
    "degroot_equal": "#D55E00",
    "ctou_table": "#0072B2",
    "ctou_logit": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def plot_transition(frame: pd.DataFrame, evaluation: str, output: Path) -> None:
    subset = frame[
        frame.evaluation.eq(evaluation) & frame.receiver_scope.eq("readout")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, n in zip(axes, (5, 8), strict=True):
        panel = subset[subset.n.eq(n)]
        observed = (
            panel[["m", "observed_rate"]]
            .drop_duplicates()
            .sort_values("m")
        )
        axis.plot(
            observed.m,
            observed.observed_rate,
            color="#111111",
            marker="o",
            linewidth=2.4,
            label="Observed LLM",
            zorder=5,
        )
        for model in MODELS:
            curve = panel[panel.model.eq(model)].sort_values("m")
            axis.plot(
                curve.m,
                curve.predicted_rate,
                color=COLORS[model],
                marker=".",
                linewidth=1.8,
                label=LABELS[model],
            )
        axis.set_title(f"n = {n}")
        axis.set_xlabel("Directed edge count m")
        axis.set_ylabel("Transition probability")
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Readout target-error adoption" if evaluation == "adoption"
        else "Readout target-error recovery",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.input_dir / "transition_curve_predictions.csv")
    plot_transition(curves, "adoption", output_dir / "readout_adoption_prediction.png")
    plot_transition(curves, "recovery", output_dir / "readout_recovery_prediction.png")


if __name__ == "__main__":
    main()
