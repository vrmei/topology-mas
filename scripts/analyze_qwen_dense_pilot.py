"""Aggregate dense-m Qwen pilot curves and matching historical 50-task references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pilot-root", type=Path, required=True)
    result.add_argument("--old500-root", type=Path, required=True)
    result.add_argument("--old500-revised-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def load_pilot(pilot_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_frames = []
    run_frames = []
    for stratum in sorted((pilot_root / "strata").glob("n*_m*")):
        analysis = stratum / "analysis-v1"
        if not (analysis / "graph_metrics.csv").exists():
            continue
        graph = pd.read_csv(analysis / "graph_metrics.csv")
        graph.insert(0, "stratum", stratum.name)
        graph["normalized_density"] = graph.edge_count / (graph.node_count - 1) ** 2
        graph["u0"] = graph.readout_round_zero_accuracy
        graph["u_t"] = graph.utility
        graph["delta_u"] = graph.u_t - graph.u0
        graph_frames.append(graph)
        runs = pd.read_json(analysis / "run_metrics.jsonl", lines=True)
        runs.insert(0, "stratum", stratum.name)
        run_frames.append(runs)
    if not graph_frames:
        raise ValueError("pilot contains no complete analyzed strata")
    return pd.concat(graph_frames, ignore_index=True), pd.concat(run_frames, ignore_index=True)


def summarize_m(graphs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["u0", "u_t", "delta_u", "r_mean", "r_worst", "d_mean"]
    rows = []
    for (n, m), frame in graphs.groupby(["node_count", "edge_count"], sort=True):
        row: dict[str, object] = {
            "n": int(n),
            "m": int(m),
            "normalized_density": float(m / (n - 1) ** 2),
            "graph_count": len(frame),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd_across_graphs"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def old_reference(
    *,
    task_ids: set[str],
    old500_root: Path,
    revised_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows = []
    audit = []
    roots = [("base", old500_root), ("revised", revised_root)]
    seen_strata: set[str] = set()
    for source, root in roots:
        for stratum in sorted((root / "strata").glob("n*_m*")):
            if stratum.name in seen_strata:
                continue
            analysis = stratum / "analysis-v1"
            runs_path = analysis / "run_metrics.jsonl"
            if not runs_path.exists():
                continue
            runs = pd.read_json(runs_path, lines=True)
            selected = runs[runs.task_id.isin(task_ids)].copy()
            found = set(selected.task_id.unique())
            if found != task_ids:
                audit.append(
                    {
                        "stratum": stratum.name,
                        "source": source,
                        "status": "skipped_incomplete_task_match",
                        "matched_tasks": len(found),
                    }
                )
                continue
            clean = selected[selected.condition == "clean"]
            attack = selected[selected.condition == "attack"]
            n_text, m_text = stratum.name.split("_")
            n, m = int(n_text[1:]), int(m_text[1:])
            graph_rows = []
            for graph_id, graph_clean in clean.groupby("graph_id"):
                graph_attack = attack[attack.graph_id == graph_id]
                graph_rows.append(
                    {
                        "source": source,
                        "stratum": stratum.name,
                        "n": n,
                        "m": m,
                        "normalized_density": m / (n - 1) ** 2,
                        "graph_id": graph_id,
                        "task_count": len(graph_clean.task_id.unique()),
                        "u0": float(graph_clean.readout_round_zero_correct.mean()),
                        "u_t": float(graph_clean.final_correct.mean()),
                        "delta_u": float(
                            graph_clean.final_correct.mean()
                            - graph_clean.readout_round_zero_correct.mean()
                        ),
                        "r_mean": float(graph_attack.final_correct.mean()),
                    }
                )
            rows.extend(graph_rows)
            seen_strata.add(stratum.name)
            audit.append(
                {
                    "stratum": stratum.name,
                    "source": source,
                    "status": "included",
                    "matched_tasks": len(found),
                    "graphs": len(graph_rows),
                }
            )
    return pd.DataFrame(rows), audit


def plot_curves(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, axis, title in (
        ("u_t", axes[0, 0], "Clean utility $U_T$"),
        ("r_mean", axes[0, 1], "Mean robustness $R$"),
        ("u0", axes[1, 0], "Round-0 utility $U_0$"),
        ("delta_u", axes[1, 1], "Communication effect $\\Delta U$"),
    ):
        for n, frame in summary.groupby("n"):
            frame = frame.sort_values("m")
            means = frame[f"{column}_mean"].to_numpy(float)
            sd = frame[f"{column}_sd_across_graphs"].fillna(0).to_numpy(float)
            axis.plot(frame.m, means, marker="o", label=f"n={n}")
            axis.fill_between(frame.m, means - sd, means + sd, alpha=0.15)
        axis.set(title=title, xlabel="edge count m", ylabel="accuracy")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_scatter(graphs: pd.DataFrame, output: Path) -> None:
    ns = sorted(graphs.node_count.unique())
    figure, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5), squeeze=False)
    for axis, n in zip(axes[0], ns, strict=True):
        frame = graphs[graphs.node_count == n]
        scatter = axis.scatter(
            frame.u_t,
            frame.r_mean,
            c=frame.edge_count,
            cmap="viridis",
            s=55,
            alpha=0.85,
        )
        axis.set(
            title=f"n={n}: topology-level Utility–Robustness",
            xlabel="$U_T(G)$",
            ylabel="$R_{mean}(G)$",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        figure.colorbar(scatter, ax=axis, label="edge count m")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graphs, runs = load_pilot(args.pilot_root)
    summary = summarize_m(graphs)
    task_ids = set(runs.task_id.unique())
    reference, reference_audit = old_reference(
        task_ids=task_ids,
        old500_root=args.old500_root,
        revised_root=args.old500_revised_root,
    )
    graphs.to_csv(args.output_dir / "topology_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "m_response_summary.csv", index=False)
    reference.to_csv(args.output_dir / "old500_fixed50_reference.csv", index=False)
    pd.DataFrame(reference_audit).to_csv(
        args.output_dir / "old500_reference_audit.csv", index=False
    )
    plot_curves(summary, args.output_dir / "m_response_curves.png")
    plot_scatter(graphs, args.output_dir / "utility_robustness_scatter.png")
    manifest = {
        "analysis_version": "qwen-dense-pilot-v1",
        "pilot_root": str(args.pilot_root.resolve()),
        "tasks": len(task_ids),
        "topologies": len(graphs),
        "strata": len(summary),
        "metrics": {
            "U0": "clean readout Round-0 accuracy",
            "UT": "clean final readout accuracy; primary utility U(G)",
            "deltaU": "UT - U0",
            "R": "existing r_mean: mean attack-condition final accuracy over positions",
        },
        "uncertainty": (
            "m-level spread is empirical SD/min/max across the sampled graphs; complete "
            "strata contain one unique topology and therefore have no graph SD"
        ),
        "reference_warning": (
            "historical references use Llama-3.1-8B, temperature 0.3, shared Round-zero, "
            "and state replay; they are a task-matched reference, not a controlled model effect"
        ),
        "claim_status": "descriptive pilot only; trend claims require inspection of outputs",
        "reference_audit": reference_audit,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
