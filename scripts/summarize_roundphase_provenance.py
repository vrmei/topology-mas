#!/usr/bin/env python3
"""Summarize direct-versus-relayed target adoption by receiver scope and n."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_ctou_provenance import _cluster_bootstrap, _point_effect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--minimum-cell-group-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path, compression="infer") for path in args.updates]
    updates = pd.concat(frames, ignore_index=True)
    key = ["task_id", "graph_id", "attack_node", "receiver_node", "round_index"]
    if updates.duplicated(key).any():
        raise ValueError("duplicate provenance update keys across inputs")
    rows = []
    cells = []
    for n in sorted(updates.n.unique()):
        by_size = updates[updates.n.eq(n)]
        for scope in ("all", "internal", "readout"):
            selected = by_size if scope == "all" else by_size[by_size.receiver_scope.eq(scope)]
            point, cell_frame = _point_effect(
                selected,
                group_column="target_origin",
                group_a="direct_only",
                group_b="relayed_only",
                outcome_column="next_is_target",
                minimum_cell_group_rows=args.minimum_cell_group_rows,
            )
            if not point["matched_cells"]:
                continue
            task_lo, task_hi, task_success = _cluster_bootstrap(
                selected,
                selected_cells=cell_frame,
                group_column="target_origin",
                group_a="direct_only",
                group_b="relayed_only",
                outcome_column="next_is_target",
                replicates=args.bootstrap_reps,
                seed=args.seed + int(n) * 100 + (0 if scope == "all" else 1 if scope == "internal" else 2),
            )
            row = {
                "n": int(n),
                "scope": scope,
                "comparison": "direct_only_minus_relayed_only",
                **point,
                "task_graph_cluster_ci95_low": task_lo,
                "task_graph_cluster_ci95_high": task_hi,
                "successful_bootstraps": task_success,
            }
            rows.append(row)
            cell_frame = cell_frame.assign(n=int(n), scope=scope)
            cells.append(cell_frame)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "direct_vs_relayed_by_size.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(
        args.out / "direct_vs_relayed_matched_cells.csv", index=False
    )
    updates.to_parquet(args.out / "provenance_updates_all_sizes.parquet", index=False)
    lines = [
        "# Target provenance diagnostic", "",
        "Direct and relayed target messages are compared within the same previous state, round, incoming CTOU counts, and receiver scope.", "",
        "| n | scope | P(next=T), direct | P(next=T), relayed | difference (pp) | 95% CI (pp) | matched rows |",
        "|---:|:---|---:|---:|---:|:---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.n} | {row.scope} | {100*row.group_a_probability:.2f} | "
            f"{100*row.group_b_probability:.2f} | {100*row.risk_difference:+.2f} | "
            f"[{100*row.task_graph_cluster_ci95_low:+.2f}, {100*row.task_graph_cluster_ci95_high:+.2f}] | "
            f"{row.matched_rows} |"
        )
    lines += ["", "This is an observational matched comparison. Because the receiver is not shown a trusted attacker label, 'origin' mainly indexes how the target rationale was generated and transformed, not an independently randomized source identity."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "roundphase-provenance-v1",
        "inputs": [str(path) for path in args.updates],
        "updates": len(updates),
        "tasks": int(updates.task_id.nunique()),
        "graphs": int(updates.graph_id.nunique()),
        "bootstrap_reps": args.bootstrap_reps,
        "minimum_cell_group_rows": args.minimum_cell_group_rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
