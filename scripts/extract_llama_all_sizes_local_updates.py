#!/usr/bin/env python3
"""Extract paired clean/attack local updates for all Llama/GSM8K sizes.

This is a read-only normalization pass over the completed n=5,6,7,8 T=3
experiments.  It preserves each recorded round and does not invoke an LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_node_round_adoption import extract_updates, read_json


SHORT = {"correct": "C", "target": "T", "other": "O", "unparsed": "U"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n58-root", type=Path, required=True)
    parser.add_argument("--n67-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def normalize(frame: pd.DataFrame, node_count: int) -> pd.DataFrame:
    out = frame.copy()
    out["system"] = f"Llama-GSM8K-n{node_count}"
    out["run_key"] = (
        out.task_id.astype(str)
        + "|" + out.graph_id.astype(str)
        + "|" + out.attack_node.astype(str)
    )
    out["prev_state"] = out.previous_attack_state.map(SHORT)
    out["next_state"] = out.current_attack_state.map(SHORT)
    for long, short in SHORT.items():
        out[f"in_{short}"] = out[f"incoming_{long}_count"].astype(int)
    out["degree"] = out[["in_C", "in_T", "in_O", "in_U"]].sum(axis=1)
    out["target_share"] = np.where(out.degree > 0, out.in_T / out.degree, 0.0)
    out["receiver_scope"] = np.where(out.receiver_is_readout.eq(1), "readout", "internal")
    return out


def extract_size(root: Path, node_count: int) -> tuple[pd.DataFrame, dict]:
    status = read_json(root / "orchestrator_status.json")
    selected = [x for x in status["strata"] if str(x["key"]).startswith(f"n{node_count}_")]
    if not selected:
        raise RuntimeError(f"no n={node_count} strata found in {root}")
    frame, audit = extract_updates(root, {**status, "strata": selected})
    if frame.empty:
        raise RuntimeError(f"n={node_count} extraction returned no updates")
    if frame["n"].nunique() != 1 or int(frame["n"].iloc[0]) != node_count:
        raise RuntimeError(f"n={node_count} extraction contains inconsistent node counts")
    audit = {**audit, "n": node_count, "strata": len(selected), "root": str(root)}
    return normalize(frame, node_count), audit


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    audits = []
    for node_count, root in ((5, args.n58_root), (6, args.n67_root), (7, args.n67_root), (8, args.n58_root)):
        frame, audit = extract_size(root, node_count)
        frames.append(frame)
        audits.append(audit)
        print(json.dumps(audit, ensure_ascii=False))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(args.out, index=False)
    manifest = {
        "output": str(args.out),
        "rows": len(combined),
        "tasks": int(combined.task_id.nunique()),
        "graphs": int(combined.graph_id.nunique()),
        "attack_cells": int(combined.run_key.nunique()),
        "sizes": {
            str(int(n)): {
                "rows": len(group),
                "graphs": int(group.graph_id.nunique()),
                "attack_cells": int(group.run_key.nunique()),
                "m_levels": sorted(int(x) for x in group.m.unique()),
            }
            for n, group in combined.groupby("n")
        },
        "audits": audits,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
