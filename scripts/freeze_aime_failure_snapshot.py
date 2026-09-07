#!/usr/bin/env python3
"""Freeze unresolved failure specs from a resumable batch output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    traces = {path.stem for path in (args.batch_dir / "traces").glob("*.json")}
    rows = []
    for path in sorted((args.batch_dir / "failures").glob("*.json")):
        if path.stem in traces:
            continue
        failure = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run_spec": failure["run_spec"],
            "exception_type": failure.get("exception_type"),
            "message": failure.get("message"),
            "failure_artifact": str(path.resolve()),
        })

    payload = "".join(canonical(row) + "\n" for row in rows)
    (args.out_dir / "unresolved_failures.jsonl").write_text(payload, encoding="utf-8")
    run_ids = [row["run_spec"]["run_spec_id"] for row in rows]
    (args.out_dir / "run_spec_ids.txt").write_text("\n".join(run_ids) + "\n", encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_batch_dir": str(args.batch_dir.resolve()),
        "trace_count_at_snapshot": len(traces),
        "unresolved_failure_count": len(rows),
        "exception_counts": dict(Counter(row["exception_type"] for row in rows)),
        "run_spec_ids_sha256": hashlib.sha256(("\n".join(run_ids) + "\n").encode()).hexdigest(),
    }
    (args.out_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
