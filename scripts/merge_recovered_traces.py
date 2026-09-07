#!/usr/bin/env python3
"""Validate and atomically merge frozen technical-recovery traces.

The recovery directory is retained. Existing destination traces must either be
absent or byte-identical; a conflicting successful trace is never overwritten.
Historical failure artifacts are retained for audit, because trace presence is
the authoritative success signal used by the resumable runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-ids", type=Path, required=True)
    parser.add_argument("--recovery-output", type=Path, required=True)
    parser.add_argument("--destination-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_ids = [line.strip() for line in args.frozen_ids.read_text().splitlines() if line.strip()]
    if len(frozen_ids) != len(set(frozen_ids)):
        raise SystemExit("Frozen ID list contains duplicates")

    src_dir = args.recovery_output / "traces"
    dst_dir = args.destination_output / "traces"
    dst_dir.mkdir(parents=True, exist_ok=True)

    recovered = {path.stem: path for path in src_dir.glob("*.json")}
    missing = sorted(set(frozen_ids) - set(recovered))
    unexpected = sorted(set(recovered) - set(frozen_ids))
    if missing or unexpected:
        raise SystemExit(f"Recovery set mismatch: missing={missing}, unexpected={unexpected}")

    entries = []
    for run_spec_id in frozen_ids:
        src = recovered[run_spec_id]
        payload = json.loads(src.read_text(encoding="utf-8"))
        embedded_id = payload.get("run_spec", {}).get("run_spec_id")
        if embedded_id != run_spec_id:
            raise SystemExit(f"Embedded run_spec_id mismatch in {src}: {embedded_id}")

        src_hash = sha256(src)
        dst = dst_dir / src.name
        action = "copied"
        if dst.exists():
            if sha256(dst) != src_hash:
                raise SystemExit(f"Refusing to overwrite conflicting successful trace: {dst}")
            action = "already_identical"
        else:
            fd, temporary_name = tempfile.mkstemp(prefix=f".{src.name}.", dir=dst_dir)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(src, temporary)
                if sha256(temporary) != src_hash:
                    raise RuntimeError(f"Hash changed while copying {src}")
                os.replace(temporary, dst)
            finally:
                temporary.unlink(missing_ok=True)

        entries.append(
            {
                "run_spec_id": run_spec_id,
                "source": str(src),
                "destination": str(dst),
                "sha256": src_hash,
                "action": action,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_ids": str(args.frozen_ids),
        "recovery_output": str(args.recovery_output),
        "destination_output": str(args.destination_output),
        "count": len(entries),
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"merged": len(entries), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
