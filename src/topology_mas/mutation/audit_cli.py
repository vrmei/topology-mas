"""Audit and index a completed target-error mutation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.mutation.audit import (
    audit_mutation_cache,
    write_mutation_cache_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.mutation_dir / "selection-index")
    audit, answers = audit_mutation_cache(args.mutation_dir)
    write_mutation_cache_index(output_dir, audit, answers)
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
