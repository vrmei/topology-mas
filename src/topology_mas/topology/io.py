"""Artifact storage for sampled graph collections."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path

from topology_mas.models import GraphSpec
from topology_mas.topology.sampling import SAMPLER_VERSION
from topology_mas.topology.schemas import SampledGraphCollection


class GraphArtifactConflictError(RuntimeError):
    pass


def read_graphs_jsonl(path: str | Path) -> tuple[GraphSpec, ...]:
    graphs: list[GraphSpec] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                graph = GraphSpec.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid GraphSpec at line {line_number}") from exc
            if graph.graph_id in seen_ids:
                raise ValueError(f"duplicate graph_id {graph.graph_id!r}")
            seen_ids.add(graph.graph_id)
            graphs.append(graph)
    return tuple(graphs)


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _graphs_jsonl(graphs: Iterable[GraphSpec]) -> str:
    return "".join(graph.model_dump_json() + "\n" for graph in graphs)


def write_graph_collection(
    output_dir: str | Path,
    collection: SampledGraphCollection,
) -> tuple[Path, Path]:
    """Write once, or confirm an existing directory has the identical identity."""

    destination = Path(output_dir)
    graphs_path = destination / "graphs.jsonl"
    manifest_path = destination / "manifest.json"
    manifest = {
        "schema_version": 1,
        "sampler_version": SAMPLER_VERSION,
        "config": collection.config.model_dump(mode="json"),
        "summary": collection.summary.model_dump(mode="json"),
        "collection_fingerprint": collection.collection_fingerprint,
        "graph_ids": [graph.graph_id for graph in collection.graphs],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise GraphArtifactConflictError(
                "existing graph manifest differs; use a new output directory"
            )
        if not graphs_path.exists() or graphs_path.read_text(
            encoding="utf-8"
        ) != _graphs_jsonl(collection.graphs):
            raise GraphArtifactConflictError(
                "graph manifest matches but graphs.jsonl is missing or differs"
            )
        return graphs_path, manifest_path

    destination.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(graphs_path, _graphs_jsonl(collection.graphs))
    _write_text_atomically(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return graphs_path, manifest_path
