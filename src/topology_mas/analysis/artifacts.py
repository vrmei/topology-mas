"""Atomic, conflict-safe analysis artifact writer."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from topology_mas.analysis.schemas import AnalysisResult


class AnalysisArtifactConflictError(RuntimeError):
    pass


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _jsonl(values: tuple[BaseModel, ...]) -> str:
    return "".join(value.model_dump_json() + "\n" for value in values)


def _write_or_validate(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise AnalysisArtifactConflictError(
                f"analysis artifact differs at {path}; use a new output directory"
            )
        return
    _atomic_write(path, content)


def _graph_csv(result: AnalysisResult) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "graph_id",
        "node_count",
        "edge_count",
        "clean_samples",
        "paired_attack_samples",
        "utility",
        "readout_round_zero_accuracy",
        "communication_correction_rate",
        "communication_corruption_rate",
        "r_mean",
        "r_worst",
        "d_mean",
        "d_max",
        "node_attack_accuracy_std",
        "final_target_match_rate",
        "induced_readout_target_rate",
        "correct_to_target_flip_rate",
        "mean_max_induced_nonattacker_count",
        "clean_mean_model_calls",
        "attack_mean_model_calls",
        "clean_mean_input_tokens",
        "attack_mean_input_tokens",
        "clean_mean_output_tokens",
        "attack_mean_output_tokens",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for metric in result.graph_metrics:
        row = metric.model_dump(mode="json", exclude={"node_metrics"})
        writer.writerow({field: row[field] for field in fieldnames})
    return buffer.getvalue()


def write_analysis(output_dir: str | Path, result: AnalysisResult) -> None:
    destination = Path(output_dir)
    artifacts = {
        "manifest.json": json.dumps(
            result.manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "run_metrics.jsonl": _jsonl(result.run_metrics),
        "paired_attacks.jsonl": _jsonl(result.paired_attacks),
        "graph_metrics.json": json.dumps(
            [metric.model_dump(mode="json") for metric in result.graph_metrics],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "graph_metrics.csv": _graph_csv(result),
        "classical_initial_states.jsonl": _jsonl(result.classical_initial_states),
    }
    for name, content in artifacts.items():
        _write_or_validate(destination / name, content)
