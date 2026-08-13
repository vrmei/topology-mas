from pathlib import Path

from scripts.run_qwen_dense_pilot import run_stage


def _status() -> dict[str, object]:
    return {"status": "running", "stages": [], "current_stage": None}


def test_run_stage_restarts_and_resumes_atomic_outputs(tmp_path: Path) -> None:
    marker = tmp_path / "attempts.txt"
    batch = tmp_path / "batch"
    traces = batch / "traces"
    command = [
        "python",
        "-c",
        (
            "from pathlib import Path; "
            f"p=Path({str(marker)!r}); "
            "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
            f"t=Path({str(traces)!r}); t.mkdir(parents=True, exist_ok=True); "
            "(t/f'{n}.json').write_text('{}'); "
            "raise SystemExit(1 if n == 1 else 0)"
        ),
    ]
    status = _status()

    succeeded = run_stage(
        name="batch_test",
        command=command,
        project_root=tmp_path,
        run_root=tmp_path / "run",
        batch_root=batch,
        expected=2,
        poll_seconds=0,
        status=status,
        max_restarts=1,
    )

    assert succeeded
    assert marker.read_text() == "2"
    assert [stage["status"] for stage in status["stages"]] == [
        "failed",
        "completed",
    ]


def test_run_stage_returns_false_after_restart_budget(tmp_path: Path) -> None:
    status = _status()
    succeeded = run_stage(
        name="batch_test",
        command=["python", "-c", "raise SystemExit(1)"],
        project_root=tmp_path,
        run_root=tmp_path / "run",
        batch_root=tmp_path / "batch",
        expected=1,
        poll_seconds=0,
        status=status,
        max_restarts=1,
    )

    assert not succeeded
    assert len(status["stages"]) == 2
    assert all(stage["status"] == "failed" for stage in status["stages"])
