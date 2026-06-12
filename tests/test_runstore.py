"""Unit tests for the persisted pipeline-run store.

Covers round-trip persistence, interrupted-run recovery, and corrupt-file
tolerance — the durability guarantees the engine relies on across restarts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from eugene_plexus_coordinator._generated.common_models import ComponentKind
from eugene_plexus_coordinator._generated.models import (
    Kind4,
    PipelineRun,
    PipelineStage,
    PipelineStatus,
)
from eugene_plexus_coordinator.engine.runstore import PipelineRunStore


def _run(project_id, status: PipelineStatus, *, stage_status: PipelineStatus) -> PipelineRun:
    return PipelineRun(
        pipelineRunId=uuid4(),
        projectId=project_id,
        status=status,
        createdAt=datetime.now(UTC),
        stages=[
            PipelineStage(
                kind=Kind4.training,
                component=ComponentKind.trainer,
                status=stage_status,
            )
        ],
    )


def test_add_and_get_round_trips(tmp_path: Path) -> None:
    store = PipelineRunStore(tmp_path)
    project_id = uuid4()
    run = _run(project_id, PipelineStatus.completed, stage_status=PipelineStatus.completed)
    store.add(run)

    got = store.get(run.pipelineRunId)
    assert got is not None
    assert got.status == PipelineStatus.completed
    assert got.projectId == project_id


def test_persisted_runs_survive_reload(tmp_path: Path) -> None:
    store = PipelineRunStore(tmp_path)
    project_id = uuid4()
    run = _run(project_id, PipelineStatus.completed, stage_status=PipelineStatus.completed)
    store.add(run)

    # A fresh store over the same directory sees the run.
    reloaded = PipelineRunStore(tmp_path)
    reloaded.load()
    assert [r.pipelineRunId for r in reloaded.list_for_project(project_id)] == [run.pipelineRunId]


def test_recover_fails_interrupted_runs(tmp_path: Path) -> None:
    store = PipelineRunStore(tmp_path)
    running = _run(uuid4(), PipelineStatus.running, stage_status=PipelineStatus.running)
    done = _run(uuid4(), PipelineStatus.completed, stage_status=PipelineStatus.completed)
    store.add(running)
    store.add(done)

    fresh = PipelineRunStore(tmp_path)
    fresh.load()
    recovered = fresh.recover()

    assert running.pipelineRunId in recovered
    assert done.pipelineRunId not in recovered
    got = fresh.get(running.pipelineRunId)
    assert got is not None
    assert got.status == PipelineStatus.failed
    assert got.stages[0].status == PipelineStatus.failed
    assert "interrupted" in (got.lastError or "")


def test_active_run_lookup(tmp_path: Path) -> None:
    store = PipelineRunStore(tmp_path)
    project_id = uuid4()
    store.add(_run(project_id, PipelineStatus.completed, stage_status=PipelineStatus.completed))
    assert store.active_run_for_project(project_id) is None

    active = _run(project_id, PipelineStatus.running, stage_status=PipelineStatus.running)
    store.add(active)
    found = store.active_run_for_project(project_id)
    assert found is not None
    assert found.pipelineRunId == active.pipelineRunId


def test_corrupt_file_is_skipped(tmp_path: Path) -> None:
    store = PipelineRunStore(tmp_path)
    good = _run(uuid4(), PipelineStatus.completed, stage_status=PipelineStatus.completed)
    store.add(good)
    # Drop a garbage file alongside the good one.
    (tmp_path / "pipeline-runs" / "garbage.json").write_text("{not json", encoding="utf-8")

    fresh = PipelineRunStore(tmp_path)
    fresh.load()  # must not raise
    assert fresh.get(good.pipelineRunId) is not None
