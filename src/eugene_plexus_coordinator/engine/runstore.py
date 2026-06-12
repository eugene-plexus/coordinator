"""Persisted store for `PipelineRun` records.

Pipeline runs are the engine's durable artifact: a run can take minutes to
hours (training dominates), so it must survive a coordinator restart for the
UI to show history. Each run is one JSON file under
``<projectStorePath>/pipeline-runs/<id>.json``; the store keeps an in-memory
index for fast reads and rewrites the file on every transition.

Thread-safety: the worker thread driving a run mutates *its own* `PipelineRun`
object and calls `replace()`; the store deep-copies on write, so concurrent
readers (route handlers, SSE) always get a stable snapshot and never observe a
half-mutated run.

Interrupted-run recovery: any run found in a non-terminal state at startup was
cut off by a crash/restart — the worker that owned it is gone. `recover()`
marks those `failed` so they don't masquerade as still-running.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .._generated.models import PipelineRun, PipelineStatus

log = logging.getLogger(__name__)

_TERMINAL = frozenset(
    {
        PipelineStatus.completed,
        PipelineStatus.failed,
        PipelineStatus.cancelled,
    }
)


class PipelineRunStore:
    """Disk-backed, thread-safe CRUD over PipelineRuns, keyed by pipelineRunId."""

    def __init__(self, root: Path) -> None:
        # `root` is the project store path; runs live in a subdirectory so
        # they never collide with future project persistence.
        self._dir = root / "pipeline-runs"
        self._lock = threading.Lock()
        self._runs: dict[UUID, PipelineRun] = {}

    # ----------------------------------------------------------------- #
    # lifecycle
    # ----------------------------------------------------------------- #

    def load(self) -> None:
        """Load every persisted run into the in-memory index.

        Missing directory is fine (first boot). A single corrupt file is
        skipped with a warning rather than crashing the whole component —
        per the degraded-mode contract, one bad record must not soft-brick
        the engine.
        """
        with self._lock:
            self._runs.clear()
            if not self._dir.exists():
                return
            for path in sorted(self._dir.glob("*.json")):
                try:
                    run = PipelineRun.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception as e:  # tolerate any corrupt record
                    log.warning("skipping unreadable pipeline-run file %s: %s", path, e)
                    continue
                self._runs[run.pipelineRunId] = run

    def recover(self) -> list[UUID]:
        """Fail any run left non-terminal by a crash. Returns their ids."""
        recovered: list[UUID] = []
        with self._lock:
            for run_id, run in self._runs.items():
                if run.status in _TERMINAL:
                    continue
                run.status = PipelineStatus.failed
                run.currentStage = None
                run.finishedAt = datetime.now(UTC)
                run.lastError = "interrupted by a coordinator restart"
                for stage in run.stages:
                    if stage.status in (PipelineStatus.running, PipelineStatus.paused):
                        stage.status = PipelineStatus.failed
                        stage.detail = "interrupted by a coordinator restart"
                self._write_locked(run)
                recovered.append(run_id)
        if recovered:
            log.warning("recovered %d interrupted pipeline run(s) as failed", len(recovered))
        return recovered

    # ----------------------------------------------------------------- #
    # CRUD
    # ----------------------------------------------------------------- #

    def add(self, run: PipelineRun) -> None:
        with self._lock:
            self._runs[run.pipelineRunId] = run.model_copy(deep=True)
            self._write_locked(run)

    def replace(self, run: PipelineRun) -> None:
        """Persist the latest state of a run (overwrites in-memory + disk)."""
        with self._lock:
            self._runs[run.pipelineRunId] = run.model_copy(deep=True)
            self._write_locked(run)

    def get(self, run_id: UUID) -> PipelineRun | None:
        with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run is not None else None

    def list_for_project(self, project_id: UUID) -> list[PipelineRun]:
        """Runs for a project, newest first (by createdAt)."""
        with self._lock:
            runs = [r for r in self._runs.values() if r.projectId == project_id]
        runs.sort(key=lambda r: r.createdAt, reverse=True)
        return [r.model_copy(deep=True) for r in runs]

    def active_run_for_project(self, project_id: UUID) -> PipelineRun | None:
        """The project's non-terminal run, if one exists (the active-run lock)."""
        with self._lock:
            for run in self._runs.values():
                if run.projectId == project_id and run.status not in _TERMINAL:
                    return run.model_copy(deep=True)
        return None

    def cancel_if_active(self, run_id: UUID) -> PipelineRun | None:
        """Compare-and-set finalize a run to ``cancelled`` IFF still non-terminal.

        Terminal state is sticky: this never overwrites a run a worker already
        drove to completed/failed (closes the cancel-vs-finalize race). Returns
        the resulting run snapshot, or None if it doesn't exist.
        """
        return self._terminal_cas(
            run_id,
            status=PipelineStatus.cancelled,
            last_error="cancelled by operator",
            stage_detail="not run: pipeline cancelled",
        )

    def fail_if_active(self, run_id: UUID, error: str) -> PipelineRun | None:
        """Compare-and-set finalize a run to ``failed`` IFF still non-terminal.

        Used as the engine's worker-crash safety net so a run can never be
        stranded non-terminal (which would block the project's active-run slot).
        """
        return self._terminal_cas(
            run_id,
            status=PipelineStatus.failed,
            last_error=error,
            stage_detail=error,
        )

    def _terminal_cas(
        self, run_id: UUID, *, status: PipelineStatus, last_error: str, stage_detail: str
    ) -> PipelineRun | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if run.status in _TERMINAL:
                return run.model_copy(deep=True)
            run.status = status
            run.currentStage = None
            run.finishedAt = datetime.now(UTC)
            run.lastError = last_error
            for stage in run.stages:
                if stage.status in (PipelineStatus.running, PipelineStatus.pending):
                    stage.status = status
                    stage.detail = stage_detail
            self._write_locked(run)
            return run.model_copy(deep=True)

    # ----------------------------------------------------------------- #
    # internals
    # ----------------------------------------------------------------- #

    def _write_locked(self, run: PipelineRun) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{run.pipelineRunId}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(run.model_dump_json(exclude_none=True), encoding="utf-8")
        tmp.replace(path)
