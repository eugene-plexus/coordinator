"""The pipeline-execution engine.

`PipelineEngine` is the coordinator's control surface over pipeline runs. It
owns the persisted run store, a thread pool that executes runs (one
`PipelineRunner` per run on a worker thread), and the per-run cancel events.
It enforces the platform's **one active pipeline run per project** policy and
resolves peer base URLs live from the runtime config so a re-pointed peer
takes effect on the next run without code changes (GUI-equality).

Build-failure containment: constructing the engine never touches the network
and only reads the run-store directory. If anything here raised, the app
lifespan would catch it and degrade (engine=None) rather than crash — per the
degraded-mode contract.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from .._generated.models import (
    PipelineRun,
    PipelineStatus,
    TrainingProject,
)
from ..config import ConfigStore
from .errors import BadRequestError, ConflictError, NotFoundError
from .peer_client import PeerClient, PeerClientProtocol
from .planner import ConfiguredPeers, has_runnable_stage, plan_stages
from .runner import PipelineRunner
from .runstore import PipelineRunStore

log = logging.getLogger(__name__)

_TERMINAL = frozenset({PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.cancelled})

# Production defaults. Tests inject much smaller values so suites stay fast.
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_STAGE_TIMEOUT_S = 24 * 3600.0  # a training stage can run for hours
DEFAULT_MAX_WORKERS = 4


class PeerProbe:
    """Outcome of probing one peer's ``/healthz``."""

    def __init__(self, *, label: str, configured: bool, reachable: bool, detail: str) -> None:
        self.label = label
        self.configured = configured
        self.reachable = reachable
        self.detail = detail


class PipelineEngine:
    def __init__(
        self,
        config_store: ConfigStore,
        *,
        client: PeerClientProtocol | None = None,
        service_token: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self._config = config_store
        self._client: PeerClientProtocol = client or PeerClient(service_token=service_token)
        self._poll_interval = poll_interval
        self._stage_timeout = stage_timeout

        store_root = Path(str(config_store.get("projectStorePath") or "coordinator-store"))
        self._store = PipelineRunStore(store_root)

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline")
        self._lock = threading.Lock()
        self._cancel_events: dict[UUID, threading.Event] = {}

    # ----------------------------------------------------------------- #
    # lifecycle
    # ----------------------------------------------------------------- #

    def recover(self) -> None:
        """Load persisted runs and fail any left mid-flight by a restart."""
        self._store.load()
        self._store.recover()

    def shutdown(self) -> None:
        # Signal every live worker to stop FIRST so it breaks out of its poll
        # loop at the next check (rather than running to a multi-hour stage
        # timeout — these are non-daemon threads that would otherwise block
        # interpreter exit).
        with self._lock:
            events = list(self._cancel_events.values())
        for event in events:
            event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._client.close()
        except Exception:  # best-effort on teardown
            log.exception("error closing peer client")

    # ----------------------------------------------------------------- #
    # commands
    # ----------------------------------------------------------------- #

    def start(self, project: TrainingProject) -> PipelineRun:
        """Plan + launch a pipeline run for a project.

        Raises `ConflictError` if a run is already active for the project and
        `BadRequestError` if the project has no runnable stage.
        """
        peers = self._configured_peers()
        stages = plan_stages(project, peers)
        if not has_runnable_stage(stages):
            raise BadRequestError(
                "project has no runnable stages: check that the relevant peer "
                "components are configured and the project has datasets / a "
                "recipe / eval suites for its goal"
            )

        with self._lock:
            if self._store.active_run_for_project(project.projectId) is not None:
                raise ConflictError(
                    f"project {project.projectId} already has an active pipeline run"
                )
            run = PipelineRun(
                pipelineRunId=uuid4(),
                projectId=project.projectId,
                status=PipelineStatus.pending,
                createdAt=datetime.now(UTC),
                stages=stages,
            )
            self._store.add(run)
            cancel_event = threading.Event()
            self._cancel_events[run.pipelineRunId] = cancel_event
            # Snapshot the just-created (pending) run to return BEFORE the
            # worker can mutate it, so the 201 body is deterministically the
            # pending run rather than whatever state the worker raced to.
            snapshot = self._store.get(run.pipelineRunId)
            assert snapshot is not None  # just added

        runner = PipelineRunner(
            run=run,
            project=project,
            peers=peers,
            client=self._client,
            store=self._store,
            cancel_event=cancel_event,
            poll_interval=self._poll_interval,
            stage_timeout=self._stage_timeout,
        )
        self._executor.submit(self._run_worker, run.pipelineRunId, runner)
        return snapshot

    def cancel(self, run_id: UUID) -> PipelineRun:
        """Request cancellation of a run. Raises NotFound/Conflict.

        Atomic under ``self._lock`` so the check-then-act over the cancel-event
        registry can't race a worker finalizing the run. If a worker is live we
        just set its event (it finalizes to cancelled); otherwise we
        compare-and-set the store to cancelled — which is a no-op if the run is
        already terminal, so a late cancel can never clobber a completed/failed
        run.
        """
        with self._lock:
            run = self._store.get(run_id)
            if run is None:
                raise NotFoundError(f"no pipeline run with id {run_id}")
            if run.status in _TERMINAL:
                raise ConflictError(f"pipeline run {run_id} is already {run.status.value}")

            event = self._cancel_events.get(run_id)
            if event is not None:
                event.set()  # the worker observes this and finalizes to cancelled
            else:
                # No live worker — finalize directly, but only if still
                # non-terminal (compare-and-set; never overwrite a terminal run).
                self._store.cancel_if_active(run_id)

        refreshed = self._store.get(run_id)
        assert refreshed is not None
        return refreshed

    # ----------------------------------------------------------------- #
    # queries
    # ----------------------------------------------------------------- #

    def get(self, run_id: UUID) -> PipelineRun | None:
        return self._store.get(run_id)

    def list_for_project(self, project_id: UUID) -> list[PipelineRun]:
        return self._store.list_for_project(project_id)

    def has_active_run(self, project_id: UUID) -> bool:
        return self._store.active_run_for_project(project_id) is not None

    def probe_peers(self) -> list[PeerProbe]:
        """Probe each configured peer's /healthz (for /v1/config/test)."""
        peers = self._configured_peers()
        targets = (
            ("trainer", peers.trainer),
            ("data", peers.data),
            ("eval", peers.eval),
            ("inference", peers.inference),
        )
        probes: list[PeerProbe] = []
        for label, url in targets:
            if not url:
                probes.append(
                    PeerProbe(
                        label=label, configured=False, reachable=False, detail="not configured"
                    )
                )
                continue
            try:
                body = self._client.healthz(url)
                status = str(body.get("status", "ok"))
                probes.append(
                    PeerProbe(label=label, configured=True, reachable=True, detail=status)
                )
            except Exception as e:  # any failure is "unreachable"
                probes.append(
                    PeerProbe(label=label, configured=True, reachable=False, detail=str(e))
                )
        return probes

    # ----------------------------------------------------------------- #
    # internals
    # ----------------------------------------------------------------- #

    def _run_worker(self, run_id: UUID, runner: PipelineRunner) -> None:
        try:
            runner.run()
        except Exception:  # safety net: a run must never be stranded non-terminal
            log.exception("pipeline worker for %s crashed", run_id)
            self._store.fail_if_active(run_id, "internal worker error")
        finally:
            self._cancel_events.pop(run_id, None)

    def _configured_peers(self) -> ConfiguredPeers:
        get = self._config.get
        return ConfiguredPeers(
            data=str(get("dataUrl") or ""),
            trainer=str(get("trainerUrl") or ""),
            eval=str(get("evalUrl") or ""),
            inference=str(get("inferenceUrl") or ""),
        )
