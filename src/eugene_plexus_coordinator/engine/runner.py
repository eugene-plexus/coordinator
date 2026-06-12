"""The per-run stage sequencer.

A `PipelineRunner` owns the execution of exactly one `PipelineRun` on a worker
thread. It walks the planned stages in order, delegating each to its peer
component over HTTP and polling that peer for completion, threading the
checkpoint produced by the training stage forward into eval and serve. Every
state transition is persisted immediately so a poller (`GET .../pipeline-runs/{id}`)
or the SSE stream always sees fresh status.

Cancellation is cooperative: `PipelineEngine.cancel()` sets a `threading.Event`;
the runner observes it between stages and inside every poll loop, cancels the
active peer resource (e.g. the trainer run), and finalizes the pipeline as
`cancelled`.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from uuid import uuid4

from .._generated.models import (
    HardwareTopology,
    Kind4,
    Mode1,
    PipelineRun,
    PipelineStage,
    PipelineStatus,
    TrainingProject,
    TrainingRunRequest,
)
from .errors import PeerError
from .peer_client import PeerClientProtocol
from .planner import ConfiguredPeers
from .runstore import PipelineRunStore

log = logging.getLogger(__name__)

# Peer run statuses we treat as terminal when polling.
_TRAINER_TERMINAL = {"failed", "cancelled"}
_EVAL_TERMINAL = {"completed", "failed"}

# Terminal pipeline-run statuses (used to avoid double-finalizing a run).
_PIPELINE_TERMINAL = frozenset(
    {PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.cancelled}
)


class _StageFailed(Exception):
    """A stage could not complete; carries the user-facing detail."""


class _StageCancelled(Exception):
    """The run was cancelled while this stage was executing."""


def _now() -> datetime:
    return datetime.now(UTC)


class PipelineRunner:
    def __init__(
        self,
        *,
        run: PipelineRun,
        project: TrainingProject,
        peers: ConfiguredPeers,
        client: PeerClientProtocol,
        store: PipelineRunStore,
        cancel_event: threading.Event,
        poll_interval: float,
        stage_timeout: float,
    ) -> None:
        self._run = run
        self._project = project
        self._peers = peers
        self._client = client
        self._store = store
        self._cancel = cancel_event
        self._poll_interval = poll_interval
        self._stage_timeout = stage_timeout

    # ----------------------------------------------------------------- #
    # entry point (runs on the worker thread)
    # ----------------------------------------------------------------- #

    def run(self) -> None:
        run = self._run
        checkpoint_id: str | None = None
        try:
            run.status = PipelineStatus.running
            run.startedAt = _now()
            self._persist()
            for index, stage in enumerate(run.stages):
                if self._cancel.is_set():
                    self._finalize_cancelled(from_index=index)
                    return
                if stage.status == PipelineStatus.skipped:
                    continue

                stage.status = PipelineStatus.running
                stage.startedAt = _now()
                run.currentStage = stage.kind.value
                self._persist()

                try:
                    produced = self._run_stage(stage, checkpoint_id)
                except _StageCancelled:
                    stage.status = PipelineStatus.cancelled
                    stage.finishedAt = _now()
                    self._finalize_cancelled(from_index=index + 1)
                    return
                except _StageFailed as e:
                    self._finalize_failed(stage, index, str(e))
                    return
                except PeerError as e:  # defensive: any unconverted peer error
                    self._finalize_failed(stage, index, str(e))
                    return

                if stage.kind == Kind4.training and produced is not None:
                    checkpoint_id = produced
                stage.status = PipelineStatus.completed
                stage.finishedAt = _now()
                self._persist()

            run.status = PipelineStatus.completed
            run.currentStage = None
            run.finishedAt = _now()
            self._persist()
        except Exception as e:  # never let a worker die silently
            log.exception("pipeline run %s crashed", run.pipelineRunId)
            # Only finalize if a stage handler didn't already set a terminal
            # state (e.g. a finalizer's persist raising) — never overwrite a
            # cancelled/failed run with a generic "internal error".
            if run.status not in _PIPELINE_TERMINAL:
                run.status = PipelineStatus.failed
                run.currentStage = None
                run.finishedAt = _now()
                run.lastError = f"internal error: {e}"
                self._persist()

    # ----------------------------------------------------------------- #
    # stage dispatch
    # ----------------------------------------------------------------- #

    def _run_stage(self, stage: PipelineStage, checkpoint_id: str | None) -> str | None:
        try:
            if stage.kind == Kind4.data_prep:
                self._do_data_prep(stage)
            elif stage.kind == Kind4.tokenizer:
                self._do_tokenizer(stage)
            elif stage.kind == Kind4.training:
                return self._do_training(stage)
            elif stage.kind == Kind4.eval:
                self._do_eval(stage, checkpoint_id)
            elif stage.kind == Kind4.serve:
                self._do_serve(stage, checkpoint_id)
        except PeerError as e:
            raise _StageFailed(str(e)) from e
        return None

    # ----------------------------------------------------------------- #
    # stages
    # ----------------------------------------------------------------- #

    def _do_data_prep(self, stage: PipelineStage) -> None:
        data_url = self._peers.data
        datasets = self._project.datasets or []
        names: list[str] = []
        primary: str | None = None
        for ref in datasets:
            ds_id = str(ref.datasetId)
            manifest = self._client.get_dataset(data_url, ds_id)
            status = manifest.get("status")
            if status in (None, "empty"):
                raise _StageFailed(f"dataset {ds_id} has no imported data (status={status})")
            if status == "error":
                raise _StageFailed(f"dataset {ds_id} is in an error state")
            primary = primary or ds_id
            names.append(str(manifest.get("name") or ds_id))
        stage.resourceId = primary
        stage.detail = f"verified {len(datasets)} dataset(s): {', '.join(names)}"

    def _do_tokenizer(self, stage: PipelineStage) -> None:
        data_url = self._peers.data
        tokenizer = self._project.tokenizer
        if tokenizer is None:  # planner gates this, but stay defensive
            raise _StageFailed("project has no tokenizer")
        tok_id = str(tokenizer.tokenizerId)

        arch = self._project.modelTemplate.architecture if self._project.modelTemplate else None
        if arch is None:
            raise _StageFailed(
                "project has no modelTemplate/architecture; blockSize is required to pretokenize"
            )
        block_size = arch.blockSize

        tokenizers = self._client.list_tokenizers(data_url)
        match = next((t for t in tokenizers if t.get("tokenizerId") == tok_id), None)
        if match is None:
            raise _StageFailed(f"tokenizer {tok_id} not found on the data component")
        tok_status = match.get("status")
        if tok_status not in (None, "ready"):
            raise _StageFailed(f"tokenizer {tok_id} is not ready (status={tok_status})")

        # One deadline for the whole stage, shared across all datasets, so a
        # multi-dataset stage is bounded by a single stage_timeout (not N of
        # them).
        deadline = time.monotonic() + self._stage_timeout
        for ref in self._project.datasets or []:
            ds_id = str(ref.datasetId)
            self._client.pretokenize_dataset(
                data_url, ds_id, tokenizer_id=tok_id, block_size=block_size
            )
            self._poll(
                lambda did=ds_id: self._check_dataset_ready(data_url, did),
                what=f"pretokenize dataset {ds_id}",
                deadline=deadline,
            )

        stage.resourceId = tok_id
        n = len(self._project.datasets or [])
        stage.detail = (
            f"pretokenized {n} dataset(s) with tokenizer "
            f"{match.get('name') or tok_id} @ blockSize {block_size}"
        )

    def _do_training(self, stage: PipelineStage) -> str:
        trainer_url = self._peers.trainer
        request = self._build_training_request()
        run = self._client.start_training_run(trainer_url, request)
        run_id = run.get("runId")
        if not run_id:
            raise _StageFailed("trainer did not return a runId")
        run_id = str(run_id)
        stage.resourceId = run_id
        stage.detail = f"training run {run_id}: {run.get('status', 'queued')}"
        self._persist()

        def poll_training() -> dict | None:
            snapshot = self._client.get_training_run(trainer_url, run_id)
            status = snapshot.get("status")
            step = snapshot.get("currentStep")
            stage.detail = f"training run {run_id}: {status}" + (
                f" (step {step})" if step is not None else ""
            )
            self._persist()
            if status == "completed":
                return snapshot
            if status in _TRAINER_TERMINAL:
                raise _StageFailed(
                    f"trainer run {status}: {snapshot.get('lastError') or 'no detail'}"
                )
            return None

        self._poll(
            poll_training,
            what=f"training run {run_id}",
            on_cancel=lambda: self._safe_cancel_training(trainer_url, run_id),
        )

        checkpoints = self._client.list_run_checkpoints(trainer_url, run_id)
        if not checkpoints:
            raise _StageFailed("training completed but produced no checkpoints")
        chosen = (
            next((c for c in checkpoints if c.get("isBest")), None)
            or next((c for c in checkpoints if c.get("isLatest")), None)
            or max(checkpoints, key=lambda c: c.get("step") or 0)
        )
        cp_id = chosen.get("checkpointId")
        if not cp_id:
            raise _StageFailed("trainer's selected checkpoint has no checkpointId")
        stage.detail = f"trained run {run_id}; checkpoint {cp_id} @ step {chosen.get('step')}"
        return str(cp_id)

    def _do_eval(self, stage: PipelineStage, checkpoint_id: str | None) -> None:
        eval_url = self._peers.eval
        cp = checkpoint_id or self._base_checkpoint_id()
        if cp is None:
            raise _StageFailed(
                "no checkpoint to evaluate (no training stage ran and "
                "recipe.baseCheckpoint is unset)"
            )
        suites = self._project.evalSuites or []
        summaries: list[str] = []
        primary: str | None = None
        deadline = time.monotonic() + self._stage_timeout
        for suite in suites:
            suite_id = str(suite.evalSuiteId)
            result = self._client.start_eval_run(eval_url, suite_id=suite_id, checkpoint_id=cp)
            eval_run_id = result.get("evalRunId")
            status = result.get("status")
            # Only "completed"/"failed" are terminal. Anything else (running,
            # or an out-of-spec value) must be polled to a terminal result —
            # and if we can't (no evalRunId), that's a hard failure, not a
            # silent success with null metrics.
            if status not in ("completed", "failed"):
                if not eval_run_id:
                    raise _StageFailed(
                        f"eval suite {suite_id} returned status={status} with no evalRunId to poll"
                    )
                result = self._poll(
                    lambda rid=str(eval_run_id): self._check_eval_done(eval_url, rid),
                    what=f"eval run {eval_run_id}",
                    deadline=deadline,
                )
            if result.get("status") == "failed":
                raise _StageFailed(f"eval suite {suite_id} failed")
            primary = primary or (str(eval_run_id) if eval_run_id else None)
            summaries.append(
                f"{suite.name or suite_id}: valLoss={result.get('valLoss')} "
                f"ppl={result.get('perplexity')}"
            )
        stage.resourceId = primary
        stage.detail = "; ".join(summaries) if summaries else "no suites evaluated"

    def _do_serve(self, stage: PipelineStage, checkpoint_id: str | None) -> None:
        inference_url = self._peers.inference
        cp = checkpoint_id or self._base_checkpoint_id()
        if cp is None:
            raise _StageFailed(
                "no checkpoint to serve (no training stage ran and recipe.baseCheckpoint is unset)"
            )
        endpoint_id = str(uuid4())
        # The OpenAI-compatible model id clients will pass as `model`. We use
        # the project name; exportSettings.target's semantics are ambiguous in
        # the v0.3-draft spec (documented as both an endpoint name and a peer
        # URL), so the peer URL is taken from config (inferenceUrl), not target.
        model_name = self._project.name
        self._client.create_endpoint(inference_url, {"endpointId": endpoint_id, "name": model_name})
        stage.resourceId = endpoint_id
        self._persist()

        endpoint = self._client.load_endpoint(inference_url, endpoint_id, checkpoint_id=cp)
        status = endpoint.get("status")
        # Poll while not yet terminal. Positive-gate on "ready": treat anything
        # that isn't "ready" after loading (error, or an unexpected/blank
        # status) as a failure rather than reporting a serve that never loaded.
        if status in ("loading", "unloaded"):
            endpoint = self._poll(
                lambda: self._check_endpoint_ready(inference_url, endpoint_id),
                what=f"load endpoint {endpoint_id}",
            )
            status = endpoint.get("status")
        if status != "ready":
            raise _StageFailed(
                f"inference endpoint {endpoint_id} did not reach 'ready' (status={status})"
            )
        stage.detail = f"serving checkpoint {cp} as model '{model_name}' (endpoint {endpoint_id})"

    # ----------------------------------------------------------------- #
    # poll predicates
    # ----------------------------------------------------------------- #

    def _check_dataset_ready(self, data_url: str, dataset_id: str) -> dict | None:
        manifest = self._client.get_dataset(data_url, dataset_id)
        status = manifest.get("status")
        if status == "ready":
            return manifest
        if status == "error":
            raise _StageFailed(f"dataset {dataset_id} entered an error state during pretokenize")
        return None

    def _check_eval_done(self, eval_url: str, eval_run_id: str) -> dict | None:
        result = self._client.get_eval_result(eval_url, eval_run_id)
        if result.get("status") in _EVAL_TERMINAL:
            return result
        return None

    def _check_endpoint_ready(self, inference_url: str, endpoint_id: str) -> dict | None:
        for endpoint in self._client.list_endpoints(inference_url):
            if endpoint.get("endpointId") == endpoint_id:
                status = endpoint.get("status")
                if status in ("ready", "error"):
                    return endpoint
                return None
        # Endpoint vanished from the listing — treat as a hard failure.
        raise _StageFailed(f"endpoint {endpoint_id} disappeared from the inference component")

    # ----------------------------------------------------------------- #
    # helpers
    # ----------------------------------------------------------------- #

    def _build_training_request(self) -> dict:
        project = self._project
        template = project.modelTemplate
        if template is None or template.architecture is None:
            raise _StageFailed("project has no modelTemplate/architecture")
        if project.recipe is None:
            raise _StageFailed("project has no training recipe")
        if project.hyperparameters is None:
            raise _StageFailed("project has no hyperparameters")
        if project.tokenizer is None:
            raise _StageFailed("project has no tokenizer")
        if not project.datasets:
            raise _StageFailed("project has no datasets")

        # Hardware is optional on a project; default to CPU so the pipeline is
        # runnable everywhere. Operators pick a GPU topology in the wizard.
        hardware = project.hardware or HardwareTopology(mode=Mode1.cpu)

        request = TrainingRunRequest(
            projectId=project.projectId,
            architecture=template.architecture,
            recipe=project.recipe,
            hyperparameters=project.hyperparameters,
            hardware=hardware,
            tokenizer=project.tokenizer,
            datasets=project.datasets,
        )
        return request.model_dump(mode="json", exclude_none=True)

    def _base_checkpoint_id(self) -> str | None:
        recipe = self._project.recipe
        if recipe and recipe.baseCheckpoint:
            return str(recipe.baseCheckpoint.checkpointId)
        return None

    def _safe_cancel_training(self, trainer_url: str, run_id: str) -> None:
        try:
            self._client.cancel_training_run(trainer_url, run_id)
        except PeerError as e:  # best-effort; we're cancelling anyway
            log.warning("failed to cancel trainer run %s: %s", run_id, e)

    def _poll(self, fn, *, what: str, on_cancel=None, deadline=None):  # type: ignore[no-untyped-def]
        """Call ``fn`` until it returns a non-None value.

        ``fn`` returns None to keep waiting or a value to stop, and may raise
        `_StageFailed`. Honors the cancel event (runs ``on_cancel`` then raises
        `_StageCancelled`) and a timeout. The wait between polls is
        interruptible — a cancel wakes it immediately.

        ``deadline`` (a ``time.monotonic()`` value) lets a multi-resource stage
        share one budget across several polls; when omitted a fresh
        ``stage_timeout`` window is used for this single poll.
        """
        if deadline is None:
            deadline = time.monotonic() + self._stage_timeout
        while True:
            if self._cancel.is_set():
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:  # cancel is best-effort
                        log.exception("on_cancel hook failed for %s", what)
                raise _StageCancelled()
            result = fn()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise _StageFailed(f"{what}: timed out after {self._stage_timeout:.0f}s")
            self._cancel.wait(self._poll_interval)

    # ----------------------------------------------------------------- #
    # finalizers
    # ----------------------------------------------------------------- #

    def _persist(self) -> None:
        self._store.replace(self._run)

    def _finalize_failed(self, stage: PipelineStage, index: int, detail: str) -> None:
        stage.status = PipelineStatus.failed
        stage.detail = detail
        stage.finishedAt = _now()
        self._run.status = PipelineStatus.failed
        self._run.lastError = f"{stage.kind.value}: {detail}"
        self._run.currentStage = None
        self._run.finishedAt = _now()
        self._cancel_remaining(
            from_index=index + 1, detail=f"not run: pipeline failed at {stage.kind.value}"
        )
        self._persist()

    def _finalize_cancelled(self, *, from_index: int) -> None:
        self._run.status = PipelineStatus.cancelled
        self._run.lastError = "cancelled by operator"
        self._run.currentStage = None
        self._run.finishedAt = _now()
        self._cancel_remaining(from_index=from_index, detail="not run: pipeline cancelled")
        self._persist()

    def _cancel_remaining(self, *, from_index: int, detail: str) -> None:
        for stage in self._run.stages[from_index:]:
            if stage.status == PipelineStatus.pending:
                stage.status = PipelineStatus.cancelled
                stage.detail = detail
