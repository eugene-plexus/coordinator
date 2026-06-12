"""Server-Sent Events for live pipeline-run progress.

The coordinator streams progress by polling its own run store and emitting an
SSE event whenever a stage's status changes — a poll-and-diff design that
needs no cross-thread queue between the worker (which mutates the run) and the
async route (which serializes the stream), so it can't deadlock or leak a
subscriber. Named events mirror the spec: ``stage_started`` (a stage went
running), ``stage_status`` (any other status change), ``stage_completed`` (a
stage finished cleanly), and a terminal ``done`` / ``error`` carrying the full
final run.

(`metrics` events — forwarding the active training stage's TrainingMetricPoint —
are deferred to v0.3+; poll the trainer's own ``/metrics`` for curves.)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from .._generated.models import Kind4, PipelineStatus
from .engine import PipelineEngine

_TERMINAL = frozenset({PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.cancelled})

# How often the stream re-reads the (in-memory) run snapshot. Independent of
# the engine's peer-poll cadence; this is cheap (a dict lookup + deep copy).
DEFAULT_SSE_INTERVAL_S = 0.5


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def pipeline_event_stream(
    engine: PipelineEngine,
    run_id: UUID,
    *,
    interval: float = DEFAULT_SSE_INTERVAL_S,
) -> AsyncIterator[str]:
    """Yield SSE frames until the run reaches a terminal state.

    The caller is responsible for having checked the run exists (the route
    returns 404 first); if it vanishes mid-stream we emit a final ``error``.
    """
    seen: dict[Kind4, PipelineStatus] = {}
    while True:
        run = engine.get(run_id)
        if run is None:
            yield _sse("error", '{"detail":"pipeline run no longer exists"}')
            return

        for stage in run.stages:
            if seen.get(stage.kind) == stage.status:
                continue
            seen[stage.kind] = stage.status
            if stage.status == PipelineStatus.running:
                name = "stage_started"
            elif stage.status == PipelineStatus.completed:
                name = "stage_completed"
            else:
                name = "stage_status"
            yield _sse(name, stage.model_dump_json(exclude_none=True))

        if run.status in _TERMINAL:
            name = "done" if run.status != PipelineStatus.failed else "error"
            yield _sse(name, run.model_dump_json(exclude_none=True))
            return

        await asyncio.sleep(interval)
