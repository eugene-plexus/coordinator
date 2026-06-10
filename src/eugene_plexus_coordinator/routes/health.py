"""GET /healthz — liveness / readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.common_models import Health, Status

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    # The component always serves /healthz so config endpoints stay
    # reachable. When the pipeline-execution engine isn't initialized
    # (safe mode, bad config, or — in the v0.3 skeleton — because the
    # engine isn't built yet), surface that as `degraded` so consumers
    # can tell the component is alive but can't run pipelines until it's
    # fixed. Project CRUD stays available regardless.
    engine = getattr(request.app.state, "engine", None)
    engine_error = getattr(request.app.state, "engine_error", None)
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))

    if safe_mode or engine is None:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="coordinator",
            safeMode=safe_mode,
            details={"engine_error": engine_error},
        )

    return Health(
        status=Status.ok,
        version=__version__,
        component="coordinator",
        safeMode=False,
    )
