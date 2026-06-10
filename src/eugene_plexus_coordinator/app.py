"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import coordinator as coordinator_routes
from .routes import health as health_routes
from .settings import Settings, load_settings
from .store import ProjectStore

log = logging.getLogger(__name__)

# The v0.3 skeleton ships no pipeline-execution engine. `app.state.engine`
# stays None and pipeline-control routes (start/cancel/events) return 501;
# `engine_error` explains why in /healthz details. Project CRUD is real
# (in-memory store), so the wire shape is fully exercised — only the
# cross-component sequencing is future work.
_SKELETON_ENGINE_ERROR = (
    "pipeline-execution engine not implemented in the v0.3 skeleton; "
    "pipeline start/cancel/events return 501 (project CRUD is live)"
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    config_store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_CRD_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix config via /v1/config, then "
            "restart without the env var.",
            settings.config_file,
        )
    else:
        config_store.load()
    app.state.config_store = config_store
    app.state.safe_mode = settings.safe_mode

    # v0.2 auth state. Tests can pre-populate `app.state.auth_state` to
    # exercise authed paths; the default lifespan build reads env vars
    # via Settings and produces an auth-disabled state when the watchdog
    # didn't supply AUTH_SIGNING_KEY.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    # The project store is real in the skeleton — an in-memory CRUD store
    # for TrainingProjects. Tests can pre-populate it; otherwise the
    # lifespan builds a fresh empty one.
    if not hasattr(app.state, "project_store"):
        app.state.project_store = ProjectStore()

    # The pipeline-execution engine is future work. We wire `app.state.engine`
    # (None for now) and an explanatory `engine_error` so /healthz reports
    # `degraded` and the pipeline-control routes have a uniform place to
    # check. When the engine is implemented this is where it gets built —
    # and a build failure here surfaces as degraded mode instead of
    # crashing the process, per feedback_degraded_mode_required.md.
    if not hasattr(app.state, "engine"):
        app.state.engine = None
        app.state.engine_error = _SKELETON_ENGINE_ERROR

    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — coordinator",
        description=(
            "Control plane for the local-LLM-training platform. Owns the "
            "TrainingProject aggregate and sequences pipeline runs across "
            "components. v0.3 skeleton ships the control-plane wire shape; "
            "the pipeline-execution engine is future work."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Config edits are operator-only — service tokens are rejected so a
    # compromised peer can't reconfigure the coordinator (e.g. repoint a
    # peer URL).
    operator = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator)
    app.include_router(admin_routes.router, dependencies=operator)

    # Project + pipeline control: operators drive these through the UI; a
    # future peer may drive them with a service token.
    app.include_router(coordinator_routes.router, dependencies=[Depends(require_authorized)])

    return app
