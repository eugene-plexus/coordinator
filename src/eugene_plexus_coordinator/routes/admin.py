"""Admin endpoints: restart."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter

from .._generated.common_models import RestartResult

router = APIRouter(tags=["config"])

log = logging.getLogger(__name__)

# Long enough for the 202 response body to flush back to the client over
# a slow LAN, short enough that the operator doesn't sit waiting.
_RESTART_DELAY_MS = 500


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart() -> RestartResult:
    """Schedule a process exit so a supervisor can relaunch with new config.

    Mirrors the orchestrator/hemisphere-driver restart endpoints. The
    coordinator only re-reads `requiresRestart: true` config keys
    (logLevel, port, …) at startup; this is the UI's mechanism for
    completing a config-change flow.
    """
    log.warning("restart requested via /v1/admin/restart; exiting in %dms", _RESTART_DELAY_MS)

    loop = asyncio.get_event_loop()
    loop.call_later(_RESTART_DELAY_MS / 1000.0, lambda: os._exit(0))

    return RestartResult(
        scheduled=True,
        delayMs=_RESTART_DELAY_MS,
        message=(
            f"Process exiting in {_RESTART_DELAY_MS}ms. A supervisor (systemd, "
            "docker, deploy launcher, …) is expected to relaunch it; in "
            "personal-use installs without one, relaunch manually."
        ),
    )
