"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from .._generated.common_models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])

_PEER_KEYS = (
    ("trainer", "trainerUrl"),
    ("data", "dataUrl"),
    ("eval", "evalUrl"),
    ("inference", "inferenceUrl"),
)


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(
    request: Request,
    body: ConfigUpdateRequest,
) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Report which peer components the coordinator is configured to route to.

    The spec contract is "probe each peer's /healthz". The v0.3 skeleton
    has no execution engine and makes no outbound calls, so it reports the
    configured-vs-unconfigured peer set — the cheap, real part of the
    contract — in the standard `ConfigTestResult` shape. The engine work
    adds the live /healthz probe here.
    """
    start = time.perf_counter()
    # Body overrides are accepted for protocol uniformity but the skeleton
    # tests the saved config as-is; the engine work will honor overrides.
    _ = body
    store: ConfigStore = request.app.state.config_store

    configured: list[str] = []
    for label, key in _PEER_KEYS:
        if store.get(key):
            configured.append(label)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if not configured:
        return ConfigTestResult(
            ok=False,
            component="coordinator",
            latencyMs=elapsed_ms,
            error=(
                "No peer components configured. Set at least one of "
                "trainerUrl/dataUrl/evalUrl/inferenceUrl so the coordinator "
                "has a stage to delegate to."
            ),
        )
    return ConfigTestResult(
        ok=True,
        component="coordinator",
        latencyMs=elapsed_ms,
        summary=(
            f"Configured peers: {', '.join(configured)}. Live reachability "
            "is probed once the pipeline-execution engine is implemented."
        ),
    )
