"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

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
    """Probe each configured peer component's ``/healthz``.

    Per the spec contract, the coordinator's config test reports which of its
    routing targets (trainer/data/eval/inference) are reachable. The live
    probe is delegated to the engine (blocking HTTP, run off the event loop);
    when the engine is absent (safe mode) we fall back to reporting the
    configured-vs-unconfigured set without probing.
    """
    start = time.perf_counter()
    # Body overrides are accepted for protocol uniformity; the coordinator
    # tests its saved config as-is.
    _ = body
    store: ConfigStore = request.app.state.config_store

    configured = [label for label, key in _PEER_KEYS if store.get(key)]
    if not configured:
        return ConfigTestResult(
            ok=False,
            component="coordinator",
            latencyMs=int((time.perf_counter() - start) * 1000),
            error=(
                "No peer components configured. Set at least one of "
                "trainerUrl/dataUrl/evalUrl/inferenceUrl so the coordinator "
                "has a stage to delegate to."
            ),
        )

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return ConfigTestResult(
            ok=False,
            component="coordinator",
            latencyMs=int((time.perf_counter() - start) * 1000),
            error=(
                "Coordinator is in safe mode / degraded; peer reachability "
                f"was not probed. Configured peers: {', '.join(configured)}."
            ),
        )

    probes = await run_in_threadpool(engine.probe_peers)
    reachable = [p.label for p in probes if p.configured and p.reachable]
    unreachable = [(p.label, p.detail) for p in probes if p.configured and not p.reachable]
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if unreachable:
        detail = "; ".join(f"{label} ({why})" for label, why in unreachable)
        return ConfigTestResult(
            ok=False,
            component="coordinator",
            latencyMs=elapsed_ms,
            error=f"Unreachable peer(s): {detail}.",
            summary=f"Reachable: {', '.join(reachable) or 'none'}.",
        )
    return ConfigTestResult(
        ok=True,
        component="coordinator",
        latencyMs=elapsed_ms,
        summary=f"All {len(reachable)} configured peer(s) reachable: {', '.join(reachable)}.",
    )
