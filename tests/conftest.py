"""Pytest fixtures shared across the test suite.

The default `client` fixture wires a fake peer client into the pipeline engine
(no network) with fast poll/timeout settings, and seeds `trainerUrl` so the
config/health tests see a configured, reachable peer. Pipeline tests that need
all four peers configured build their own client (see test_pipeline.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_coordinator.app import create_app
from eugene_plexus_coordinator.settings import Settings

from .fakes import FakePeerClient

# Fast engine timings so worker threads finish within a test.
ENGINE_OVERRIDES = {"poll_interval": 0.01, "stage_timeout": 10.0, "max_workers": 2}


@pytest.fixture
def fake_peer() -> FakePeerClient:
    return FakePeerClient()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Pre-seed config so projectStorePath lands inside tmp_path and at
    # least one peer is configured (so /v1/config/test reports ok).
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projectStorePath": str(tmp_path / "store"),
                "trainerUrl": "http://127.0.0.1:8087",
            }
        )
    )
    return Settings(config_file=config_path)


@pytest.fixture
def app(settings: Settings, fake_peer: FakePeerClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.peer_client_override = fake_peer
    app.state.engine_overrides = ENGINE_OVERRIDES
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
