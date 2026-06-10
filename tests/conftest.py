"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_coordinator.app import create_app
from eugene_plexus_coordinator.settings import Settings


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
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
