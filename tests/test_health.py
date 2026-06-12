"""Tests for /healthz.

With the pipeline-execution engine built at startup (torch-free construction),
a normally-started coordinator reports `ok`. The degraded path (engine unbuilt)
is covered by test_safe_mode.py.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_is_reachable_and_well_formed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "coordinator"
    assert "version" in body


def test_healthz_reports_ok_when_engine_built(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["safeMode"] is False
