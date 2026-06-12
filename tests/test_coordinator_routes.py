"""Tests for the coordinator domain routes.

Project CRUD (in-memory store) and the pipeline-control routes' validation
paths (404 for unknown project/run, empty run list for a known project). The
full pipeline-execution behavior — start/cancel/events against the engine —
lives in test_pipeline.py, which configures all peers and drives worker
threads to completion.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _project_body(name: str = "test-model") -> dict:
    # projectId / createdAt are required by the schema; the store mints
    # its own server-side, but the request body must still validate.
    return {
        "projectId": str(uuid4()),
        "name": name,
        "goal": "pretrain_from_scratch",
        "createdAt": "2026-06-10T00:00:00Z",
    }


# --------------------------------------------------------------------------- #
# Projects (live CRUD)
# --------------------------------------------------------------------------- #


def test_list_projects_starts_empty(client: TestClient) -> None:
    response = client.get("/v1/coordinator/projects")
    assert response.status_code == 200
    assert response.json() == {"projects": []}


def test_create_then_get_and_list_project(client: TestClient) -> None:
    create = client.post("/v1/coordinator/projects", json=_project_body("alpha"))
    assert create.status_code == 201
    created = create.json()
    assert created["name"] == "alpha"
    project_id = created["projectId"]
    assert project_id  # server-assigned

    got = client.get(f"/v1/coordinator/projects/{project_id}")
    assert got.status_code == 200
    assert got.json()["name"] == "alpha"

    listed = client.get("/v1/coordinator/projects")
    assert listed.status_code == 200
    names = [p["name"] for p in listed.json()["projects"]]
    assert names == ["alpha"]


def test_get_missing_project_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/coordinator/projects/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["component"] == "coordinator"


def test_update_project(client: TestClient) -> None:
    created = client.post("/v1/coordinator/projects", json=_project_body("before")).json()
    project_id = created["projectId"]

    patch_body = _project_body("after")
    response = client.patch(f"/v1/coordinator/projects/{project_id}", json=patch_body)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "after"
    # Server-owned id + createdAt are preserved across the update.
    assert body["projectId"] == project_id
    assert body["createdAt"] == created["createdAt"]


def test_update_missing_project_returns_404(client: TestClient) -> None:
    response = client.patch(f"/v1/coordinator/projects/{uuid4()}", json=_project_body())
    assert response.status_code == 404


def test_delete_project(client: TestClient) -> None:
    created = client.post("/v1/coordinator/projects", json=_project_body()).json()
    project_id = created["projectId"]

    delete = client.delete(f"/v1/coordinator/projects/{project_id}")
    assert delete.status_code == 204

    assert client.get(f"/v1/coordinator/projects/{project_id}").status_code == 404


def test_delete_missing_project_returns_404(client: TestClient) -> None:
    response = client.delete(f"/v1/coordinator/projects/{uuid4()}")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Pipeline-run route validation (no worker threads spawned here)
# --------------------------------------------------------------------------- #


def test_pipeline_runs_list_empty_for_known_project(client: TestClient) -> None:
    created = client.post("/v1/coordinator/projects", json=_project_body()).json()
    project_id = created["projectId"]
    response = client.get(f"/v1/coordinator/projects/{project_id}/pipeline-runs")
    assert response.status_code == 200
    assert response.json() == {"pipelineRuns": []}


def test_pipeline_runs_list_404_for_unknown_project(client: TestClient) -> None:
    response = client.get(f"/v1/coordinator/projects/{uuid4()}/pipeline-runs")
    assert response.status_code == 404


def test_start_pipeline_404_for_unknown_project(client: TestClient) -> None:
    response = client.post(f"/v1/coordinator/projects/{uuid4()}/pipeline")
    assert response.status_code == 404


def test_get_pipeline_run_unknown_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/coordinator/pipeline-runs/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["component"] == "coordinator"


def test_cancel_pipeline_run_unknown_returns_404(client: TestClient) -> None:
    response = client.post(f"/v1/coordinator/pipeline-runs/{uuid4()}/cancel")
    assert response.status_code == 404


def test_pipeline_events_unknown_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/coordinator/pipeline-runs/{uuid4()}/events")
    assert response.status_code == 404
