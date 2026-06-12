"""End-to-end tests for the pipeline-execution engine via the HTTP API.

Each test drives a real `PipelineEngine` (real worker threads, real persisted
run store) but with a deterministic in-memory `FakePeerClient` standing in for
the data/trainer/eval/inference components. Hang-mode runs are always
cancelled before the test ends so worker threads exit promptly.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_coordinator.app import create_app
from eugene_plexus_coordinator.settings import Settings

from .conftest import ENGINE_OVERRIDES
from .fakes import FakePeerClient

_TERMINAL = {"completed", "failed", "cancelled"}


def _all_peers_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projectStorePath": str(tmp_path / "store"),
                "dataUrl": "http://data:8088",
                "trainerUrl": "http://trainer:8087",
                "evalUrl": "http://eval:8089",
                "inferenceUrl": "http://inference:8090",
            }
        )
    )
    return config_path


@contextmanager
def _client_with_overrides(
    config_path: Path, fake: FakePeerClient, overrides: dict
) -> Iterator[TestClient]:
    app = create_app(settings=Settings(config_file=config_path))
    app.state.peer_client_override = fake
    app.state.engine_overrides = overrides
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pipeline_client(tmp_path: Path, fake_peer: FakePeerClient) -> Iterator[TestClient]:
    """A client whose coordinator has all four peer components configured."""
    with _client_with_overrides(_all_peers_config(tmp_path), fake_peer, ENGINE_OVERRIDES) as c:
        yield c


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_training_project(
    fake: FakePeerClient,
    *,
    with_eval: bool = True,
    auto_serve: bool = True,
) -> dict:
    """Build a full pretraining project body and register its refs in the fake."""
    dataset_id = str(uuid4())
    tokenizer_id = str(uuid4())
    fake.register_dataset(dataset_id)
    fake.register_tokenizer(tokenizer_id)

    body: dict = {
        "projectId": str(uuid4()),
        "name": "tiny-model",
        "goal": "pretrain_from_scratch",
        "createdAt": "2026-06-12T00:00:00Z",
        "modelTemplate": {
            "name": "tiny",
            "architecture": {
                "modelType": "decoder_only",
                "nLayer": 2,
                "nHead": 2,
                "nEmbd": 8,
                "blockSize": 16,
                "vocabSize": 64,
            },
        },
        "tokenizer": {"tokenizerId": tokenizer_id, "name": "bpe", "vocabSize": 64},
        "datasets": [{"datasetId": dataset_id, "name": "corpus"}],
        "recipe": {"kind": "pretraining"},
        "hyperparameters": {"batchSize": 2, "maxSteps": 5},
    }
    if with_eval:
        body["evalSuites"] = [{"evalSuiteId": str(uuid4()), "name": "smoke"}]
    if auto_serve:
        body["exportSettings"] = {"autoServeOnComplete": True}
    return body


def _create_project(client: TestClient, body: dict) -> str:
    resp = client.post("/v1/coordinator/projects", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["projectId"]


def _start(client: TestClient, project_id: str) -> dict:
    resp = client.post(f"/v1/coordinator/projects/{project_id}/pipeline")
    assert resp.status_code == 201, resp.text
    return resp.json()


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    run: dict = {}
    while time.monotonic() < deadline:
        run = client.get(f"/v1/coordinator/pipeline-runs/{run_id}").json()
        if run.get("status") in _TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not terminate; last={run}")


def _wait_stage(
    client: TestClient, run_id: str, kind: str, status: str, timeout: float = 5.0
) -> dict:
    deadline = time.monotonic() + timeout
    run: dict = {}
    while time.monotonic() < deadline:
        run = client.get(f"/v1/coordinator/pipeline-runs/{run_id}").json()
        for stage in run.get("stages", []):
            if stage["kind"] == kind and stage["status"] == status:
                return run
        time.sleep(0.02)
    raise AssertionError(f"stage {kind} never reached {status}; last={run}")


def _stage_statuses(run: dict) -> dict[str, str]:
    return {s["kind"]: s["status"] for s in run["stages"]}


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_full_pipeline_completes(pipeline_client: TestClient, fake_peer: FakePeerClient) -> None:
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])

    assert final["status"] == "completed"
    assert _stage_statuses(final) == {
        "data_prep": "completed",
        "tokenizer": "completed",
        "training": "completed",
        "eval": "completed",
        "serve": "completed",
    }
    # The whole chain was exercised against the peers.
    assert fake_peer.was_called("get_dataset")
    assert fake_peer.was_called("pretokenize_dataset")
    assert fake_peer.was_called("start_training_run")
    assert fake_peer.was_called("start_eval_run")
    assert fake_peer.was_called("load_endpoint")


def test_completed_run_records_resource_ids(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])
    stages = {s["kind"]: s for s in final["stages"]}
    # training tracks the trainer runId; serve tracks the endpoint id.
    assert stages["training"]["resourceId"]
    assert stages["serve"]["resourceId"]
    assert final["currentStage"] is None


def test_pipeline_run_is_listed_for_project(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    _wait_terminal(pipeline_client, run["pipelineRunId"])
    listed = pipeline_client.get(f"/v1/coordinator/projects/{project_id}/pipeline-runs")
    assert listed.status_code == 200
    ids = [r["pipelineRunId"] for r in listed.json()["pipelineRuns"]]
    assert run["pipelineRunId"] in ids


# --------------------------------------------------------------------------- #
# stage planning / skipping
# --------------------------------------------------------------------------- #


def test_evaluate_only_project_runs_eval_against_base_checkpoint(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    body = {
        "projectId": str(uuid4()),
        "name": "eval-only",
        "goal": "evaluate",
        "createdAt": "2026-06-12T00:00:00Z",
        "recipe": {"kind": "sft", "baseCheckpoint": {"checkpointId": str(uuid4())}},
        "evalSuites": [{"evalSuiteId": str(uuid4()), "name": "regression"}],
    }
    project_id = _create_project(pipeline_client, body)
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])

    assert final["status"] == "completed"
    statuses = _stage_statuses(final)
    assert statuses["eval"] == "completed"
    assert statuses["training"] == "skipped"
    assert statuses["data_prep"] == "skipped"
    assert statuses["serve"] == "skipped"
    # eval ran against the base checkpoint (no training stage produced one).
    assert fake_peer.was_called("start_eval_run")
    assert not fake_peer.was_called("start_training_run")


def test_project_with_no_runnable_stages_400(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    # evaluate goal, but no eval suites and nothing else applicable.
    body = {
        "projectId": str(uuid4()),
        "name": "empty",
        "goal": "evaluate",
        "createdAt": "2026-06-12T00:00:00Z",
    }
    project_id = _create_project(pipeline_client, body)
    resp = pipeline_client.post(f"/v1/coordinator/projects/{project_id}/pipeline")
    assert resp.status_code == 400
    assert "runnable" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# failure propagation
# --------------------------------------------------------------------------- #


def test_training_failure_fails_run_and_cancels_downstream(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    fake_peer.training_mode = "fail"
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])

    assert final["status"] == "failed"
    assert "training" in final["lastError"]
    statuses = _stage_statuses(final)
    assert statuses["data_prep"] == "completed"
    assert statuses["tokenizer"] == "completed"
    assert statuses["training"] == "failed"
    # downstream stages were not run.
    assert statuses["eval"] == "cancelled"
    assert statuses["serve"] == "cancelled"
    assert not fake_peer.was_called("start_eval_run")


def test_empty_dataset_fails_data_prep(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    body = make_training_project(fake_peer)
    # Mark the project's dataset as never-imported.
    dataset_id = body["datasets"][0]["datasetId"]
    fake_peer.register_dataset(dataset_id, status="empty")
    project_id = _create_project(pipeline_client, body)
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])

    assert final["status"] == "failed"
    assert _stage_statuses(final)["data_prep"] == "failed"
    assert not fake_peer.was_called("start_training_run")


# --------------------------------------------------------------------------- #
# one-active-run-per-project
# --------------------------------------------------------------------------- #


def test_second_pipeline_for_active_project_409(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    fake_peer.training_mode = "hang"  # keep the run active
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)

    # A second start while the first is active is rejected.
    _wait_stage(pipeline_client, run["pipelineRunId"], "training", "running")
    second = pipeline_client.post(f"/v1/coordinator/projects/{project_id}/pipeline")
    assert second.status_code == 409

    # cleanup: cancel so the worker exits before teardown.
    pipeline_client.post(f"/v1/coordinator/pipeline-runs/{run['pipelineRunId']}/cancel")
    _wait_terminal(pipeline_client, run["pipelineRunId"])


# --------------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------------- #


def test_cancel_running_pipeline(pipeline_client: TestClient, fake_peer: FakePeerClient) -> None:
    fake_peer.training_mode = "hang"
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    run_id = run["pipelineRunId"]

    _wait_stage(pipeline_client, run_id, "training", "running")
    cancel = pipeline_client.post(f"/v1/coordinator/pipeline-runs/{run_id}/cancel")
    assert cancel.status_code == 202

    final = _wait_terminal(pipeline_client, run_id)
    assert final["status"] == "cancelled"
    assert _stage_statuses(final)["training"] == "cancelled"
    # the underlying trainer run was cancelled too.
    assert fake_peer.was_called("cancel_training_run")


def test_cancel_terminal_run_409(pipeline_client: TestClient, fake_peer: FakePeerClient) -> None:
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    _wait_terminal(pipeline_client, run["pipelineRunId"])
    resp = pipeline_client.post(f"/v1/coordinator/pipeline-runs/{run['pipelineRunId']}/cancel")
    assert resp.status_code == 409


def test_cancel_unknown_run_404(pipeline_client: TestClient) -> None:
    resp = pipeline_client.post(f"/v1/coordinator/pipeline-runs/{uuid4()}/cancel")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# config freeze while a run is active
# --------------------------------------------------------------------------- #


def test_project_config_frozen_during_active_run(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    fake_peer.training_mode = "hang"
    body = make_training_project(fake_peer)
    project_id = _create_project(pipeline_client, body)
    run = _start(pipeline_client, project_id)
    run_id = run["pipelineRunId"]
    _wait_stage(pipeline_client, run_id, "training", "running")

    patch_body = {**body, "name": "renamed"}
    update = pipeline_client.patch(f"/v1/coordinator/projects/{project_id}", json=patch_body)
    assert update.status_code == 409
    delete = pipeline_client.delete(f"/v1/coordinator/projects/{project_id}")
    assert delete.status_code == 409

    # once cancelled, edits are allowed again.
    pipeline_client.post(f"/v1/coordinator/pipeline-runs/{run_id}/cancel")
    _wait_terminal(pipeline_client, run_id)
    assert pipeline_client.delete(f"/v1/coordinator/projects/{project_id}").status_code == 204


# --------------------------------------------------------------------------- #
# SSE events
# --------------------------------------------------------------------------- #


def test_pipeline_events_stream_emits_done(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    run_id = run["pipelineRunId"]
    _wait_terminal(pipeline_client, run_id)

    with pipeline_client.stream(
        "GET", f"/v1/coordinator/pipeline-runs/{run_id}/events"
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "event: stage_completed" in body
    assert "event: done" in body


def test_pipeline_events_unknown_run_404(pipeline_client: TestClient) -> None:
    resp = pipeline_client.get(f"/v1/coordinator/pipeline-runs/{uuid4()}/events")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# serve stage: load polling + error
# --------------------------------------------------------------------------- #


def test_serve_polls_load_until_ready(
    pipeline_client: TestClient, fake_peer: FakePeerClient
) -> None:
    fake_peer.inference_mode = "loading_then_ready"
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])
    assert final["status"] == "completed"
    assert _stage_statuses(final)["serve"] == "completed"
    # the readiness poll path was exercised.
    assert fake_peer.was_called("list_endpoints")


def test_serve_load_error_fails_run(pipeline_client: TestClient, fake_peer: FakePeerClient) -> None:
    fake_peer.inference_mode = "error"
    project_id = _create_project(pipeline_client, make_training_project(fake_peer))
    run = _start(pipeline_client, project_id)
    final = _wait_terminal(pipeline_client, run["pipelineRunId"])
    assert final["status"] == "failed"
    assert _stage_statuses(final)["serve"] == "failed"
    assert "serve" in final["lastError"]


# --------------------------------------------------------------------------- #
# stage timeout (no operator cancel)
# --------------------------------------------------------------------------- #


def test_stage_times_out_without_cancel(tmp_path: Path, fake_peer: FakePeerClient) -> None:
    # A wedged peer (training never leaves 'running') must trip the per-stage
    # deadline and fail the run on its own — the only guard against a worker
    # polling forever.
    fake_peer.training_mode = "hang"
    overrides = {"poll_interval": 0.01, "stage_timeout": 0.15, "max_workers": 2}
    with _client_with_overrides(_all_peers_config(tmp_path), fake_peer, overrides) as client:
        project_id = _create_project(client, make_training_project(fake_peer))
        run = _start(client, project_id)
        final = _wait_terminal(client, run["pipelineRunId"])
        assert final["status"] == "failed"
        assert "timed out" in (final["lastError"] or "")
        assert _stage_statuses(final)["training"] == "failed"
