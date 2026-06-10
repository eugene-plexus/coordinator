"""Coordinator domain routes: TrainingProject CRUD + pipeline runs.

v0.3 SKELETON. The coordinator owns the TrainingProject aggregate; that
half is real here — projects are created/read/listed/updated/deleted
against an in-memory store, so the full project wire shape is exercised
end-to-end. The cross-component pipeline-execution engine (sequencing
data prep -> tokenizer -> training -> eval -> serve across the peer
components) is NOT implemented yet, so:

  * `GET  /v1/coordinator/projects/{id}/pipeline-runs` returns an empty
    list (no runs exist) — real, poll-friendly.
  * `POST /v1/coordinator/projects/{id}/pipeline` (start a run) returns
    `501 Not Implemented`.
  * `GET  /v1/coordinator/pipeline-runs/{id}` returns `404` (no runs are
    ever created in the skeleton).
  * `POST /v1/coordinator/pipeline-runs/{id}/cancel` returns `501`.
  * `GET  /v1/coordinator/pipeline-runs/{id}/events` (SSE) returns `501`.

When the engine lands it replaces the 501s; the wire shapes here are the
long-term contract.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from .._generated.common_models import Problem
from .._generated.models import (
    TrainingProject,
    V1CoordinatorProjectsGetResponse,
    V1CoordinatorProjectsProjectIdPipelineRunsGetResponse,
)
from ..store import ProjectStore

router = APIRouter(tags=["coordinator"])

_ENGINE_NOT_IMPLEMENTED = (
    "pipeline-execution engine not implemented in the v0.3 skeleton; "
    "this repo ships the control-plane wire shape (and live project CRUD) only"
)


def _problem_response(operation: str, *, status_code: int, title: str, detail: str) -> Response:
    problem = Problem(
        type=f"https://github.com/eugene-plexus/coordinator#{operation}",
        title=title,
        status=status_code,
        detail=detail,
        component="coordinator",
    )
    return Response(
        content=problem.model_dump_json(exclude_none=True),
        status_code=status_code,
        media_type="application/problem+json",
    )


def _not_implemented(operation: str) -> Response:
    return _problem_response(
        "engine-not-implemented",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        title="Coordinator pipeline engine not implemented",
        detail=f"{operation}: {_ENGINE_NOT_IMPLEMENTED}.",
    )


def _not_found(operation: str, detail: str) -> Response:
    return _problem_response(
        "not-found",
        status_code=status.HTTP_404_NOT_FOUND,
        title="Not found",
        detail=f"{operation}: {detail}",
    )


def _store(request: Request) -> ProjectStore:
    store: ProjectStore = request.app.state.project_store
    return store


# --------------------------------------------------------------------------- #
# Projects (live CRUD)
# --------------------------------------------------------------------------- #


@router.get("/v1/coordinator/projects", response_model=V1CoordinatorProjectsGetResponse)
async def list_projects(request: Request) -> V1CoordinatorProjectsGetResponse:
    return V1CoordinatorProjectsGetResponse(projects=_store(request).list())


@router.post(
    "/v1/coordinator/projects",
    response_model=TrainingProject,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(request: Request, body: TrainingProject) -> TrainingProject:
    return _store(request).create(body)


@router.get("/v1/coordinator/projects/{project_id}", response_model=TrainingProject)
async def get_project(request: Request, project_id: UUID) -> TrainingProject | Response:
    project = _store(request).get(project_id)
    if project is None:
        return _not_found("getProject", f"no project with id {project_id}")
    return project


@router.patch("/v1/coordinator/projects/{project_id}", response_model=TrainingProject)
async def update_project(
    request: Request, project_id: UUID, body: TrainingProject
) -> TrainingProject | Response:
    updated = _store(request).update(project_id, body)
    if updated is None:
        return _not_found("updateProject", f"no project with id {project_id}")
    return updated


@router.delete("/v1/coordinator/projects/{project_id}", status_code=204)
async def delete_project(request: Request, project_id: UUID) -> Response:
    if not _store(request).delete(project_id):
        return _not_found("deleteProject", f"no project with id {project_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Pipeline runs (engine is future work)
# --------------------------------------------------------------------------- #


@router.post(
    "/v1/coordinator/projects/{project_id}/pipeline",
    status_code=status.HTTP_201_CREATED,
)
async def start_pipeline(request: Request, project_id: UUID) -> Response:
    if _store(request).get(project_id) is None:
        return _not_found("startPipeline", f"no project with id {project_id}")
    return _not_implemented("startPipeline")


@router.get(
    "/v1/coordinator/projects/{project_id}/pipeline-runs",
    response_model=V1CoordinatorProjectsProjectIdPipelineRunsGetResponse,
)
async def list_project_pipeline_runs(
    request: Request, project_id: UUID
) -> V1CoordinatorProjectsProjectIdPipelineRunsGetResponse | Response:
    """List a project's pipeline runs. The skeleton has no engine and
    therefore no runs — returns an empty list (not 501) for a known
    project so callers polling for history get a valid empty result."""
    if _store(request).get(project_id) is None:
        return _not_found("listProjectPipelineRuns", f"no project with id {project_id}")
    return V1CoordinatorProjectsProjectIdPipelineRunsGetResponse(pipelineRuns=[])


@router.get("/v1/coordinator/pipeline-runs/{pipeline_run_id}")
async def get_pipeline_run(request: Request, pipeline_run_id: UUID) -> Response:
    # No pipeline runs are ever created in the skeleton (start returns
    # 501), so every lookup is honestly a 404.
    return _not_found("getPipelineRun", f"no pipeline run with id {pipeline_run_id}")


@router.post("/v1/coordinator/pipeline-runs/{pipeline_run_id}/cancel", status_code=202)
async def cancel_pipeline_run(request: Request, pipeline_run_id: UUID) -> Response:
    return _not_implemented("cancelPipelineRun")


@router.get("/v1/coordinator/pipeline-runs/{pipeline_run_id}/events")
async def stream_pipeline_events(request: Request, pipeline_run_id: UUID) -> Response:
    return _not_implemented("streamPipelineEvents")
