"""Coordinator domain routes: TrainingProject CRUD + pipeline runs.

Project CRUD runs against the in-memory `ProjectStore`. Pipeline control
delegates to the `PipelineEngine` (`app.state.engine`): start plans + launches
a run, get/list read the persisted run store, cancel requests cooperative
cancellation, and events streams live progress over SSE. When the engine is
absent (safe mode, or a build failure degraded the component) pipeline-control
routes return 503 while project CRUD keeps working.

While a project has an active pipeline run its config is frozen: update and
delete return 409 until the run finishes or is cancelled.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import StreamingResponse

from .._generated.common_models import Problem
from .._generated.models import (
    PipelineRun,
    TrainingProject,
    V1CoordinatorProjectsGetResponse,
    V1CoordinatorProjectsProjectIdPipelineRunsGetResponse,
)
from ..engine import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    PipelineEngine,
)
from ..engine.events import pipeline_event_stream
from ..store import ProjectStore

router = APIRouter(tags=["coordinator"])

_ENGINE_UNAVAILABLE = (
    "the pipeline-execution engine is not available (safe mode or a "
    "configuration error degraded the coordinator); fix config via "
    "/v1/config and restart. Project CRUD remains available."
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


def _not_found(operation: str, detail: str) -> Response:
    return _problem_response(
        "not-found",
        status_code=status.HTTP_404_NOT_FOUND,
        title="Not found",
        detail=f"{operation}: {detail}",
    )


def _conflict(operation: str, detail: str) -> Response:
    return _problem_response(
        "conflict",
        status_code=status.HTTP_409_CONFLICT,
        title="Conflict",
        detail=f"{operation}: {detail}",
    )


def _bad_request(operation: str, detail: str) -> Response:
    return _problem_response(
        "bad-request",
        status_code=status.HTTP_400_BAD_REQUEST,
        title="Bad request",
        detail=f"{operation}: {detail}",
    )


def _unavailable(operation: str) -> Response:
    return _problem_response(
        "engine-unavailable",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        title="Pipeline engine unavailable",
        detail=f"{operation}: {_ENGINE_UNAVAILABLE}",
    )


def _store(request: Request) -> ProjectStore:
    store: ProjectStore = request.app.state.project_store
    return store


def _engine(request: Request) -> PipelineEngine | None:
    return getattr(request.app.state, "engine", None)


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
    store = _store(request)
    if store.get(project_id) is None:
        return _not_found("updateProject", f"no project with id {project_id}")
    engine = _engine(request)
    if engine is not None and engine.has_active_run(project_id):
        return _conflict(
            "updateProject",
            "a pipeline run is active for this project; config is frozen until it finishes",
        )
    updated = store.update(project_id, body)
    if updated is None:  # raced with a delete
        return _not_found("updateProject", f"no project with id {project_id}")
    return updated


@router.delete("/v1/coordinator/projects/{project_id}", status_code=204)
async def delete_project(request: Request, project_id: UUID) -> Response:
    store = _store(request)
    if store.get(project_id) is None:
        return _not_found("deleteProject", f"no project with id {project_id}")
    engine = _engine(request)
    if engine is not None and engine.has_active_run(project_id):
        return _conflict("deleteProject", "project has an active pipeline run; cancel it first")
    if not store.delete(project_id):
        return _not_found("deleteProject", f"no project with id {project_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Pipeline runs
# --------------------------------------------------------------------------- #


@router.post(
    "/v1/coordinator/projects/{project_id}/pipeline",
    response_model=PipelineRun,
    status_code=status.HTTP_201_CREATED,
)
async def start_pipeline(request: Request, project_id: UUID) -> PipelineRun | Response:
    project = _store(request).get(project_id)
    if project is None:
        return _not_found("startPipeline", f"no project with id {project_id}")
    engine = _engine(request)
    if engine is None:
        return _unavailable("startPipeline")
    try:
        return engine.start(project)
    except ConflictError as e:
        return _conflict("startPipeline", str(e))
    except BadRequestError as e:
        return _bad_request("startPipeline", str(e))


@router.get(
    "/v1/coordinator/projects/{project_id}/pipeline-runs",
    response_model=V1CoordinatorProjectsProjectIdPipelineRunsGetResponse,
)
async def list_project_pipeline_runs(
    request: Request, project_id: UUID
) -> V1CoordinatorProjectsProjectIdPipelineRunsGetResponse | Response:
    if _store(request).get(project_id) is None:
        return _not_found("listProjectPipelineRuns", f"no project with id {project_id}")
    engine = _engine(request)
    runs = engine.list_for_project(project_id) if engine is not None else []
    return V1CoordinatorProjectsProjectIdPipelineRunsGetResponse(pipelineRuns=runs)


@router.get("/v1/coordinator/pipeline-runs/{pipeline_run_id}", response_model=PipelineRun)
async def get_pipeline_run(request: Request, pipeline_run_id: UUID) -> PipelineRun | Response:
    engine = _engine(request)
    run = engine.get(pipeline_run_id) if engine is not None else None
    if run is None:
        return _not_found("getPipelineRun", f"no pipeline run with id {pipeline_run_id}")
    return run


@router.post(
    "/v1/coordinator/pipeline-runs/{pipeline_run_id}/cancel",
    response_model=PipelineRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_pipeline_run(request: Request, pipeline_run_id: UUID) -> PipelineRun | Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable("cancelPipelineRun")
    try:
        return engine.cancel(pipeline_run_id)
    except NotFoundError as e:
        return _not_found("cancelPipelineRun", str(e))
    except ConflictError as e:
        return _conflict("cancelPipelineRun", str(e))


@router.get("/v1/coordinator/pipeline-runs/{pipeline_run_id}/events")
async def stream_pipeline_events(request: Request, pipeline_run_id: UUID) -> Response:
    engine = _engine(request)
    if engine is None or engine.get(pipeline_run_id) is None:
        return _not_found("streamPipelineEvents", f"no pipeline run with id {pipeline_run_id}")
    return StreamingResponse(
        pipeline_event_stream(engine, pipeline_run_id),
        media_type="text/event-stream",
    )
