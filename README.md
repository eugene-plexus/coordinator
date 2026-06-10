# eugene-plexus-coordinator

Control plane for the [Eugene Plexus](https://github.com/eugene-plexus)
local-LLM-training platform.

## What this is

The coordinator component of Eugene Plexus. It owns the **TrainingProject**
aggregate — the user-facing "model you are building" (template/architecture,
tokenizer + dataset selection, recipe, hyperparameters, hardware, eval suites,
export settings) — and it executes **pipeline runs** that sequence the work
across the other platform components: data prep + tokenizer (`data`), training
(`trainer`), evaluation (`eval`), and serving (`inference`). It resolves each
peer's URL from the watchdog topology and tracks the underlying resource id
(e.g. a `trainer` runId) on the corresponding pipeline stage.

```
GET    /v1/coordinator/projects                              list projects
POST   /v1/coordinator/projects                              create a project
GET    /v1/coordinator/projects/{projectId}                  read one project
PATCH  /v1/coordinator/projects/{projectId}                  partial update
DELETE /v1/coordinator/projects/{projectId}                  delete a project
POST   /v1/coordinator/projects/{projectId}/pipeline         start a pipeline run
GET    /v1/coordinator/projects/{projectId}/pipeline-runs    list a project's runs
GET    /v1/coordinator/pipeline-runs/{pipelineRunId}         current run status
POST   /v1/coordinator/pipeline-runs/{pipelineRunId}/cancel  cancel a run
GET    /v1/coordinator/pipeline-runs/{pipelineRunId}/events  live progress (SSE)
```

Plus the standard Eugene Plexus config trio (`GET /v1/config`,
`GET /v1/config/schema`, `PATCH /v1/config`), `POST /v1/config/test`,
`POST /v1/admin/restart`, and `GET /healthz`.

## v0.3 skeleton status

This repo currently ships the **control-plane skeleton**. The HTTP wire shape
(routes + generated models + config + auth + health + safe mode) is complete,
and **project CRUD is fully live** against an in-memory store — you can
create/read/list/update/delete TrainingProjects end-to-end. The
cross-component **pipeline-execution engine** is **not implemented yet**:

- `POST /v1/coordinator/projects/{id}/pipeline` (start a run) returns
  `501 Not Implemented`.
- `POST /v1/coordinator/pipeline-runs/{id}/cancel` and
  `GET /v1/coordinator/pipeline-runs/{id}/events` (SSE) return `501`.
- `GET /v1/coordinator/projects/{id}/pipeline-runs` returns an empty list
  (no runs exist yet) for a known project.
- `GET /v1/coordinator/pipeline-runs/{id}` returns `404` (no runs are ever
  created in the skeleton).

`/healthz` reports `degraded` while the pipeline engine is absent — the
component is alive and serves project CRUD + config, but can't run pipelines.

## Quick start

```bash
pip install -e ".[dev]"
python -m eugene_plexus_coordinator
# default port 8086; override via PATCH /v1/config or the config file
```

The first run creates a `config.yaml` in the working directory with the
component's defaults. Edit through the UI, through `PATCH /v1/config`, or
by hand.

## Degraded-mode startup

Per the project-wide rule (`feedback_degraded_mode_required.md`), a bad
config never prevents the component from starting. Config endpoints stay
reachable so operators can fix the broken setting through the UI;
pipeline endpoints behave according to the skeleton (501) until the
execution engine lands.

## Codegen

Pydantic models for the coordinator and shared schemas are generated from
the pinned `eugene-plexus/specs` commit:

```bash
python scripts/codegen.py
```

`SPECS_REF` records the commit SHA. Bump it to track a newer specs
release; CI re-runs codegen and fails if the working tree drifts.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (DCO sign-off required).
