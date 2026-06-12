"""Stage planning: turn a TrainingProject + configured peers into a stage list.

The pipeline is a fixed canonical sequence —
``data_prep -> tokenizer -> training -> eval -> serve`` — but not every
project exercises every stage. A stage is:

* **pending** — it will run: its peer component is configured AND the
  project's goal calls for it.
* **skipped** — it will not run: the peer isn't configured, or the goal
  doesn't include this stage (e.g. a `serve`-only project skips training).

Planning decides pending-vs-skipped purely from (goal, configured peers,
which references exist). It deliberately does NOT validate the *contents* of a
pending stage — a pending training stage with no architecture, say, is left
pending so the runner can fail it with a precise, user-facing detail rather
than silently skipping it. A project whose every stage is skipped has nothing
to run and is rejected at start (`400`).
"""

from __future__ import annotations

from .._generated.models import (
    ComponentKind,
    Kind4,
    PipelineStage,
    PipelineStatus,
    TrainingGoal,
    TrainingProject,
)

# Goals whose pipeline includes an actual training run.
TRAINING_GOALS = frozenset(
    {
        TrainingGoal.pretrain_from_scratch,
        TrainingGoal.continue_pretraining,
        TrainingGoal.finetune,
        TrainingGoal.train_adapter,
    }
)

# Each pipeline stage delegates to exactly one peer component.
STAGE_COMPONENT: dict[Kind4, ComponentKind] = {
    Kind4.data_prep: ComponentKind.data,
    Kind4.tokenizer: ComponentKind.data,
    Kind4.training: ComponentKind.trainer,
    Kind4.eval: ComponentKind.eval,
    Kind4.serve: ComponentKind.inference,
}


class ConfiguredPeers:
    """Which peer base URLs the coordinator has configured (empty == off)."""

    def __init__(self, *, data: str, trainer: str, eval: str, inference: str) -> None:
        self.data = data or ""
        self.trainer = trainer or ""
        self.eval = eval or ""
        self.inference = inference or ""

    def url_for(self, component: ComponentKind) -> str:
        return {
            ComponentKind.data: self.data,
            ComponentKind.trainer: self.trainer,
            ComponentKind.eval: self.eval,
            ComponentKind.inference: self.inference,
        }.get(component, "")


def _stage(kind: Kind4, *, will_run: bool, skip_detail: str) -> PipelineStage:
    return PipelineStage(
        kind=kind,
        component=STAGE_COMPONENT[kind],
        status=PipelineStatus.pending if will_run else PipelineStatus.skipped,
        detail=None if will_run else skip_detail,
    )


def plan_stages(project: TrainingProject, peers: ConfiguredPeers) -> list[PipelineStage]:
    """Build the ordered stage list for a project. Pure (no I/O)."""
    goal = project.goal
    is_training = goal in TRAINING_GOALS
    has_datasets = bool(project.datasets)
    has_tokenizer = project.tokenizer is not None
    has_suites = bool(project.evalSuites)
    auto_serve = bool(project.exportSettings and project.exportSettings.autoServeOnComplete)

    stages: list[PipelineStage] = []

    # data_prep: verify the project's datasets are imported before training.
    data_prep_on = bool(peers.data) and is_training and has_datasets
    stages.append(
        _stage(
            Kind4.data_prep,
            will_run=data_prep_on,
            skip_detail=_skip_reason(
                peer_off=not peers.data,
                not_for_goal=not is_training,
                missing="no datasets selected" if is_training and not has_datasets else "",
            ),
        )
    )

    # tokenizer: pretokenize each dataset against the project tokenizer.
    tokenizer_on = bool(peers.data) and is_training and has_datasets and has_tokenizer
    stages.append(
        _stage(
            Kind4.tokenizer,
            will_run=tokenizer_on,
            skip_detail=_skip_reason(
                peer_off=not peers.data,
                not_for_goal=not is_training,
                missing=_first_missing(
                    [
                        ("no datasets selected", is_training and not has_datasets),
                        ("no tokenizer selected", is_training and not has_tokenizer),
                    ]
                ),
            ),
        )
    )

    # training: execute the run on the trainer.
    training_on = bool(peers.trainer) and is_training
    stages.append(
        _stage(
            Kind4.training,
            will_run=training_on,
            skip_detail=_skip_reason(peer_off=not peers.trainer, not_for_goal=not is_training),
        )
    )

    # eval: score the produced (or base) checkpoint against the project suites.
    eval_on = bool(peers.eval) and has_suites
    stages.append(
        _stage(
            Kind4.eval,
            will_run=eval_on,
            skip_detail=_skip_reason(
                peer_off=not peers.eval,
                missing="no eval suites selected" if peers.eval and not has_suites else "",
            ),
        )
    )

    # serve: load the checkpoint into an inference endpoint.
    serve_on = bool(peers.inference) and auto_serve and (is_training or goal == TrainingGoal.serve)
    stages.append(
        _stage(
            Kind4.serve,
            will_run=serve_on,
            skip_detail=_skip_reason(
                peer_off=not peers.inference,
                not_for_goal=not (is_training or goal == TrainingGoal.serve),
                missing=(
                    "exportSettings.autoServeOnComplete is off"
                    if peers.inference and not auto_serve
                    else ""
                ),
            ),
        )
    )

    return stages


def has_runnable_stage(stages: list[PipelineStage]) -> bool:
    return any(s.status == PipelineStatus.pending for s in stages)


def _first_missing(candidates: list[tuple[str, bool]]) -> str:
    for message, active in candidates:
        if active:
            return message
    return ""


def _skip_reason(*, peer_off: bool = False, not_for_goal: bool = False, missing: str = "") -> str:
    if missing:
        return missing
    if peer_off:
        return "peer component not configured"
    if not_for_goal:
        return "not part of this project's goal"
    return "not requested"
