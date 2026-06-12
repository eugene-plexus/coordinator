"""Pipeline-execution engine for the coordinator.

Sequences a `PipelineRun` across the platform's peer components
(data prep -> tokenizer -> training -> eval -> serve), delegating each stage
over HTTP and threading the produced checkpoint forward. Pure orchestration:
no model/torch code lives here — that is owned by the trainer/eval/inference
engines this layer calls.
"""

from __future__ import annotations

from .engine import PipelineEngine
from .errors import (
    BadRequestError,
    ConflictError,
    EngineError,
    NotFoundError,
    PeerError,
    PeerUnavailable,
)

__all__ = [
    "BadRequestError",
    "ConflictError",
    "EngineError",
    "NotFoundError",
    "PeerError",
    "PeerUnavailable",
    "PipelineEngine",
]
