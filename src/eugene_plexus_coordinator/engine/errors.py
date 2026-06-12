"""Typed errors for the pipeline-execution engine.

Mirrors the error taxonomy used by the peer engines (trainer/eval/inference):
a small hierarchy the route layer maps to HTTP status codes. `PeerError`
and its `PeerUnavailable` subclass model failures talking to the peer
components the coordinator delegates stages to.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every pipeline-engine error."""


class NotFoundError(EngineError):
    """A referenced pipeline run (or project) does not exist -> 404."""


class ConflictError(EngineError):
    """The operation conflicts with current state -> 409.

    Raised when a second pipeline run is started for a project that already
    has an active one, or when cancelling a run that is already terminal.
    """


class BadRequestError(EngineError):
    """The request cannot be satisfied as posed -> 400.

    Raised when a project has no runnable stages (no peers configured / no
    datasets / no recipe), or a stage's prerequisites are missing.
    """


class PeerError(EngineError):
    """A peer component returned an error response (non-2xx).

    Carries the peer's HTTP status so the runner can record a precise
    stage failure detail.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PeerUnavailable(PeerError):
    """A peer component could not be reached at all (connect/timeout error).

    Distinct from `PeerError` (which is a reachable peer returning non-2xx)
    so the engine can map an unreachable required peer to a `503` at start
    time and a clear stage failure at run time.
    """
