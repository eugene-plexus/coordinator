"""In-memory TrainingProject store for the v0.3 skeleton.

The coordinator owns the TrainingProject aggregate. The real component
will persist these to disk (see the `projectStorePath` config field); the
skeleton keeps them in a thread-safe in-memory dict, which is enough to
exercise the full project-CRUD wire shape end-to-end. Pipeline-run
storage + the cross-component execution engine are future work, so this
store covers projects only.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from uuid import UUID, uuid4

from ._generated.models import TrainingProject


class ProjectStore:
    """Thread-safe in-memory CRUD over TrainingProjects, keyed by projectId."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._projects: dict[UUID, TrainingProject] = {}

    def list(self) -> list[TrainingProject]:
        with self._lock:
            return list(self._projects.values())

    def get(self, project_id: UUID) -> TrainingProject | None:
        with self._lock:
            return self._projects.get(project_id)

    def create(self, project: TrainingProject) -> TrainingProject:
        """Store a new project, assigning a server-side id + timestamps.

        The client may post a projectId, but the coordinator owns the id
        space and mints a fresh one so two clients can't collide.
        """
        now = datetime.now(UTC)
        with self._lock:
            stored = project.model_copy(
                update={"projectId": uuid4(), "createdAt": now, "updatedAt": now}
            )
            self._projects[stored.projectId] = stored
            return stored

    def update(self, project_id: UUID, patch: TrainingProject) -> TrainingProject | None:
        """Replace a stored project's mutable fields, preserving its id +
        createdAt and bumping updatedAt. Returns None if absent."""
        with self._lock:
            existing = self._projects.get(project_id)
            if existing is None:
                return None
            stored = patch.model_copy(
                update={
                    "projectId": existing.projectId,
                    "createdAt": existing.createdAt,
                    "updatedAt": datetime.now(UTC),
                }
            )
            self._projects[project_id] = stored
            return stored

    def delete(self, project_id: UUID) -> bool:
        """Remove a project. Returns True if it existed."""
        with self._lock:
            return self._projects.pop(project_id, None) is not None
