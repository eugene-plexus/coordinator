"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config` at runtime. These settings only control bootstrap:
where to find the config file, which interface to bind. Once the config
file is loaded, runtime config takes precedence for everything it covers.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_CRD_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. Override to 0.0.0.0 for tailnet exposure."""

    safe_mode: bool = False
    """If true, skip loading the persisted config file at startup and run on
    built-in defaults. Set by the watchdog via EUGENE_PLEXUS_CRD_SAFE_MODE=1
    when a previous boot failed. PATCH /v1/config still writes to
    `config_file` normally so the operator's repair survives the next
    non-safe-mode boot. Per the safe-mode contract in
    specs/openapi/coordinator.yaml: while safe mode is active the coordinator
    refuses to start pipeline runs and reports degraded health, but config
    endpoints stay reachable so a config that breaks startup never
    soft-bricks the component. Project CRUD over already-stored projects
    continues to work."""

    auth_signing_key: str | None = None
    """Base64-encoded 32-byte HMAC signing key, supplied by the watchdog at
    spawn time (EUGENE_PLEXUS_CRD_AUTH_SIGNING_KEY). When absent the
    component runs unauthenticated — dev / standalone path only."""

    service_token: str | None = None
    """Long-lived service JWT (EUGENE_PLEXUS_CRD_SERVICE_TOKEN). The
    coordinator presents this on outbound calls to its peer components
    (trainer/data/eval/inference) as `aud: service:coordinator`."""

    master_key: str | None = None
    """Base64-encoded 32-byte secretbox key (EUGENE_PLEXUS_CRD_MASTER_KEY).
    Not used in the v0.3 skeleton — no coordinator secrets are encrypted at
    rest yet. Reserved."""


def load_settings() -> Settings:
    return Settings()
