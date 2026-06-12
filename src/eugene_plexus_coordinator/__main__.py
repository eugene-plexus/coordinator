"""Entrypoint: `python -m eugene_plexus_coordinator`."""

from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

# Default bind port for standalone launch. The watchdog overrides via
# EUGENE_PLEXUS_CRD_BIND_PORT.
_DEFAULT_PORT = 8086


def main() -> None:
    settings = load_settings()

    bootstrap_store = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        bootstrap_store.load()

    env_port = os.environ.get("EUGENE_PLEXUS_CRD_BIND_PORT")
    port = int(env_port) if env_port else _DEFAULT_PORT

    # See the other platform components' __main__ for the rationale.
    # uvicorn only touches its own loggers; basicConfig with force=True
    # gives our application warnings/info a timestamp + level + logger
    # name so they're scannable in the watchdog's combined log.
    log_level = str(bootstrap_store.get("logLevel") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
