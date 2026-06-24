"""Prefect Cloud flow — optional remote variant.

Skipped at runtime if ``prefect`` isn't installed. Falls back to the local
flow with a warning. The Prefect API key + workspace are read from env vars
(``PREFECT_API_KEY``, ``PREFECT_API_URL``)."""

from __future__ import annotations

import os
from pathlib import Path

from marl_lab.cloud.local import run_local_flow
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import GameReport, StudentEntry

LOG = get_logger("cloud.prefect")


def run_prefect_flow(
    config_path: str | Path,
    students: list[StudentEntry],
    *,
    n_episodes: int | None = None,
    seed: int = 0,
    checkpoint_path: str | Path | None = None,
) -> GameReport:
    """Wrap run_local_flow with @flow if prefect is installed; else fall back.

    Returns the same GameReport as run_local_flow. If Prefect is missing or
    no API key is set, logs a warning and runs locally."""
    try:
        import prefect  # noqa: F401
    except ImportError:
        LOG.warning("prefect not installed — falling back to local flow")
        return run_local_flow(config_path, students, n_episodes=n_episodes,
                                seed=seed, checkpoint_path=checkpoint_path)
    if not os.environ.get("PREFECT_API_KEY"):
        LOG.warning("PREFECT_API_KEY not set — running flow locally")
        return run_local_flow(config_path, students, n_episodes=n_episodes,
                                seed=seed, checkpoint_path=checkpoint_path)
    from prefect import flow

    @flow(name="marl-train-and-play")
    def _flow():
        return run_local_flow(config_path, students, n_episodes=n_episodes,
                                seed=seed, checkpoint_path=checkpoint_path)
    LOG.info("Prefect Cloud flow scheduled")
    return _flow()
