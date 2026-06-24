"""Local-only flow execution — works without Prefect / network credentials.

This is the always-available cloud entry. Wraps the SDK call sequence
into a single function that can be invoked from the CLI or a notebook.
``cloud/prefect.py`` is the optional remote variant."""

from __future__ import annotations

from pathlib import Path

from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import GameReport, StudentEntry

LOG = get_logger("cloud.local")


def run_local_flow(
    config_path: str | Path,
    students: list[StudentEntry],
    *,
    n_episodes: int | None = None,
    seed: int = 0,
    checkpoint_path: str | Path | None = None,
) -> GameReport:
    """Train → play 6 sub-games → return GameReport. Pure-local execution."""
    sdk = MarlSDK(cfg_path=config_path)
    LOG.info("local flow: train")
    sdk.train(n_episodes=n_episodes)
    if checkpoint_path:
        sdk.save_checkpoint(checkpoint_path)
    LOG.info("local flow: play_game")
    return sdk.play_game(
        group_name=sdk.config.get("submission.group_name", "TBD"),
        group_code=sdk.config.get("submission.group_code", "TBD"),
        github_repo=sdk.config.get("submission.github_repo", "?"),
        students=students,
        timezone_name=sdk.config.get("submission.timezone", "UTC"),
        seed=seed,
    )
