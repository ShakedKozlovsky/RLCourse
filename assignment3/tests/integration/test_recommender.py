"""WorkoutRecommender + CLI recommend command — integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.interface.cli.main import cli
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.services.recommender import (
    RecommendedDay,
    WorkoutPlan,
    WorkoutRecommender,
)
from fitness_rl.shared.types import Action


def _identity(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def _env(episode_length: int = 7) -> WorldEnv:
    initial = np.zeros(16, dtype=np.float32)
    initial[0] = 0.5
    initial[1:6] = 0.2
    return WorldEnv(_identity, RewardFunction(), initial, episode_length)


def test_recommend_returns_n_days() -> None:
    torch.manual_seed(0)
    plan = WorkoutRecommender().recommend(
        net=PolicyNet(hidden_size=8),
        env=_env(episode_length=14),
        reward_fn=RewardFunction(),
        n_days=7,
    )
    assert isinstance(plan, WorkoutPlan)
    assert len(plan.days) == 7
    for d in plan.days:
        assert isinstance(d, RecommendedDay)
        assert d.action_name in {"PUSH", "PULL", "LEGS", "CARDIO", "REST"}


def test_recommend_rejects_zero_days() -> None:
    with pytest.raises(ValueError):
        WorkoutRecommender().recommend(
            net=PolicyNet(hidden_size=8),
            env=_env(), reward_fn=RewardFunction(), n_days=0,
        )


def test_parse_history_round_trips() -> None:
    actions = WorkoutRecommender.parse_history("PUSH, PULL ,Rest")
    assert actions == [int(Action.PUSH), int(Action.PULL), int(Action.REST)]


def test_parse_history_empty() -> None:
    assert WorkoutRecommender.parse_history("") == []
    assert WorkoutRecommender.parse_history("   ") == []


def test_parse_history_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        WorkoutRecommender.parse_history("FOO,BAR")


def test_recommend_with_history_mask_respects_constraint() -> None:
    """Pre-seed the history with 2 PUSH; the next pick must not be PUSH."""
    torch.manual_seed(0)
    rec = WorkoutRecommender(action_mask=ActionMask(max_same_group=2))
    plan = rec.recommend(
        net=PolicyNet(hidden_size=8),
        env=_env(episode_length=14),
        reward_fn=RewardFunction(),
        n_days=1,
        recent_actions=[int(Action.PUSH), int(Action.PUSH)],
    )
    assert plan.days[0].action_name != "PUSH"


def test_as_table_contains_summary() -> None:
    torch.manual_seed(0)
    plan = WorkoutRecommender().recommend(
        net=PolicyNet(hidden_size=8),
        env=_env(episode_length=14),
        reward_fn=RewardFunction(),
        n_days=3,
    )
    text = plan.as_table()
    assert "Workout" in text
    assert "Expected total reward" in text


def test_cli_recommend_command(sdk_config: Path) -> None:
    """End-to-end: train briefly, recommend 5 days, parse table."""
    result = CliRunner().invoke(
        cli,
        ["--config", str(sdk_config), "recommend",
         "--algo", "a2c", "--days", "5", "--episodes", "2",
         "--history", "PUSH,PUSH"],
        obj={},
    )
    assert result.exit_code == 0, result.output
    assert "Expected total reward" in result.output


def test_cli_recommend_rejects_unknown_history(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["--config", str(sdk_config), "recommend",
         "--algo", "a2c", "--days", "3", "--episodes", "2",
         "--history", "FOO"],
        obj={},
    )
    assert result.exit_code != 0
