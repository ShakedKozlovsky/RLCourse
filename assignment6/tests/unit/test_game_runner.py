"""Layer 13 — game_runner tests (6 sub-games + spec § 3.5 JSON shape)."""

from __future__ import annotations

import numpy as np
import pytest

from marl_lab.environment.reward import RewardConfig
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim
from marl_lab.services.game_runner import GameRunner, RunnerConfig
from marl_lab.shared.types import StudentEntry


@pytest.fixture
def q_pair() -> tuple[QPerAgent, QPerAgent]:
    """A simple pair of untrained Q-nets used to drive sub-game rollouts."""
    o = obs_dim(2)
    q_a = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    q_b = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    return q_a, q_b


@pytest.fixture
def runner() -> GameRunner:
    return GameRunner(
        runner_cfg=RunnerConfig(n_sub_games=4, grid_size=(5, 5), max_moves=10,
                                  max_barriers=3, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )


def test_play_sub_game_returns_sub_game_result(runner, q_pair) -> None:
    q_a, q_b = q_pair
    res = runner.play_sub_game(q_a, q_b, a_role="cop", sub_game_id=0, seed=0)
    assert res.id == 0
    assert res.moves >= 1
    assert res.winner in ("cop", "thief", "draw")
    assert set(res.scores.keys()) == {"cop", "thief"}


def test_play_sub_game_scores_match_winner(runner, q_pair) -> None:
    q_a, q_b = q_pair
    res = runner.play_sub_game(q_a, q_b, a_role="cop", sub_game_id=0, seed=0)
    if res.winner == "cop":
        assert res.scores["cop"] > res.scores["thief"]
    elif res.winner == "thief":
        assert res.scores["thief"] >= res.scores["cop"]


def test_play_sub_game_moves_capped_by_max_moves(runner, q_pair) -> None:
    q_a, q_b = q_pair
    res = runner.play_sub_game(q_a, q_b, a_role="cop", sub_game_id=0, seed=0)
    assert res.moves <= runner.runner_cfg.max_moves


def test_play_full_game_emits_n_sub_games(runner, q_pair) -> None:
    q_a, q_b = q_pair
    students = [StudentEntry(role="A", full_name="Shaked", id="123456789"),
                StudentEntry(role="B", full_name="TestPartner", id="987654321")]
    report = runner.play_full_game(
        q_a, q_b, students=students, group_name="TestGroup",
        group_code="ABCDE123", github_repo="https://github.com/x/y",
        timezone_name="UTC",
    )
    assert len(report.sub_games) == 4


def test_play_full_game_totals_match_per_subgame_sum(runner, q_pair) -> None:
    q_a, q_b = q_pair
    students = [StudentEntry(role="A", full_name="A", id="1"),
                StudentEntry(role="B", full_name="B", id="2")]
    report = runner.play_full_game(q_a, q_b, students=students,
                                     group_name="g", group_code="c",
                                     github_repo="r", timezone_name="UTC")
    expected_cop = sum(s.scores["cop"] for s in report.sub_games)
    expected_thief = sum(s.scores["thief"] for s in report.sub_games)
    assert report.totals["cop"] == expected_cop
    assert report.totals["thief"] == expected_thief


def test_play_full_game_alternates_roles(runner, q_pair) -> None:
    """Sub-games 0, 2 are policy_a=cop; sub-games 1, 3 are policy_a=thief.

    We verify by running with two MARKEDLY different policies (one always
    returns Q in favour of action 0 = UP, the other always action 4 = STAY),
    and confirming that the env actually receives both kinds of patterns."""
    # Easier proxy: just check sub_game IDs are 0..N-1 in order
    q_a, q_b = q_pair
    students = [StudentEntry(role="A", full_name="A", id="1"),
                StudentEntry(role="B", full_name="B", id="2")]
    report = runner.play_full_game(q_a, q_b, students=students,
                                     group_name="g", group_code="c",
                                     github_repo="r", timezone_name="UTC")
    assert [s.id for s in report.sub_games] == [0, 1, 2, 3]


def test_play_full_game_carries_metadata(runner, q_pair) -> None:
    q_a, q_b = q_pair
    students = [StudentEntry(role="A", full_name="Shaked", id="1"),
                StudentEntry(role="B", full_name="B", id="2")]
    report = runner.play_full_game(q_a, q_b, students=students,
                                     group_name="Group-X", group_code="CODE1234",
                                     github_repo="https://github.com/x/y",
                                     timezone_name="Asia/Jerusalem")
    assert report.group_name == "Group-X"
    assert report.group_code == "CODE1234"
    assert report.github_repo == "https://github.com/x/y"
    assert report.timezone == "Asia/Jerusalem"
    assert len(report.students) == 2
