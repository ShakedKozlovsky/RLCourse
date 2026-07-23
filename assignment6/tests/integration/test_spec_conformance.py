"""Spec § 3.5 JSON shape + § 3.6 yaml wiring conformance — catches drift
between the spec example and the runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from marl_lab.gmail.formatter import report_to_json
from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.shared.types import StudentEntry


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.19", "seed": 0, "device": "cpu",
        "game": {"grid_size": [4, 4], "max_moves": 6, "num_games": 6,
                  "max_barriers": 2, "enable_barriers": False, "observation_radius": 1},
        "scoring": {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5},
        "marl": {
            "algorithm": "qmix", "gamma": 0.99, "tau": 0.005, "critic_lr": 1e-3,
            "batch_size": 4, "replay_capacity": 32, "warmup_steps": 25,
            "max_grad_norm": 1.0, "hidden_sizes": [16], "rnn_hidden_size": 8,
            "embed_dim": 8, "hyper_hidden": 16,
        },
        "exploration": {"epsilon_initial": 1.0, "epsilon_final": 0.05, "decay_steps": 100},
        "training": {"total_episodes": 3},
        "experiments": {}, "mcp": {}, "gmail": {},
        "submission": {
            "group_code": "ABCDE123", "group_name": "Team-Alpha",
            "github_repo": "https://github.com/team-alpha/marl-cop-thief",
            "timezone": "Asia/Jerusalem",
        },
        "paths": {}, "graphify": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_json_shape_matches_spec_section_3_5(cfg: Path) -> None:
    """Validate every field in the spec § 3.5 JSON example."""
    sdk = MarlSDK(cfg_path=cfg)
    sdk.train(n_episodes=2)
    report = sdk.play_game(
        group_name="Team-Alpha", group_code="ABCDE123",
        github_repo="https://github.com/team-alpha/marl-cop-thief",
        students=[StudentEntry(role="A", full_name="Israel Israeli", id="123456789")],
        timezone_name="Asia/Jerusalem",
    )
    payload = json.loads(report_to_json(report))
    assert payload["group_name"] == "Team-Alpha"
    assert payload["github_repo"] == "https://github.com/team-alpha/marl-cop-thief"
    assert payload["timezone"] == "Asia/Jerusalem"
    assert "students" in payload
    assert "sub_games" in payload
    assert "totals" in payload
    assert payload["totals"].keys() == {"cop", "thief"}


def test_sub_game_ids_are_1_based(cfg: Path) -> None:
    """Spec § 3.5 example shows id ∈ {1, 2, 3, 4, 5, 6}."""
    sdk = MarlSDK(cfg_path=cfg)
    sdk.train(n_episodes=2)
    report = sdk.play_game(
        group_name="g", group_code="c", github_repo="r",
        students=[StudentEntry(role="A", full_name="?", id="1")],
    )
    ids = [sg.id for sg in report.sub_games]
    assert ids == [1, 2, 3, 4, 5, 6]


def test_sub_game_datetimes_use_asia_jerusalem(cfg: Path) -> None:
    """Spec § 3.5 example shows +03:00 offsets — must be Asia/Jerusalem-aware."""
    sdk = MarlSDK(cfg_path=cfg)
    sdk.train(n_episodes=2)
    report = sdk.play_game(
        group_name="g", group_code="c", github_repo="r",
        students=[StudentEntry(role="A", full_name="?", id="1")],
    )
    sg = report.sub_games[0]
    assert sg.start.tzinfo is not None
    assert "Jerusalem" in str(sg.start.tzinfo)


def test_yaml_scoring_is_wired_to_reward_config(cfg: Path, tmp_path: Path) -> None:
    """Changing yaml scoring should flow through to the runtime scoring."""
    # Build a config with scoring.cop_win = 99 instead of 20
    text = cfg.read_text()
    altered = text.replace("cop_win: 20", "cop_win: 99")
    cfg2 = tmp_path / "cfg_altered.yaml"
    cfg2.write_text(altered)
    sdk = MarlSDK(cfg_path=cfg2)
    # The trainer holds the reward cfg via env.reward_cfg
    assert sdk.env.reward_cfg.score_cop_win == 99


def test_totals_match_sum_of_per_sub_game_scores(cfg: Path) -> None:
    sdk = MarlSDK(cfg_path=cfg)
    sdk.train(n_episodes=2)
    report = sdk.play_game(
        group_name="g", group_code="c", github_repo="r",
        students=[StudentEntry(role="A", full_name="?", id="1")],
    )
    assert report.totals["cop"] == sum(sg.scores["cop"] for sg in report.sub_games)
    assert report.totals["thief"] == sum(sg.scores["thief"] for sg in report.sub_games)
