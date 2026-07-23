"""Layer 1 — `shared/*` round-trip tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import yaml

from marl_lab.shared.config import ConfigError, ConfigManager
from marl_lab.shared.seed import set_global_seed
from marl_lab.shared.types import (
    EpisodeSequence,
    GameReport,
    Obs,
    StepDiagnostic,
    StudentEntry,
    SubGameResult,
    Transition,
)
from marl_lab.shared.version import __version__


def test_version_pinned() -> None:
    assert __version__ == "1.19"


def test_config_loads_default() -> None:
    cfg = ConfigManager()
    assert cfg.get("marl.gamma") == pytest.approx(0.99)
    assert cfg.get("marl.algorithm") == "maddpg"   # v1.13 default (was "qmix"; see FAILURE_MODES § 8)
    assert cfg.get("game.grid_size") == [5, 5]
    assert cfg.get("game.num_games") == 6
    assert cfg.get("missing.key", default="dflt") == "dflt"


def test_config_version_mismatch_raises(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"version": "0.99"}))
    with pytest.raises(ConfigError, match="Version mismatch"):
        ConfigManager(bad)


def test_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Config not found"):
        ConfigManager(tmp_path / "does_not_exist.yaml")


def test_config_non_mapping_root_raises(tmp_path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        ConfigManager(bad)


def test_config_dotted_access_deep() -> None:
    cfg = ConfigManager()
    assert cfg.get("mcp.cop_port") == 7301
    assert cfg.get("mcp.thief_port") == 7302
    assert cfg.get("scoring.cop_win") == 20


def test_seed_reproducible() -> None:
    set_global_seed(42)
    a = np.random.rand(5)
    t1 = torch.randn(5)
    set_global_seed(42)
    b = np.random.rand(5)
    t2 = torch.randn(5)
    np.testing.assert_array_equal(a, b)
    torch.testing.assert_close(t1, t2)


def test_obs_frozen() -> None:
    o = Obs(agent_role="cop", vector=np.zeros(18, dtype=np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.agent_role = "thief"


def test_transition_holds_global_state() -> None:
    t = Transition(
        global_state=np.zeros(10, dtype=np.float32),
        joint_obs={"cop": np.zeros(18), "thief": np.zeros(18)},
        joint_action={"cop": 0, "thief": 1},
        joint_reward={"cop": 0.0, "thief": 0.0},
        next_global_state=np.zeros(10, dtype=np.float32),
        next_joint_obs={"cop": np.zeros(18), "thief": np.zeros(18)},
        done=False,
    )
    assert t.joint_action["cop"] == 0
    assert t.joint_action["thief"] == 1


def test_episode_sequence_length() -> None:
    ep = EpisodeSequence()
    assert len(ep) == 0
    for _ in range(5):
        ep.transitions.append(_dummy_transition())
    assert len(ep) == 5


def test_game_report_construction() -> None:
    from datetime import datetime, timezone
    sg = SubGameResult(id=1, start=datetime.now(tz=timezone.utc),
                       end=datetime.now(tz=timezone.utc), moves=17,
                       winner="cop", scores={"cop": 20, "thief": 5})
    rep = GameReport(
        group_name="Test-Group", group_code="TBD-8CHR",
        students=[StudentEntry(role="A", full_name="X", id="1")],
        github_repo="https://github.com/x/y", timezone="Asia/Jerusalem",
        sub_games=[sg], totals={"cop": 20, "thief": 5},
    )
    assert len(rep.sub_games) == 1
    assert rep.totals["cop"] == 20


def test_step_diagnostic_fields() -> None:
    d = StepDiagnostic(step=0, episode=0, actor_loss=0.1, critic_loss=0.2,
                       mixer_loss=0.05, mean_q_cop=1.0, mean_q_thief=0.9,
                       epsilon=0.5, episode_reward_cop=10.0,
                       episode_reward_thief=5.0)
    assert d.episode_reward_cop > d.episode_reward_thief


def _dummy_transition() -> Transition:
    return Transition(
        global_state=np.zeros(4), joint_obs={"cop": np.zeros(2), "thief": np.zeros(2)},
        joint_action={"cop": 0, "thief": 0},
        joint_reward={"cop": 0.0, "thief": 0.0},
        next_global_state=np.zeros(4),
        next_joint_obs={"cop": np.zeros(2), "thief": np.zeros(2)},
        done=False,
    )
