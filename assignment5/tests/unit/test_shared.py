"""Layer 1 — `shared/*` round-trip tests."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from roomba_lab.shared.config import ConfigError, ConfigManager
from roomba_lab.shared.seed import set_global_seed
from roomba_lab.shared.types import EpisodeMetrics, StepDiagnostic, Transition
from roomba_lab.shared.version import __version__


def test_version_pinned() -> None:
    assert __version__ == "1.00"


def test_config_loads_default() -> None:
    cfg = ConfigManager()
    assert cfg.get("ddpg.tau") == pytest.approx(0.005)
    assert cfg.get("noise.sigma_initial") == pytest.approx(0.2)
    assert cfg.get("missing.key", default="d") == "d"


def test_config_version_mismatch_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "0.99"}))
    with pytest.raises(ConfigError, match="Version mismatch"):
        ConfigManager(bad)


def test_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Config not found"):
        ConfigManager(tmp_path / "does_not_exist.json")


def test_seed_reproducible() -> None:
    set_global_seed(42)
    a = np.random.rand(5)
    t1 = torch.randn(5)
    set_global_seed(42)
    b = np.random.rand(5)
    t2 = torch.randn(5)
    np.testing.assert_array_equal(a, b)
    torch.testing.assert_close(t1, t2)


def test_transition_frozen() -> None:
    t = Transition(
        state=np.zeros(3),
        action=np.zeros(2),
        reward=1.0,
        next_state=np.ones(3),
        done=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.reward = 2.0


def test_episode_metrics_construction() -> None:
    m = EpisodeMetrics(reward=1.5, length=42, coverage=0.5, collisions=2)
    assert m.coverage == 0.5


def test_step_diagnostic_construction() -> None:
    d = StepDiagnostic(
        step=10,
        actor_loss=0.1,
        critic_loss=0.2,
        mean_q=1.0,
        sigma=0.2,
        episode_reward=5.0,
        coverage=0.3,
    )
    assert d.step == 10
