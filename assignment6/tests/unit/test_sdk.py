"""Layer 14 — SDK end-to-end (tiny config, quick train + play)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.shared.types import StudentEntry


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Path:
    """Write a tiny yaml so train() finishes quickly."""
    cfg = {
        "version": "1.18",
        "seed": 0,
        "device": "cpu",
        "game": {"grid_size": [4, 4], "max_moves": 8, "num_games": 2,
                  "max_barriers": 2, "enable_barriers": False,
                  "observation_radius": 1},
        "scoring": {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5},
        "marl": {
            "algorithm": "qmix", "gamma": 0.99, "tau": 0.005,
            "critic_lr": 1e-3, "batch_size": 4, "replay_capacity": 32,
            "warmup_steps": 25, "max_grad_norm": 1.0,
            "hidden_sizes": [16], "rnn_hidden_size": 8,
            "embed_dim": 8, "hyper_hidden": 16,
        },
        "exploration": {
            "kind": "epsilon_greedy",
            "epsilon_initial": 1.0, "epsilon_final": 0.05, "decay_steps": 100,
        },
        "training": {"total_episodes": 3},
        "experiments": {},
        "mcp": {}, "gmail": {}, "submission": {}, "paths": {}, "graphify": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_sdk_loads_from_yaml(tiny_cfg: Path) -> None:
    sdk = MarlSDK(cfg_path=tiny_cfg)
    assert sdk.trainer.cfg.algo == "qmix"
    assert sdk.runner.runner_cfg.n_sub_games == 2


def test_sdk_train_returns_history(tiny_cfg: Path) -> None:
    sdk = MarlSDK(cfg_path=tiny_cfg)
    history = sdk.train(n_episodes=3)
    assert len(history) == 3


def test_sdk_play_game_returns_full_report(tiny_cfg: Path) -> None:
    sdk = MarlSDK(cfg_path=tiny_cfg)
    sdk.train(n_episodes=2)
    students = [StudentEntry(role="A", full_name="Shaked", id="1")]
    report = sdk.play_game(group_name="g", group_code="c",
                            github_repo="r", students=students,
                            timezone_name="UTC")
    assert len(report.sub_games) == 2
    assert "cop" in report.totals
    assert "thief" in report.totals


def test_sdk_save_load_roundtrip(tiny_cfg: Path) -> None:
    sdk = MarlSDK(cfg_path=tiny_cfg)
    sdk.train(n_episodes=2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    sdk.save_checkpoint(ckpt_path)
    sdk2 = MarlSDK(cfg_path=tiny_cfg)
    sdk2.load_checkpoint(ckpt_path)
    # Weight equality check
    p1 = next(sdk.trainer.q_nets["cop"].parameters())
    p2 = next(sdk2.trainer.q_nets["cop"].parameters())
    import torch
    torch.testing.assert_close(p1, p2)
