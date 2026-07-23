"""Layer 23 — reproducibility drift test.

Two SDK instances with the SAME yaml + SAME seed must produce
bit-identical Q-net weights after training for the same number of episodes.
This catches drift from un-seeded RNGs, dict ordering, etc."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from marl_lab.sdk.marl_sdk import MarlSDK


@pytest.fixture
def repro_cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.16", "seed": 12345, "device": "cpu",
        "game": {"grid_size": [4, 4], "max_moves": 6, "num_games": 2,
                  "max_barriers": 2, "enable_barriers": False, "observation_radius": 1},
        "scoring": {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5},
        "marl": {
            "algorithm": "qmix", "gamma": 0.99, "tau": 0.005, "critic_lr": 1e-3,
            "batch_size": 4, "replay_capacity": 32, "warmup_steps": 25,
            "max_grad_norm": 1.0, "hidden_sizes": [16], "rnn_hidden_size": 8,
            "embed_dim": 8, "hyper_hidden": 16,
        },
        "exploration": {"epsilon_initial": 1.0, "epsilon_final": 0.05, "decay_steps": 100},
        "training": {"total_episodes": 5},
        "experiments": {}, "mcp": {}, "gmail": {},
        "submission": {"group_code": "x", "group_name": "y",
                        "github_repo": "r", "timezone": "UTC"},
        "paths": {}, "graphify": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_repro_two_runs_identical_q_weights(repro_cfg: Path) -> None:
    """Same yaml + same seed → identical Q-net weights after 5 episodes."""
    sdk_a = MarlSDK(cfg_path=repro_cfg)
    sdk_a.train(n_episodes=5)
    sdk_b = MarlSDK(cfg_path=repro_cfg)
    sdk_b.train(n_episodes=5)
    for a_param, b_param in zip(
        sdk_a.trainer.q_nets["cop"].parameters(),
        sdk_b.trainer.q_nets["cop"].parameters(),
        strict=True,
    ):
        torch.testing.assert_close(a_param, b_param)


def test_repro_different_seeds_diverge(repro_cfg: Path, tmp_path: Path) -> None:
    """Different seed → different weights. (Drift detection sanity check.)"""
    # Build a second config with a DIFFERENT seed
    text = repro_cfg.read_text()
    text_b = text.replace("seed: 12345", "seed: 99999")
    cfg_b = tmp_path / "cfg_b.yaml"
    cfg_b.write_text(text_b)
    sdk_a = MarlSDK(cfg_path=repro_cfg)
    sdk_a.train(n_episodes=5)
    sdk_b = MarlSDK(cfg_path=cfg_b)
    sdk_b.train(n_episodes=5)
    p_a = next(sdk_a.trainer.q_nets["cop"].parameters())
    p_b = next(sdk_b.trainer.q_nets["cop"].parameters())
    assert (p_a - p_b).abs().max() > 1e-7
