"""Audit finding #2: same seed → bit-identical training history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proximal_lab.sdk.sdk import ProximalLab


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "gamma": 0.99, "n_parallel_envs": 2,
                 "secondary_id": "Walker2d-v5", "max_episode_steps": 1000},
        "gae": {"lambda": 0.95},
        "actor_critic": {"hidden_sizes": [32, 32], "activation": "tanh",
                          "shared_trunk": False,
                          "log_std_init": -0.5, "log_std_min": -5.0, "log_std_max": 2.0},
        "ppo": {"total_timesteps": 512, "steps_per_rollout": 256,
                 "minibatch_size": 32, "n_epochs_per_update": 1,
                 "clip_eps": 0.2, "lr": 3e-4, "value_coef": 0.5,
                 "entropy_coef": 0.0, "max_grad_norm": 0.5, "target_kl_stop": None},
        "paths": {"results_dir": str(tmp_path / "results"),
                   "assets_dir": str(tmp_path / "assets"),
                   "checkpoints_dir": str(tmp_path / "saved_models"),
                   "wiki_dir": str(tmp_path / "wiki")},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def _run_and_get_rewards(config: Path) -> list[float]:
    sdk = ProximalLab(config_path=config)
    result = sdk.train_ppo()
    return [d.mean_episode_reward for d in result.diagnostics]


def test_same_seed_produces_identical_rewards(sdk_config: Path) -> None:
    rewards_a = _run_and_get_rewards(sdk_config)
    rewards_b = _run_and_get_rewards(sdk_config)
    np.testing.assert_allclose(rewards_a, rewards_b, atol=1e-5,
                                err_msg="same-seed PPO diverged — RNG not fully seeded")


def _run_and_get_kls(config: Path) -> list[float]:
    sdk = ProximalLab(config_path=config)
    result = sdk.train_ppo()
    return [d.mean_kl for d in result.diagnostics]


def test_same_seed_produces_identical_kl_trajectory(sdk_config: Path) -> None:
    """KL trajectory is the most sensitive diagnostic to RNG state.

    Each call constructs a fresh SDK (which re-seeds the global RNG)
    before training — that's the documented reproducibility contract.
    Running two SDKs in parallel before training is NOT supported;
    the second would see an advanced RNG state.
    """
    kls_a = _run_and_get_kls(sdk_config)
    kls_b = _run_and_get_kls(sdk_config)
    np.testing.assert_allclose(kls_a, kls_b, atol=1e-5)
