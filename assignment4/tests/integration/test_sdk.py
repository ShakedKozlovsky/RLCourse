"""End-to-end SDK smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proximal_lab.sdk.sdk import ProximalLab


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    """Tiny config so smoke tests finish in seconds."""
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "secondary_id": "Walker2d-v5",
                 "max_episode_steps": 1000, "n_parallel_envs": 2, "gamma": 0.99},
        "gae": {"lambda": 0.95},
        "actor_critic": {"hidden_sizes": [32, 32], "activation": "tanh",
                          "shared_trunk": False,
                          "log_std_init": -0.5, "log_std_min": -5.0, "log_std_max": 2.0},
        "ppo": {"total_timesteps": 512, "steps_per_rollout": 256,
                 "minibatch_size": 32, "n_epochs_per_update": 2,
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


def test_sdk_constructs(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    assert sdk.config is not None
    assert sdk.net is None


def test_sdk_train_ppo_returns_result(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    result = sdk.train_ppo()
    assert result.total_timesteps >= 512
    assert sdk.net is not None
    assert sdk.train_result is result


def test_sdk_evaluate_after_training(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    sdk.train_ppo()
    result = sdk.evaluate(n_episodes=1)
    assert result.n_episodes == 1


def test_sdk_evaluate_without_training_raises(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    with pytest.raises(RuntimeError):
        sdk.evaluate()


def test_sdk_predict_returns_action(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    sdk.train_ppo()
    obs = np.zeros(17, dtype=np.float32)
    action = sdk.predict(obs)
    assert action.shape == (6,)


def test_sdk_predict_without_training_raises(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    with pytest.raises(RuntimeError):
        sdk.predict(np.zeros(17, dtype=np.float32))


def test_sdk_load_checkpoint_round_trip(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    sdk.train_ppo()
    obs = np.zeros(17, dtype=np.float32)
    action_before = sdk.predict(obs)
    # Reload from disk into a fresh SDK
    sdk2 = ProximalLab(config_path=sdk_config)
    sdk2.load_checkpoint()
    action_after = sdk2.predict(obs)
    np.testing.assert_allclose(action_before, action_after, atol=1e-6)
