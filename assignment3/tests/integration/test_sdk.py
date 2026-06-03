"""End-to-end SDK smoke tests on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.comparison_service import ComparisonResult
from fitness_rl.services.evaluation_service import EvaluationResult
from fitness_rl.shared.types import Action


def test_prepare_data_returns_pipeline_output(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    out = sdk.prepare_data()
    assert out.chosen_title == "Program Match"
    assert out.states.shape[1] == 16


def test_train_world_model_runs_and_saves_checkpoint(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    result = sdk.train_world_model()
    assert len(result.train_losses) >= 1
    ckpt = Path(sdk.config.path("checkpoints_dir")) / "world_model.pt"
    assert ckpt.exists()


def test_train_reinforce_returns_history(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    history = sdk.train_reinforce()
    assert len(history) == 3  # from sdk_config


def test_train_a2c_returns_history(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    history = sdk.train_a2c()
    assert len(history) == 3


def test_compare_requires_both_trained(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_reinforce()
    with pytest.raises(RuntimeError):
        sdk.compare()  # a2c not trained yet
    sdk.train_a2c()
    result = sdk.compare()
    assert isinstance(result, ComparisonResult)
    assert result.winner in {"reinforce", "a2c", "tie"}


def test_evaluate_returns_eval_result(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_a2c()
    result = sdk.evaluate(algo="a2c")
    assert isinstance(result, EvaluationResult)
    assert len(result.action_sequence) > 0


def test_predict_returns_valid_action(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_reinforce()
    state = np.zeros(16, dtype=np.float32)
    action = sdk.predict(state, algo="reinforce")
    assert 0 <= action < Action.n()


def test_predict_unknown_algo_raises(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_a2c()
    with pytest.raises(ValueError):
        sdk.predict(np.zeros(16, dtype=np.float32), algo="dqn")


def test_train_ppo_returns_history(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    history = sdk.train_ppo()
    # episodes default falls back to a2c.episodes = 3 from sdk_config
    assert len(history) == 3


def test_predict_with_ppo(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_ppo()
    action = sdk.predict(np.zeros(16, dtype=np.float32), algo="ppo")
    assert 0 <= action < 5


def test_predict_untrained_raises(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    with pytest.raises(RuntimeError):
        sdk.predict(np.zeros(16, dtype=np.float32), algo="reinforce")


def test_full_pipeline_with_world_model(sdk_config: Path) -> None:
    """Train the LSTM, then use it as the env's transition fn for REINFORCE."""
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_world_model()
    history = sdk.train_reinforce()
    assert len(history) == 3
