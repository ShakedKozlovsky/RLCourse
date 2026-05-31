"""ExperimentService — masking ablation, reward-weight sweep, collapse analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from fitness_rl.services.experiment_service import ExperimentService


def test_invalid_episodes_raises(sdk_config: Path) -> None:
    with pytest.raises(ValueError):
        ExperimentService(config_path=sdk_config, episodes=0)


def test_action_masking_ablation_runs(sdk_config: Path) -> None:
    svc = ExperimentService(config_path=sdk_config, episodes=2)
    result = svc.run_action_masking_ablation()
    assert set(result.keys()) == {
        "reinforce_mask_off", "reinforce_mask_on",
        "a2c_mask_off", "a2c_mask_on",
    }
    for key, entry in result.items():
        assert entry["n_episodes"] == 2, key
        assert isinstance(entry["mean_reward"], float)
        assert len(entry["action_distribution"]) == 5
        assert sum(entry["action_distribution"]) == pytest.approx(1.0, abs=1e-6)


def test_reward_weight_sweep_runs(sdk_config: Path) -> None:
    svc = ExperimentService(config_path=sdk_config, episodes=2)
    result = svc.run_reward_weight_sweep(
        overload_lambdas=(0.0, 0.2),
        imbalance_lambdas=(0.0, 0.3),
    )
    assert len(result) == 4  # 2 × 2 grid
    for entry in result.values():
        assert entry["n_episodes"] == 2
        assert "final_reward" in entry


def test_collapse_analysis_runs(sdk_config: Path) -> None:
    svc = ExperimentService(config_path=sdk_config, episodes=2)
    result = svc.run_collapse_analysis()
    assert set(result.keys()) == {"reinforce", "a2c"}
    for entry in result.values():
        assert isinstance(entry["collapsed"], bool)
        assert 0.0 <= entry["max_action_frac"] <= 1.0
        assert len(entry["action_distribution"]) == 5
