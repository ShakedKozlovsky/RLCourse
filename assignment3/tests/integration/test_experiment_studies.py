"""Layer-13 ExperimentStudies — multi-seed, entropy, chain, gamma, masking-on-LSTM."""

from __future__ import annotations

from pathlib import Path

import pytest

from fitness_rl.services.experiment_base import aggregate_with_ci
from fitness_rl.services.experiment_studies import ExperimentStudies


def test_invalid_episodes_raises(sdk_config: Path) -> None:
    with pytest.raises(ValueError):
        ExperimentStudies(config_path=sdk_config, episodes=0)


def test_multi_seed_comparison_returns_ci_per_algo(sdk_config: Path) -> None:
    studies = ExperimentStudies(config_path=sdk_config, episodes=2)
    result = studies.multi_seed_comparison(seeds=(0, 1))
    assert set(result.keys()) == {"reinforce", "a2c"}
    for algo_result in result.values():
        assert algo_result["n_seeds"] == 2
        assert "final_30pct_mean_avg" in algo_result
        assert "final_30pct_mean_ci" in algo_result
        assert algo_result["final_30pct_mean_ci"] >= 0
        assert len(algo_result["individual_reports"]) == 2


def test_entropy_sweep_returns_one_cell_per_bonus(sdk_config: Path) -> None:
    studies = ExperimentStudies(config_path=sdk_config, episodes=2)
    result = studies.entropy_sweep(bonuses=(0.0, 0.5))
    assert set(result.keys()) == {"entropy=0.0", "entropy=0.5"}
    for entry in result.values():
        assert entry["n_episodes"] == 2


def test_reinforce_chain_returns_three_variants(sdk_config: Path) -> None:
    studies = ExperimentStudies(config_path=sdk_config, episodes=2)
    result = studies.reinforce_variant_chain()
    assert set(result.keys()) == {"no_baseline", "mean_baseline",
                                    "state_value_baseline_a2c"}


def test_gamma_ablation_returns_one_cell_per_gamma(sdk_config: Path) -> None:
    studies = ExperimentStudies(config_path=sdk_config, episodes=2)
    result = studies.gamma_ablation(gammas=(0.9, 0.99))
    assert set(result.keys()) == {"gamma=0.9", "gamma=0.99"}


def test_masking_on_lstm_env_runs_four_cells(sdk_config: Path) -> None:
    studies = ExperimentStudies(config_path=sdk_config, episodes=2)
    result = studies.masking_on_lstm_env()
    assert set(result.keys()) == {
        "reinforce_mask_off", "reinforce_mask_on",
        "a2c_mask_off", "a2c_mask_on",
    }


def test_aggregate_with_ci_single_seed_returns_zero_ci() -> None:
    reports = [{"final_30pct_mean": 1.0, "mean_reward": 1.0}]
    agg = aggregate_with_ci(reports)
    assert agg["n_seeds"] == 1
    assert agg["final_30pct_mean_ci"] == 0.0


def test_aggregate_with_ci_two_seeds_positive_ci() -> None:
    reports = [
        {"final_30pct_mean": 1.0, "mean_reward": 1.0},
        {"final_30pct_mean": 3.0, "mean_reward": 2.0},
    ]
    agg = aggregate_with_ci(reports)
    assert agg["final_30pct_mean_avg"] == pytest.approx(2.0)
    assert agg["final_30pct_mean_ci"] > 0
