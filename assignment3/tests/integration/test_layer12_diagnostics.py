"""Layer 12 integration tests: WorldModelEvaluator + BaselinePolicies +
qualitative rollout, exposed through ``FitnessRLEvaluator``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitness_rl.sdk.evaluator import FitnessRLEvaluator
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.baseline_policies import (
    BenchmarkResult,
    KaggleProgramPolicy,
    PolicyBenchmark,
    RandomPolicy,
    RoundRobinPolicy,
)
from fitness_rl.services.world_model_evaluator import WorldModelEvaluator


def test_world_model_evaluator_returns_finite_mses(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_world_model()
    # Use horizon 1 only — the 28-day fixture trajectory split 80/20 leaves a
    # 5–6-state test slice; window_size=4 + horizon=3 needs 7 states.
    report = FitnessRLEvaluator(sdk).evaluate_world_model(horizons=(1,))
    assert np.isfinite(report.persistence_one_step_mse)
    assert np.isfinite(report.linear_one_step_mse)
    assert np.isfinite(report.lstm_one_step_mse)
    assert 1 in report.lstm_rollout_mse
    assert report.lstm_rollout_mse[1] >= 0


def test_world_model_evaluator_requires_trained_model(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    with pytest.raises(RuntimeError):
        FitnessRLEvaluator(sdk).evaluate_world_model()


def test_baselines_return_three_results(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    results = FitnessRLEvaluator(sdk).benchmark_baselines()
    assert len(results) == 3
    names = {r.name for r in results}
    assert names == {"random", "round_robin", "kaggle_program"}
    for r in results:
        assert isinstance(r, BenchmarkResult)
        assert sum(r.action_distribution) == pytest.approx(1.0, abs=1e-6)


def test_round_robin_actions_cycle() -> None:
    p = RoundRobinPolicy(cycle=(0, 1, 2, 3))
    assert [p.select_action(np.zeros(16), [], i) for i in range(8)] == [0, 1, 2, 3] * 2


def test_kaggle_program_wraps_modulo_length() -> None:
    p = KaggleProgramPolicy(actions=np.array([2, 4, 1], dtype=np.int64))
    assert [p.select_action(np.zeros(16), [], i) for i in range(6)] == [2, 4, 1, 2, 4, 1]


def test_random_policy_uses_provided_rng() -> None:
    a = RandomPolicy(rng=np.random.default_rng(7))
    b = RandomPolicy(rng=np.random.default_rng(7))
    seq_a = [a.select_action(np.zeros(16), [], i) for i in range(20)]
    seq_b = [b.select_action(np.zeros(16), [], i) for i in range(20)]
    assert seq_a == seq_b


def test_qualitative_rollout_returns_step_table(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    sdk.train_a2c()
    traj = FitnessRLEvaluator(sdk).qualitative_rollout(algo="a2c")
    assert len(traj.steps) > 0
    table = traj.as_table()
    assert "action" in table  # header
    assert "Total reward" in table
    # Each step should have a known action name
    for step in traj.steps:
        assert step.action_name in {"PUSH", "PULL", "LEGS", "CARDIO", "REST"}


def test_world_model_evaluator_invalid_init() -> None:
    with pytest.raises(ValueError):
        WorldModelEvaluator(window_size=0)
    with pytest.raises(ValueError):
        WorldModelEvaluator(test_pct=0.0)


def test_baseline_round_robin_empty_cycle_rejected() -> None:
    with pytest.raises(ValueError):
        RoundRobinPolicy(cycle=())


def test_kaggle_program_rejects_non_1d() -> None:
    with pytest.raises(ValueError):
        KaggleProgramPolicy(actions=np.zeros((3, 3), dtype=np.int64))


def test_policy_benchmark_runs_full_episode(sdk_config: Path) -> None:
    sdk = FitnessRL(config_path=sdk_config)
    sdk.prepare_data()
    env = sdk.make_env()
    result = PolicyBenchmark.run(RandomPolicy(rng=np.random.default_rng(0)), env)
    assert len(result.rewards) == int(sdk.config.get("env.episode_length"))
