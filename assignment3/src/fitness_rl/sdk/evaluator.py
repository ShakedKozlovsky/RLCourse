"""Companion evaluator class — keeps the SDK facade strictly under the 150-LOC cap.

The SDK ``FitnessRL`` owns training; this class owns the after-training
diagnostics added in Layer 12 (audit findings #1 + #4 + #15 + #19).
"""

from __future__ import annotations

import numpy as np

from fitness_rl.environment.reward import RewardFunction
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.baseline_policies import (
    BenchmarkResult,
    KaggleProgramPolicy,
    PolicyBenchmark,
    RandomPolicy,
    RoundRobinPolicy,
)
from fitness_rl.services.diagnostics import GreedyTrajectory, record_greedy_rollout
from fitness_rl.services.world_model_evaluator import WorldModelEvaluator, WorldModelReport


class FitnessRLEvaluator:
    """Post-training diagnostics over a fully or partially trained :class:`FitnessRL`."""

    def __init__(self, sdk: FitnessRL):
        self._sdk = sdk

    def evaluate_world_model(
        self, horizons: tuple[int, ...] = (1, 7, 28)
    ) -> WorldModelReport:
        """Audit #1: persistence + linear vs LSTM, plus multi-step rollout error."""
        if self._sdk.world_model is None:
            raise RuntimeError("world model not trained")
        data = self._sdk.data or self._sdk.prepare_data()
        evaluator = WorldModelEvaluator(
            window_size=int(self._sdk.config.get("world_model.window_size"))
        )
        return evaluator.evaluate(
            self._sdk.world_model, data.states, data.actions, horizons
        )

    def benchmark_baselines(self) -> list[BenchmarkResult]:
        """Audit #4: run random, round-robin, and Kaggle-program policies."""
        data = self._sdk.data or self._sdk.prepare_data()
        rng = np.random.default_rng(int(self._sdk.config.get("seed")))
        baselines = [
            RandomPolicy(rng=rng),
            RoundRobinPolicy(),
            KaggleProgramPolicy(actions=data.actions),
        ]
        return [PolicyBenchmark.run(p, self._sdk.make_env()) for p in baselines]

    def qualitative_rollout(self, algo: str = "a2c") -> GreedyTrajectory:
        """Audit #15: one greedy rollout + per-step reward decomposition."""
        net = self._sdk._require_net(algo)  # noqa: SLF001
        reward_fn = RewardFunction(
            gain_weight=float(self._sdk.config.get("env.reward_gain_weight")),
            overload_lambda=float(self._sdk.config.get("env.reward_overload_lambda")),
            imbalance_lambda=float(self._sdk.config.get("env.reward_imbalance_lambda")),
        )
        return record_greedy_rollout(net, self._sdk.make_env(), reward_fn)
