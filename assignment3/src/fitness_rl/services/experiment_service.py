"""Layer-9 differentiator experiments (originals):

1. Action-masking ablation — REINFORCE + A2C with masking on/off
2. Reward-weight sweep — vary λ_1 (overload) and λ_2 (imbalance)
3. Action-distribution collapse detection — flag policies that collapse

Layer-13 audit-driven experiments live in :class:`ExperimentStudies`.
"""

from __future__ import annotations

import json
from pathlib import Path

from fitness_rl.services.evaluation_service import EvaluationService
from fitness_rl.services.experiment_base import (
    make_sdk,
    summarise_history,
    train_one,
)


class ExperimentService:
    """Run the three Layer-9 experiments against a config-driven base setup."""

    def __init__(self, config_path: Path, episodes: int = 30):
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        self._config_path = Path(config_path)
        self._episodes = int(episodes)
        self._base_cfg: dict = json.loads(self._config_path.read_text())

    def run_action_masking_ablation(self) -> dict:
        """Train REINFORCE + A2C four ways: {algo} × {masking on/off}."""
        results: dict[str, dict] = {}
        for algo in ("reinforce", "a2c"):
            for masking in (False, True):
                key = f"{algo}_mask_{'on' if masking else 'off'}"
                history = train_one(
                    self._base_cfg, algo=algo, episodes=self._episodes,
                    mask_enabled=masking,
                )
                results[key] = summarise_history(history)
        return results

    def run_reward_weight_sweep(
        self,
        overload_lambdas: tuple[float, ...] = (0.0, 0.2, 0.5),
        imbalance_lambdas: tuple[float, ...] = (0.0, 0.3, 0.6),
    ) -> dict:
        """Sweep over `(λ_overload, λ_imbalance)` pairs with A2C as the trainer."""
        results: dict[str, dict] = {}
        for lo in overload_lambdas:
            for li in imbalance_lambdas:
                key = f"overload={lo}_imbalance={li}"
                overrides = {"env": {"reward_overload_lambda": float(lo),
                                     "reward_imbalance_lambda": float(li)}}
                history = train_one(
                    self._base_cfg, algo="a2c", episodes=self._episodes,
                    extra_overrides=overrides,
                )
                results[key] = summarise_history(history)
        return results

    def run_collapse_analysis(self) -> dict:
        """Train both algos, then evaluate; report whether either collapsed."""
        out: dict[str, dict] = {}
        for algo in ("reinforce", "a2c"):
            sdk = make_sdk(self._base_cfg)
            sdk.prepare_data()
            history = (sdk.train_reinforce(self._episodes) if algo == "reinforce"
                       else sdk.train_a2c(self._episodes))
            eval_result = sdk.evaluate(algo=algo)
            dist = EvaluationService.action_distribution(eval_result)
            out[algo] = {
                "collapsed": EvaluationService.collapsed(eval_result, threshold=0.8),
                "max_action_frac": float(dist.max()),
                "action_distribution": dist.tolist(),
                "final_reward": float(history[-1].total_reward),
            }
        return out
