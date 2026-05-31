"""Three differentiator experiments for the README's analysis section:

1. Action-masking ablation — REINFORCE + A2C with masking on/off
2. Reward-weight sweep — vary λ_1 (overload) and λ_2 (imbalance)
3. Action-distribution collapse detection — flag policies that collapse

Each method returns a JSON-serialisable dict so the README can embed it
directly and the GUI/CLI can dump it to disk.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.evaluation_service import EvaluationService
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.types import EpisodeMetrics

_logger = get_logger(__name__)


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
                history = self._train_one(algo=algo, mask_enabled=masking)
                results[key] = self._summarise_history(history)
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
                history = self._train_one(algo="a2c", extra_overrides=overrides)
                results[key] = self._summarise_history(history)
        return results

    def run_collapse_analysis(self) -> dict:
        """Train both algos, then evaluate; report whether either collapsed."""
        out: dict[str, dict] = {}
        for algo in ("reinforce", "a2c"):
            sdk = self._make_sdk()
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

    def _train_one(
        self,
        algo: str,
        mask_enabled: bool = False,
        extra_overrides: dict | None = None,
    ) -> list[EpisodeMetrics]:
        overrides: dict = {"env": {"action_masking_enabled": bool(mask_enabled)}}
        if extra_overrides is not None:
            for k, v in extra_overrides.items():
                overrides.setdefault(k, {}).update(v)
        sdk = self._make_sdk(overrides=overrides)
        sdk.prepare_data()
        if algo == "reinforce":
            return sdk.train_reinforce(self._episodes)
        return sdk.train_a2c(self._episodes)

    def _make_sdk(self, overrides: dict | None = None) -> FitnessRL:
        """Build a FitnessRL using a temp config file with optional overrides."""
        merged = deepcopy(self._base_cfg)
        for section, vals in (overrides or {}).items():
            merged.setdefault(section, {}).update(vals)
        tmp = NamedTemporaryFile(  # noqa: SIM115 - kept alive via path
            mode="w", suffix=".json", delete=False)
        tmp.write(json.dumps(merged))
        tmp.flush()
        tmp.close()
        return FitnessRL(config_path=Path(tmp.name))

    @staticmethod
    def _summarise_history(history: list[EpisodeMetrics]) -> dict:
        rewards = np.array([m.total_reward for m in history], dtype=np.float64)
        counts = np.zeros_like(history[0].action_counts, dtype=np.float64)
        for m in history:
            counts += m.action_counts
        total = float(counts.sum())
        return {
            "n_episodes": int(rewards.size),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()) if rewards.size > 1 else 0.0,
            "final_reward": float(rewards[-1]),
            "action_distribution": (
                (counts / total).tolist() if total > 0
                else [0.0] * counts.size
            ),
        }
