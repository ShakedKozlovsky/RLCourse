"""Layer-13 audit-driven experiments (additions to Layer-9 originals).

* multi_seed_comparison(seeds) — audit #3 + #18
* entropy_sweep(bonuses)      — audit #2 (fix A2C collapse)
* reinforce_variant_chain()   — audit #7 (no-baseline → mean → state-value)
* gamma_ablation(gammas)      — audit #14
* masking_on_lstm_env()       — audit #6 (mask vs LSTM dynamics, not identity)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.experiment_base import (
    aggregate_with_ci,
    summarise_history,
    train_one,
    write_temp_cfg,
)


class ExperimentStudies:
    """Container for the five audit-driven experiments."""

    def __init__(self, config_path: Path, episodes: int = 30):
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        self._episodes = int(episodes)
        self._base_cfg: dict = json.loads(Path(config_path).read_text())

    def multi_seed_comparison(self, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
        """Audit #3 + #18 — REINFORCE & A2C across seeds, 95 % CI on final reward."""
        runs: dict[str, list[dict]] = {"reinforce": [], "a2c": []}
        for seed in seeds:
            for algo in ("reinforce", "a2c"):
                merged = deepcopy(self._base_cfg)
                merged["seed"] = int(seed)
                sdk = FitnessRL(config_path=write_temp_cfg(merged))
                sdk.prepare_data()
                hist = (sdk.train_reinforce(self._episodes) if algo == "reinforce"
                        else sdk.train_a2c(self._episodes))
                runs[algo].append(summarise_history(hist))
        return {algo: aggregate_with_ci(reports) for algo, reports in runs.items()}

    def entropy_sweep(self, bonuses: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5)) -> dict:
        """Audit #2 — A2C with increasing entropy bonus to fight policy collapse."""
        results: dict[str, dict] = {}
        for b in bonuses:
            history = train_one(
                self._base_cfg, algo="a2c", episodes=self._episodes,
                extra_overrides={"a2c": {"entropy_bonus": float(b)}},
            )
            results[f"entropy={b}"] = summarise_history(history)
        return results

    def reinforce_variant_chain(self) -> dict:
        """Audit #7 — REINFORCE → +mean baseline → +state-value baseline (A2C)."""
        return {
            "no_baseline": summarise_history(train_one(
                self._base_cfg, algo="reinforce", episodes=self._episodes,
                extra_overrides={"reinforce": {"use_baseline": False}},
            )),
            "mean_baseline": summarise_history(train_one(
                self._base_cfg, algo="reinforce", episodes=self._episodes,
                extra_overrides={"reinforce": {"use_baseline": True}},
            )),
            "state_value_baseline_a2c": summarise_history(train_one(
                self._base_cfg, algo="a2c", episodes=self._episodes,
            )),
        }

    def gamma_ablation(self, gammas: tuple[float, ...] = (0.9, 0.95, 0.99, 0.999)) -> dict:
        """Audit #14 — vary the discount factor for A2C."""
        results: dict[str, dict] = {}
        for g in gammas:
            history = train_one(
                self._base_cfg, algo="a2c", episodes=self._episodes,
                extra_overrides={"env": {"gamma": float(g)}},
            )
            results[f"gamma={g}"] = summarise_history(history)
        return results

    def masking_on_lstm_env(self) -> dict:
        """Audit #6 — masking ablation with the trained LSTM as the env dynamics."""
        results: dict[str, dict] = {}
        for algo in ("reinforce", "a2c"):
            for mask in (False, True):
                history = train_one(
                    self._base_cfg, algo=algo, episodes=self._episodes,
                    mask_enabled=mask, with_world_model=True,
                )
                results[f"{algo}_mask_{'on' if mask else 'off'}"] = summarise_history(history)
        return results
