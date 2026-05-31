"""Shared infrastructure for experiment services (Layers 9 + 13).

Module-level helpers so both ``ExperimentService`` (original differentiators)
and ``ExperimentStudies`` (audit-driven additions) re-use the same
config-override + SDK-construction + history-summarisation logic.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.shared.types import EpisodeMetrics


def write_temp_cfg(merged: dict) -> Path:
    """Write a merged config dict to a temp file and return its path."""
    tmp = NamedTemporaryFile(mode="w", suffix=".json", delete=False)  # noqa: SIM115
    tmp.write(json.dumps(merged))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def make_sdk(base_cfg: dict, overrides: dict | None = None) -> FitnessRL:
    """Build an SDK against ``base_cfg`` with optional section overrides."""
    merged = deepcopy(base_cfg)
    for section, vals in (overrides or {}).items():
        if isinstance(vals, dict):
            merged.setdefault(section, {}).update(vals)
        else:
            merged[section] = vals
    return FitnessRL(config_path=write_temp_cfg(merged))


def train_one(
    base_cfg: dict,
    algo: str,
    episodes: int,
    mask_enabled: bool = False,
    extra_overrides: dict | None = None,
    with_world_model: bool = False,
) -> list[EpisodeMetrics]:
    """Spin up an SDK with overrides, optionally train the LSTM, then train the algo."""
    overrides: dict = {"env": {"action_masking_enabled": bool(mask_enabled)}}
    if extra_overrides is not None:
        for k, v in extra_overrides.items():
            if isinstance(v, dict):
                overrides.setdefault(k, {}).update(v)
            else:
                overrides[k] = v
    sdk = make_sdk(base_cfg, overrides=overrides)
    sdk.prepare_data()
    if with_world_model:
        sdk.train_world_model()
    if algo == "reinforce":
        return sdk.train_reinforce(episodes)
    return sdk.train_a2c(episodes)


def summarise_history(history: list[EpisodeMetrics]) -> dict:
    """Per-episode aggregates used as the JSON unit across all experiments."""
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
        "final_30pct_mean": float(rewards[-max(1, int(0.3 * rewards.size)):].mean()),
        "action_distribution": (
            (counts / total).tolist() if total > 0 else [0.0] * counts.size
        ),
    }


def aggregate_with_ci(reports: list[dict], z: float = 1.96) -> dict:
    """Mean ± 95 % normal-approx CI on final_30pct_mean across seeds."""
    finals = np.array([r["final_30pct_mean"] for r in reports], dtype=np.float64)
    means = np.array([r["mean_reward"] for r in reports], dtype=np.float64)
    n = finals.size
    se = float(finals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n_seeds": int(n),
        "final_30pct_mean_avg": float(finals.mean()),
        "final_30pct_mean_ci": z * se,
        "overall_mean_avg": float(means.mean()),
        "individual_reports": reports,
    }
