"""Empirical sweeps: λ-sweep, γ-sweep, clip-ε sweep with multi-seed CIs.

Each cell trains one PPO config × ``n_seeds`` and aggregates with
``aggregate_with_ci``. JSON output is consumed by Layer 14's plots +
README headline table.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.services.comparison_service import ComparisonReport, ComparisonService
from proximal_lab.shared.logger import get_logger

_logger = get_logger(__name__)


class ExperimentService:
    """Run a sweep over one hyperparameter, aggregating across seeds."""

    def __init__(
        self,
        sdk: ProximalLab,
        timesteps_per_cell: int = 20000,
        n_seeds: int = 1,
        steps_per_rollout: int = 1024,
    ):
        if timesteps_per_cell < 1 or n_seeds < 1:
            raise ValueError("timesteps_per_cell and n_seeds must be >= 1")
        self._sdk = sdk
        self._ts = int(timesteps_per_cell)
        self._n_seeds = int(n_seeds)
        self._steps = int(steps_per_rollout)
        self._base_cfg: dict = dict(sdk.config.setup)

    def run_lambda_sweep(self, env_id: str | None = None) -> ComparisonReport:
        lambdas = list(self._sdk.config.get("experiments.lambda_sweep", [0.0, 0.95, 1.0]))
        return self._run_cells(
            label="lambda_sweep", env_id=env_id, cells={
                f"lambda={v}": {"gae": {"lambda": float(v)}} for v in lambdas
            },
        )

    def run_gamma_sweep(self, env_id: str | None = None) -> ComparisonReport:
        gammas = list(self._sdk.config.get("experiments.gamma_sweep", [0.99]))
        return self._run_cells(
            label="gamma_sweep", env_id=env_id, cells={
                f"gamma={v}": {"env": {"gamma": float(v)}} for v in gammas
            },
        )

    def run_clip_eps_sweep(self, env_id: str | None = None) -> ComparisonReport:
        eps = list(self._sdk.config.get("experiments.clip_eps_sweep", [0.2]))
        return self._run_cells(
            label="clip_eps_sweep", env_id=env_id, cells={
                f"clip_eps={v}": {"ppo": {"clip_eps": float(v)}} for v in eps
            },
        )

    def _run_cells(
        self, label: str, env_id: str | None, cells: dict[str, dict],
    ) -> ComparisonReport:
        per_cell_finals: dict[str, list[float]] = {}
        for cell_name, overrides in cells.items():
            seed_finals = []
            for seed in range(self._n_seeds):
                merged = deepcopy(self._base_cfg)
                for section, vals in overrides.items():
                    merged.setdefault(section, {}).update(vals)
                merged["seed"] = int(seed)
                tmp_cfg = _write_temp_cfg(merged)
                sub_sdk = ProximalLab(config_path=tmp_cfg)
                result = sub_sdk.train_ppo(
                    env_id=env_id, total_timesteps=self._ts,
                    steps_per_rollout=self._steps, seed=seed,
                )
                seed_finals.append(float(result.final_mean_reward))
                _logger.info(
                    "  cell=%s seed=%d final=%.2f", cell_name, seed,
                    result.final_mean_reward,
                )
            per_cell_finals[cell_name] = seed_finals
        return ComparisonService().report(label, per_cell_finals)

    @staticmethod
    def save(report: ComparisonReport, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2))


def _write_temp_cfg(payload: dict) -> Path:
    tmp = NamedTemporaryFile(mode="w", suffix=".json", delete=False)  # noqa: SIM115
    tmp.write(json.dumps(payload))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def aggregate_seeds(finals: list[float]) -> dict[str, float]:
    """Mean ± std + final reward summary; exposed for tests."""
    arr = np.array(finals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "n_seeds": int(arr.size),
    }
