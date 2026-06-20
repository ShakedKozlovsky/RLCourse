"""ExperimentService — orchestrates multi-seed sweeps for the headline empirical
study. Lives under `sdk/` per ADR-007 to keep the layering arrow straight."""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.shared.seed import set_global_seed


@dataclass(frozen=True)
class CellResult:
    cell: str
    seed: int
    final_reward: float
    final_coverage: float
    critic_loss_last: float
    final_sigma: float


def _write_temp_cfg(cfg_dict: dict) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg_dict, f, indent=2)
        name = f.name
    return Path(name)


class ExperimentService:
    def __init__(self, lab: RoombaLab, n_seeds: int = 3,
                 total_timesteps: int = 4000) -> None:
        self.lab = lab
        self.n_seeds = n_seeds
        self.total_timesteps = total_timesteps

    def run(self, kind: str) -> Path:
        if kind == "noise_sigma":
            cells = self.lab.config.get("experiments.noise_sigma_sweep")
            patch = lambda cfg, v: cfg["noise"].__setitem__("sigma_initial", float(v)) or cfg["noise"].__setitem__("sigma_final", float(v))  # noqa: E501,E731
        elif kind == "tau":
            cells = self.lab.config.get("experiments.tau_sweep")
            patch = lambda cfg, v: cfg["ddpg"].__setitem__("tau", float(v))  # noqa: E731
        elif kind == "target_network":
            cells = [0.005, 1.0]  # tau=0.005 (soft) vs tau=1.0 (hard at every step)
            patch = lambda cfg, v: cfg["ddpg"].__setitem__("tau", float(v))  # noqa: E731
        else:
            raise ValueError(f"unknown sweep kind: {kind}")
        seeds = [int(s) for s in self.lab.config.get("experiments.ablation_seeds")][: self.n_seeds]
        base_cfg = copy.deepcopy(dict(self.lab.config.setup))
        results: list[CellResult] = []
        for v in cells:
            cfg = copy.deepcopy(base_cfg)
            patch(cfg, v)
            cfg_path = _write_temp_cfg(cfg)
            for s in seeds:
                set_global_seed(s)
                cell_lab = RoombaLab(config_path=cfg_path)
                r = cell_lab.train(total_timesteps=self.total_timesteps, seed=s)
                last = r.diagnostics[-1] if r.diagnostics else None
                results.append(CellResult(
                    cell=str(v), seed=s,
                    final_reward=float(last.episode_reward) if last else 0.0,
                    final_coverage=float(last.coverage) if last else 0.0,
                    critic_loss_last=float(last.critic_loss) if last else 0.0,
                    final_sigma=float(last.sigma) if last else 0.0,
                ))
        out = self.lab.config.path("results_dir") / "sweeps" / f"{kind}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": kind, "n_seeds": self.n_seeds,
                    "total_timesteps": self.total_timesteps,
                    "cells": [r.__dict__ for r in results]}
        out.write_text(json.dumps(payload, indent=2))
        return out

    @staticmethod
    def aggregate(report_path: Path) -> dict[str, dict[str, float]]:
        payload = json.loads(Path(report_path).read_text())
        by_cell: dict[str, list[CellResult]] = {}
        for r in payload["cells"]:
            by_cell.setdefault(r["cell"], []).append(r)
        out: dict[str, dict[str, float]] = {}
        for cell, rs in by_cell.items():
            rewards = np.array([r["final_reward"] for r in rs])
            covs = np.array([r["final_coverage"] for r in rs])
            n = len(rewards)
            out[cell] = {
                "mean_reward": float(rewards.mean()),
                "ci95_reward": float(1.96 * rewards.std(ddof=1) / max(1, np.sqrt(n)))
                                if n > 1 else 0.0,
                "mean_coverage": float(covs.mean()),
                "n_seeds": float(n),
            }
        return out
