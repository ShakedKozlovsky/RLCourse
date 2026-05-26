"""TradingSDK — the single entry point all consumers use.

GUI, CLI, notebooks, and tests must go through this facade rather than
poking at services directly. This is the architecture's load-bearing rule
(PLAN.md §1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dqn_trader.environment.trading_env import TradingEnv
from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.backtest_service import (
    BacktestResult,
    BacktestService,
    save_backtest,
)
from dqn_trader.services.data_service import DataService, PipelineOutput
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.inference_service import Decision, InferenceService
from dqn_trader.services.training_service import EpisodeMetrics, TrainingService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.seed import set_global_seed
from dqn_trader.shared.types import SliceData


@dataclass(frozen=True)
class TrainResult:
    """What a `train` call hands back to the consumer."""

    metrics: list[EpisodeMetrics]
    run_dir: Path
    pipeline: PipelineOutput


class TradingSDK:
    """High-level orchestration over the data → train → backtest → infer flow."""

    def __init__(self, config: ConfigManager | None = None, *, device: str = "cpu"):
        self._cfg = config or ConfigManager()
        self._device = device
        set_global_seed(int(self._cfg.get("seed", 0)))

    @property
    def config(self) -> ConfigManager:
        """Read-only access to the loaded ConfigManager."""
        return self._cfg

    def prepare_data(self, ticker: str | None = None) -> PipelineOutput:
        """Run the full data pipeline and return windowed train/val/test tensors."""
        return DataService(self._cfg).run(ticker)

    def train(self, ticker: str | None = None, *, pipeline: PipelineOutput | None = None) -> TrainResult:
        """Train a DQN agent end-to-end and return metrics + run directory."""
        pipeline = pipeline or self.prepare_data(ticker)
        service = TrainingService(self._cfg, pipeline, device=self._device)
        metrics, run = service.fit()
        return TrainResult(metrics=metrics, run_dir=run.root, pipeline=pipeline)

    def backtest(
        self,
        checkpoint: Path,
        *,
        slice_name: str = "test",
        pipeline: PipelineOutput | None = None,
        report_name: str | None = None,
    ) -> BacktestResult:
        """Evaluate a trained checkpoint on a slice and save the report."""
        pipeline = pipeline or self.prepare_data()
        slc: SliceData = getattr(pipeline, slice_name)
        env = self._build_env(slc)
        agent = self._build_agent_and_load(checkpoint)
        result = BacktestService(env, agent).run()
        out_dir = Path(self._cfg.get("backtest.report_dir", "results/backtest"))
        save_backtest(result, out_dir, name=report_name or f"{slc.name}_backtest")
        return result

    def predict(self, market_window: np.ndarray, *, checkpoint: Path, position: int = 0,
                pnl_unrealised_scaled: float = 0.0) -> Decision:
        """Single-step inference: return the recommended action and Q-values."""
        agent = self._build_agent_and_load(checkpoint)
        return InferenceService(agent).decide(market_window, position, pnl_unrealised_scaled)

    def run_experiments(self) -> dict[str, object]:
        """Run all four comparative experiments via ExperimentService."""
        from dqn_trader.services.experiment_service import ExperimentService

        svc = ExperimentService(self._cfg, device=self._device)
        return {
            "dqn_vs_dueling": svc.run_dqn_vs_dueling(),
            "uniform_vs_per": svc.run_uniform_vs_per(),
            "reward_variants": svc.run_reward_variants(),
            "cross_ticker": svc.run_cross_ticker(),
        }

    def _build_env(self, slc: SliceData) -> TradingEnv:
        env_cfg: dict[str, Any] = self._cfg.setup["env"]
        return TradingEnv(
            slc,
            initial_capital=env_cfg["initial_capital"],
            alpha=env_cfg["transaction_cost_alpha"],
            beta=env_cfg["slippage_beta"],
            reward=env_cfg["reward_variant"],
            sharpe_gamma=env_cfg.get("sharpe_bonus_gamma", 1.0),
            sharpe_window=env_cfg.get("sharpe_window", 20),
            invalid_action_penalty=env_cfg.get("invalid_action_penalty", 0.0),
        )

    def _build_agent_and_load(self, checkpoint: Path) -> DQNAgent:
        agent_cfg = self._cfg.setup["agent"]
        agent = DQNAgent(
            window_size=int(self._cfg.get("data.window_size")),
            n_features=int(self._cfg.get("data.features")),
            n_actions=3,
            replay=UniformReplay(capacity=2),  # buffer not used in eval
            gamma=agent_cfg["gamma"],
            lr=agent_cfg["lr"],
            dueling=agent_cfg["dueling"],
            double_dqn=agent_cfg["double_dqn"],
            device=self._device,
        )
        agent.load(str(checkpoint))
        return agent
