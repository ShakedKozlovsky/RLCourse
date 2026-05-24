"""Run all three excellence differentiators and generate their plots.

1. Window-size sensitivity sweep (10, 20, 30, 50)
2. Per-episode action distribution analysis
3. Q-value heatmap over the test slice

Usage:
    uv run python scripts/run_differentiators.py
"""

from __future__ import annotations

from pathlib import Path

from plot_differentiators import (
    plot_action_distribution,
    plot_qvalue_heatmap,
    plot_window_sweep,
)

from dqn_trader.environment.trading_env import TradingEnv
from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.analysis_service import (
    collect_action_distribution,
    collect_qvalue_heatmap,
)
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.experiment_service import ExperimentService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger

RESULTS = Path("results")
_logger = get_logger("differentiators")


def run_window_sweep(cfg: ConfigManager) -> None:
    """Train with window_size in {10, 20, 30, 50} and compare test metrics."""
    _logger.info("=== window-size sensitivity sweep ===")
    exp = ExperimentService(cfg)
    overrides = {f"window_{ws}": {"data.window_size": ws} for ws in (10, 20, 30, 50)}
    result = exp._compare("window_sensitivity", overrides)
    plot_window_sweep(result)
    _logger.info("window sweep done")


def run_action_distribution(cfg: ConfigManager) -> None:
    """Analyse action distribution of the latest trained checkpoint."""
    _logger.info("=== action distribution analysis ===")
    ckpts = sorted(RESULTS.rglob("best.pt"))
    if not ckpts:
        _logger.warning("no checkpoint — skipping action distribution")
        return
    from dqn_trader.sdk.sdk import TradingSDK

    sdk = TradingSDK(cfg)
    pipeline = sdk.prepare_data()
    env_cfg = cfg.setup["env"]
    train_env = _make_env(pipeline.train, env_cfg)
    test_env = _make_env(pipeline.test, env_cfg)
    agent = _load_agent(cfg, ckpts[-1])
    train_dist = collect_action_distribution(agent, train_env, epsilon=0.0)
    test_dist = collect_action_distribution(agent, test_env, epsilon=0.0)
    plot_action_distribution(train_dist, test_dist)
    _logger.info("action distribution done")


def run_qvalue_heatmap(cfg: ConfigManager) -> None:
    """Plot Q-values over the test slice for the latest checkpoint."""
    _logger.info("=== Q-value heatmap ===")
    ckpts = sorted(RESULTS.rglob("best.pt"))
    if not ckpts:
        _logger.warning("no checkpoint — skipping Q-value heatmap")
        return
    from dqn_trader.sdk.sdk import TradingSDK

    sdk = TradingSDK(cfg)
    pipeline = sdk.prepare_data()
    env_cfg = cfg.setup["env"]
    test_env = _make_env(pipeline.test, env_cfg)
    agent = _load_agent(cfg, ckpts[-1])
    snap = collect_qvalue_heatmap(agent, test_env)
    plot_qvalue_heatmap(snap)
    _logger.info("Q-value heatmap done")


def _make_env(slc, env_cfg: dict) -> TradingEnv:
    """Build a TradingEnv from a SliceData and env config dict."""
    return TradingEnv(
        slc,
        initial_capital=env_cfg["initial_capital"],
        alpha=env_cfg["transaction_cost_alpha"],
        beta=env_cfg["slippage_beta"],
    )


def _load_agent(cfg: ConfigManager, ckpt: Path) -> DQNAgent:
    """Build a DQNAgent and load weights from a checkpoint."""
    ac = cfg.setup["agent"]
    agent = DQNAgent(
        window_size=int(cfg.get("data.window_size")),
        n_features=int(cfg.get("data.features")),
        n_actions=3,
        replay=UniformReplay(capacity=2),
        gamma=ac["gamma"],
        lr=ac["lr"],
        dueling=ac["dueling"],
        double_dqn=ac["double_dqn"],
    )
    agent.load(str(ckpt))
    return agent


def main() -> None:
    """Run all three differentiators."""
    cfg = ConfigManager()
    run_action_distribution(cfg)
    run_qvalue_heatmap(cfg)
    run_window_sweep(cfg)


if __name__ == "__main__":
    main()
