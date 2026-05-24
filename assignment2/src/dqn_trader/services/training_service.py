"""TrainingService — orchestrates the DQN training loop.

Consumes a ConfigManager and a PipelineOutput from DataService. Builds the
env(s), agent, and replay; runs N episodes; periodically evaluates greedily
on the validation slice; checkpoints the best agent by val return.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dqn_trader.environment.trading_env import TradingEnv
from dqn_trader.memory.prioritized_replay import PrioritizedReplay
from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.data_service import PipelineOutput
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.epsilon_schedule import BetaSchedule, EpsilonSchedule
from dqn_trader.services.run_directory import RunDirectory, create_run
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeMetrics:
    """One row of the per-episode metrics log."""

    episode: int
    reward: float
    loss: float
    epsilon: float
    trades: int
    final_value: float
    val_return: float


class TrainingService:
    """High-level fit() for a DQN agent on the train/val slices of a PipelineOutput."""

    def __init__(self, config: ConfigManager, pipeline: PipelineOutput, *, device: str = "cpu"):
        self._cfg = config
        self._pipeline = pipeline
        self._device = device

    def fit(self) -> tuple[list[EpisodeMetrics], RunDirectory]:
        """Run the full training loop and return per-episode metrics + run directory."""
        agent_cfg = self._cfg.setup["agent"]
        per_cfg = self._cfg.setup["per"]
        env_cfg = self._cfg.setup["env"]
        train_cfg = self._cfg.setup["training"]
        train_env = self._make_env(self._pipeline.train.__dict__["features"].shape[1], env_cfg)
        val_env = self._make_env(self._pipeline.val.__dict__["features"].shape[1], env_cfg, "val")
        replay = self._make_replay(agent_cfg, per_cfg)
        agent = DQNAgent(
            window_size=train_env.observation_shape[0],
            n_features=train_env.observation_shape[1],
            n_actions=3,
            replay=replay,
            gamma=agent_cfg["gamma"],
            lr=agent_cfg["lr"],
            huber_delta=agent_cfg["huber_delta"],
            grad_clip=agent_cfg["grad_clip"],
            target_sync_every=agent_cfg["target_sync_every"],
            dueling=agent_cfg["dueling"],
            double_dqn=agent_cfg["double_dqn"],
            device=self._device,
        )
        run = create_run(self._cfg.path("results_dir"))
        run.write_config_snapshot(dict(self._cfg.setup))
        run.write_git_hash()
        eps_sched = EpsilonSchedule(
            agent_cfg["epsilon_start"], agent_cfg["epsilon_end"], agent_cfg["epsilon_decay_steps"]
        )
        beta_sched = BetaSchedule(
            per_cfg["beta_start"], per_cfg["beta_end"], per_cfg["beta_anneal_steps"]
        )
        rng = np.random.default_rng(int(self._cfg.get("seed", 0)))
        metrics: list[EpisodeMetrics] = []
        best = -float("inf")
        global_step = 0
        for ep in range(int(train_cfg["episodes"])):
            ep_reward, ep_loss, ep_trades, final_v, global_step = self._run_episode(
                agent, train_env, eps_sched, beta_sched, agent_cfg["batch_size"], global_step, rng
            )
            val_ret = self._evaluate(agent, val_env)
            metrics.append(EpisodeMetrics(ep, ep_reward, ep_loss, eps_sched.value(global_step),
                                          ep_trades, final_v, val_ret))
            _logger.info(
                "ep=%d reward=%.4f loss=%.4f eps=%.3f trades=%d final_v=%.2f val_ret=%.4f",
                ep, ep_reward, ep_loss, eps_sched.value(global_step), ep_trades, final_v, val_ret,
            )
            if val_ret > best:
                best = val_ret
                agent.save(str(run.checkpoints / "best.pt"))
        agent.save(str(run.checkpoints / "last.pt"))
        self._write_metrics_csv(run.metrics_csv, metrics)
        return metrics, run

    def _make_env(self, _window: int, env_cfg: dict, slice_name: str = "train") -> TradingEnv:
        slc = getattr(self._pipeline, slice_name)
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

    def _make_replay(self, agent_cfg: dict, per_cfg: dict):
        if per_cfg.get("enabled", True):
            return PrioritizedReplay(
                capacity=agent_cfg["replay_capacity"], alpha=per_cfg["alpha"],
                epsilon=per_cfg["epsilon"], seed=int(self._cfg.get("seed", 0)),
            )
        return UniformReplay(capacity=agent_cfg["replay_capacity"], seed=int(self._cfg.get("seed", 0)))

    def _run_episode(self, agent, env, eps_sched, beta_sched, batch_size, step, rng):
        state, _ = env.reset()
        ep_reward, losses, trades = 0.0, [], 0
        done = False
        final_v = self._cfg.setup["env"]["initial_capital"]
        while not done:
            action = agent.act(state, epsilon=eps_sched.value(step), rng=rng)
            nxt, reward, done, _, info = env.step(action)
            agent.remember(state, action, reward, nxt, done)
            loss = agent.optimize(batch_size=batch_size, beta=beta_sched.value(step))
            if loss is not None:
                losses.append(loss)
            ep_reward += reward
            trades += int(info["trade_executed"])
            final_v = info["portfolio_value"]
            state = nxt
            step += 1
        return ep_reward, float(np.mean(losses)) if losses else 0.0, trades, final_v, step

    @staticmethod
    def _evaluate(agent: DQNAgent, env: TradingEnv) -> float:
        state, _ = env.reset()
        initial = env._portfolio.initial_capital  # type: ignore[attr-defined]
        done = False
        final_v = initial
        rng = np.random.default_rng(0)
        while not done:
            action = agent.act(state, epsilon=0.0, rng=rng)
            state, _, done, _, info = env.step(action)
            final_v = info["portfolio_value"]
        return (final_v - initial) / initial

    @staticmethod
    def _write_metrics_csv(path: Path, metrics: list[EpisodeMetrics]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "loss", "epsilon", "trades", "final_value", "val_return"])
            for m in metrics:
                writer.writerow([m.episode, m.reward, m.loss, m.epsilon, m.trades, m.final_value, m.val_return])
