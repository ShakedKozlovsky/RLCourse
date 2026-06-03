"""FitnessRL SDK — single facade over data, world model, REINFORCE, A2C, PPO.

The SDK is the boundary the CLI and GUI both call into, so the GUI never
talks to the services directly. Each method is idempotent enough that the
GUI can re-run pieces without re-running the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.sdk.env_builder import build_env
from fitness_rl.sdk.trainers import (
    build_a2c_service,
    build_ppo_service,
    build_reinforce_service,
    build_world_model_service,
)
from fitness_rl.services.comparison_service import ComparisonResult, ComparisonService
from fitness_rl.services.data_service import DataService, PipelineOutput
from fitness_rl.services.evaluation_service import (
    EvaluationResult,
    EvaluationService,
    actor_logits,
)
from fitness_rl.services.world_model_service import TrainResult
from fitness_rl.shared.config import ConfigManager
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.seed import set_global_seed
from fitness_rl.shared.types import EpisodeMetrics

_logger = get_logger(__name__)


class FitnessRL:
    """High-level facade. Holds state across method calls (data, models, histories)."""

    def __init__(self, config_path: Path):
        self._cfg = ConfigManager(Path(config_path))
        set_global_seed(int(self._cfg.get("seed")))
        self._data: PipelineOutput | None = None
        self._world_model: LSTMWorldModel | None = None
        self._reinforce_policy: PolicyNet | None = None
        self._a2c_net: ActorCriticNet | None = None
        self._ppo_net: ActorCriticNet | None = None
        self._reinforce_history: list[EpisodeMetrics] | None = None
        self._a2c_history: list[EpisodeMetrics] | None = None
        self._ppo_history: list[EpisodeMetrics] | None = None

    @property
    def config(self) -> ConfigManager:
        return self._cfg

    @property
    def world_model(self) -> LSTMWorldModel | None:
        return self._world_model

    @property
    def data(self) -> PipelineOutput | None:
        return self._data

    def make_env(self) -> WorldEnv:
        return self._make_env()

    def prepare_data(self) -> PipelineOutput:
        self._data = DataService(self._cfg).run()
        return self._data

    def train_world_model(self) -> TrainResult:
        data = self._require_data()
        self._world_model = LSTMWorldModel(
            hidden_size=int(self._cfg.get("world_model.hidden_size")),
            num_layers=int(self._cfg.get("world_model.num_layers")),
        )
        result = build_world_model_service(self._cfg).train(
            self._world_model, data.states, data.actions)
        ckpt = Path(self._cfg.path("checkpoints_dir")) / "world_model.pt"
        self._world_model.save(ckpt)
        _logger.info("world model saved to %s", ckpt)
        return result

    def train_reinforce(self, episodes: int | None = None) -> list[EpisodeMetrics]:
        env = self._make_env()
        self._reinforce_policy = PolicyNet(
            hidden_size=int(self._cfg.get("reinforce.policy_hidden")))
        n = int(episodes) if episodes is not None else int(self._cfg.get("reinforce.episodes"))
        self._reinforce_history = build_reinforce_service(self._cfg).fit(
            self._reinforce_policy, env, episodes=n)
        return self._reinforce_history

    def train_a2c(self, episodes: int | None = None) -> list[EpisodeMetrics]:
        env = self._make_env()
        self._a2c_net = ActorCriticNet(hidden_size=int(self._cfg.get("a2c.hidden")))
        n = int(episodes) if episodes is not None else int(self._cfg.get("a2c.episodes"))
        self._a2c_history = build_a2c_service(self._cfg).fit(
            self._a2c_net, env, episodes=n)
        return self._a2c_history

    def train_ppo(self, episodes: int | None = None) -> list[EpisodeMetrics]:
        env = self._make_env()
        self._ppo_net = ActorCriticNet(
            hidden_size=int(self._cfg.get("ppo.hidden", self._cfg.get("a2c.hidden"))))
        n = int(episodes) if episodes is not None else int(
            self._cfg.get("ppo.episodes", self._cfg.get("a2c.episodes")))
        self._ppo_history = build_ppo_service(self._cfg).fit(
            self._ppo_net, env, episodes=n)
        return self._ppo_history

    def compare(self) -> ComparisonResult:
        if self._reinforce_history is None or self._a2c_history is None:
            raise RuntimeError("compare() requires both REINFORCE and A2C trained first")
        return ComparisonService().compare(self._reinforce_history, self._a2c_history)

    def evaluate(self, algo: str = "a2c") -> EvaluationResult:
        net = self._require_net(algo)
        return EvaluationService().rollout(actor_logits(net), self._make_env())

    def predict(self, state: np.ndarray, algo: str = "a2c") -> int:
        net = self._require_net(algo)
        net.eval()
        with torch.no_grad():
            logits = actor_logits(net)(torch.from_numpy(state).float())
            return int(torch.argmax(logits).item())

    def _make_env(self) -> WorldEnv:
        data = self._require_data()
        return build_env(self._cfg, data.states[0], self._world_model)

    def _require_data(self) -> PipelineOutput:
        if self._data is None:
            return self.prepare_data()
        return self._data

    def _require_net(self, algo: str) -> PolicyNet | ActorCriticNet:
        if algo == "reinforce":
            if self._reinforce_policy is None:
                raise RuntimeError("REINFORCE policy not trained")
            return self._reinforce_policy
        if algo == "a2c":
            if self._a2c_net is None:
                raise RuntimeError("A2C network not trained")
            return self._a2c_net
        if algo == "ppo":
            if self._ppo_net is None:
                raise RuntimeError("PPO network not trained")
            return self._ppo_net
        raise ValueError(f"unknown algo {algo!r}; expected 'reinforce', 'a2c', 'ppo'")
