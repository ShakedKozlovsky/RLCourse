"""A2C training service — one-step TD advantage actor-critic (slide 20).

Per transition (s, a, r, s'):
    δ = r + γ·V(s') − V(s)           (TD error = advantage)
    actor_loss  = −log π(a|s) · δ.detach() − β · H(π)
    critic_loss = ½ · δ²
See ``docs/PRD_a2c.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.types import Action, EpisodeMetrics

_logger = get_logger(__name__)


@dataclass(frozen=True)
class _StepBuffer:
    rewards: list[float]
    entropies: list[float]
    actions: list[int]
    actor_loss_sum: float
    critic_loss_sum: float


class A2CService:
    """Train an :class:`ActorCriticNet` via per-step TD updates."""

    def __init__(
        self,
        gamma: float = 0.99,
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        entropy_bonus: float = 0.01,
        use_action_mask: bool = False,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if entropy_bonus < 0:
            raise ValueError("entropy_bonus must be >= 0")
        self._gamma = float(gamma)
        self._actor_lr = float(actor_lr)
        self._critic_lr = float(critic_lr)
        self._entropy_bonus = float(entropy_bonus)
        self._use_mask = bool(use_action_mask)

    def fit(
        self, net: ActorCriticNet, env: WorldEnv, episodes: int
    ) -> list[EpisodeMetrics]:
        """Train ``net`` for ``episodes`` rollouts; return per-episode metrics."""
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        actor_opt = optim.Adam(net.actor_params(), lr=self._actor_lr)
        critic_opt = optim.Adam(net.critic_params(), lr=self._critic_lr)
        history: list[EpisodeMetrics] = []
        for ep in range(episodes):
            buf = self._run_episode(net, env, actor_opt, critic_opt)
            history.append(self._make_metrics(ep, buf))
        return history

    def _run_episode(
        self,
        net: ActorCriticNet,
        env: WorldEnv,
        actor_opt: optim.Optimizer,
        critic_opt: optim.Optimizer,
    ) -> _StepBuffer:
        net.train()
        state_np, _ = env.reset()
        rewards: list[float] = []
        entropies: list[float] = []
        actions: list[int] = []
        actor_loss_sum, critic_loss_sum = 0.0, 0.0
        terminated = False
        while not terminated:
            logits, value = net(torch.from_numpy(state_np).float())
            if self._use_mask:
                mask_np = env.get_mask()
                if mask_np is not None:
                    logits = logits + torch.from_numpy(mask_np).float()
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_state_np, reward, terminated, _, _ = env.step(int(action.item()))
            with torch.no_grad():
                _, next_value = net(torch.from_numpy(next_state_np).float())
                bootstrap = 0.0 if terminated else float(next_value.item())
                td_target = reward + self._gamma * bootstrap
            delta = torch.tensor(td_target, dtype=torch.float32) - value
            actor_loss = -dist.log_prob(action) * delta.detach() - \
                self._entropy_bonus * dist.entropy()
            critic_loss = 0.5 * delta.pow(2)
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            (actor_loss + critic_loss).backward()
            actor_opt.step()
            critic_opt.step()
            rewards.append(float(reward))
            entropies.append(float(dist.entropy().item()))
            actions.append(int(action.item()))
            actor_loss_sum += float(actor_loss.item())
            critic_loss_sum += float(critic_loss.item())
            state_np = next_state_np
        return _StepBuffer(rewards, entropies, actions, actor_loss_sum, critic_loss_sum)

    @staticmethod
    def td_error(reward: float, value: float, next_value: float,
                 gamma: float, terminated: bool) -> float:
        """δ = r + γ·V(s') − V(s); V(s') is zeroed when terminated."""
        bootstrap = 0.0 if terminated else next_value
        return reward + gamma * bootstrap - value

    @staticmethod
    def _make_metrics(ep: int, buf: _StepBuffer) -> EpisodeMetrics:
        counts = np.zeros(Action.n(), dtype=np.int64)
        for a in buf.actions:
            counts[a] += 1
        mean_entropy = float(np.mean(buf.entropies)) if buf.entropies else 0.0
        return EpisodeMetrics(
            episode=int(ep),
            total_reward=float(sum(buf.rewards)),
            mean_entropy=mean_entropy,
            action_counts=counts,
        )
