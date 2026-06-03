"""PPO training service — clipped surrogate objective (Schulman et al. 2017).

PPO is **beyond the assignment spec** — added as the natural step after
REINFORCE → A2C. See ``docs/PRD_ppo.md``.
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


@dataclass
class _Rollout:
    states: list[np.ndarray]
    actions: list[int]
    log_probs_old: list[float]
    rewards: list[float]
    values: list[float]
    dones: list[bool]


class PPOService:
    """Train an :class:`ActorCriticNet` via PPO's clipped surrogate."""

    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 5e-4,
        clip_eps: float = 0.2,
        n_epochs_per_batch: int = 4,
        n_steps_per_update: int = 28,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        use_action_mask: bool = False,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if clip_eps <= 0 or n_epochs_per_batch < 1 or n_steps_per_update < 1:
            raise ValueError("clip_eps, n_epochs_per_batch, n_steps_per_update must be > 0")
        self._gamma = float(gamma)
        self._lr = float(lr)
        self._clip = float(clip_eps)
        self._n_epochs = int(n_epochs_per_batch)
        self._n_steps = int(n_steps_per_update)
        self._vcoef = float(value_coef)
        self._ecoef = float(entropy_coef)
        self._use_mask = bool(use_action_mask)

    def fit(
        self, net: ActorCriticNet, env: WorldEnv, episodes: int
    ) -> list[EpisodeMetrics]:
        """Train ``net`` for ``episodes`` rollouts; return per-episode metrics."""
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        opt = optim.Adam(net.parameters(), lr=self._lr)
        history: list[EpisodeMetrics] = []
        for ep in range(episodes):
            buf = self._collect_episode(net, env)
            self._ppo_update(net, opt, buf)
            history.append(self._make_metrics(ep, buf))
        return history

    def _collect_episode(self, net: ActorCriticNet, env: WorldEnv) -> _Rollout:
        net.eval()
        state_np, _ = env.reset()
        buf = _Rollout([], [], [], [], [], [])
        terminated = False
        while not terminated:
            state_t = torch.from_numpy(state_np).float()
            with torch.no_grad():
                logits, value = net(state_t)
                if self._use_mask:
                    mask_np = env.get_mask()
                    if mask_np is not None:
                        logits = logits + torch.from_numpy(mask_np).float()
                dist = Categorical(logits=logits)
                action = dist.sample()
            next_state, reward, terminated, _, _ = env.step(int(action.item()))
            buf.states.append(state_np.copy())
            buf.actions.append(int(action.item()))
            buf.log_probs_old.append(float(dist.log_prob(action).item()))
            buf.rewards.append(float(reward))
            buf.values.append(float(value.item()))
            buf.dones.append(bool(terminated))
            state_np = next_state
        return buf

    def _ppo_update(
        self, net: ActorCriticNet, opt: optim.Optimizer, buf: _Rollout
    ) -> None:
        net.train()
        s = torch.from_numpy(np.stack(buf.states)).float()
        a = torch.tensor(buf.actions, dtype=torch.long)
        old_lp = torch.tensor(buf.log_probs_old, dtype=torch.float32)
        rewards = torch.tensor(buf.rewards, dtype=torch.float32)
        values = torch.tensor(buf.values, dtype=torch.float32)
        # One-step TD: V(s_{t+1}) is 0 when terminal.
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        returns = rewards + self._gamma * next_values
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self._n_epochs):
            logits, new_values = net(s)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - self._clip, 1 + self._clip)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (new_values - returns).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + self._vcoef * value_loss - self._ecoef * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()

    @staticmethod
    def _make_metrics(ep: int, buf: _Rollout) -> EpisodeMetrics:
        counts = np.zeros(Action.n(), dtype=np.int64)
        for x in buf.actions:
            counts[x] += 1
        # entropy approximated from action distribution within the episode.
        probs = counts.astype(np.float64) / max(1, counts.sum())
        mean_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        return EpisodeMetrics(
            episode=int(ep),
            total_reward=float(sum(buf.rewards)),
            mean_entropy=mean_entropy,
            action_counts=counts,
        )
