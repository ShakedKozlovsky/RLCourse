"""REINFORCE training service — episodic policy gradient (Williams 1992).

Per episode: roll out the policy in the env, compute reward-to-go G_t,
subtract the episode-mean baseline, and apply ∇_θ log π_θ · (G_t - b).
Action masking and an entropy bonus are both optional.
See ``docs/PRD_reinforce.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.types import EpisodeMetrics

_logger = get_logger(__name__)


@dataclass(frozen=True)
class _RolloutBuffer:
    log_probs: list[torch.Tensor]
    rewards: list[float]
    entropies: list[torch.Tensor]
    actions: list[int]


class ReinforceService:
    """Train a :class:`PolicyNet` via episodic policy gradient with mean baseline."""

    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 1e-3,
        use_baseline: bool = True,
        entropy_bonus: float = 0.0,
        use_action_mask: bool = False,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if entropy_bonus < 0:
            raise ValueError("entropy_bonus must be >= 0")
        self._gamma = float(gamma)
        self._lr = float(lr)
        self._use_baseline = bool(use_baseline)
        self._entropy_bonus = float(entropy_bonus)
        self._use_mask = bool(use_action_mask)

    def fit(
        self, policy: PolicyNet, env: WorldEnv, episodes: int
    ) -> list[EpisodeMetrics]:
        """Train ``policy`` for ``episodes`` rollouts; return per-episode metrics."""
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        opt = optim.Adam(policy.parameters(), lr=self._lr)
        history: list[EpisodeMetrics] = []
        for ep in range(episodes):
            buf = self._rollout(policy, env)
            self._update(opt, buf)
            history.append(self._make_metrics(ep, buf))
        return history

    def _rollout(self, policy: PolicyNet, env: WorldEnv) -> _RolloutBuffer:
        policy.train()
        state_np, _ = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []
        actions: list[int] = []
        terminated = False
        while not terminated:
            logits = policy(torch.from_numpy(state_np).float())
            if self._use_mask:
                mask_np = env.get_mask()
                if mask_np is not None:
                    logits = logits + torch.from_numpy(mask_np).float()
            dist = Categorical(logits=logits)
            action = dist.sample()
            state_np, reward, terminated, _, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))
            entropies.append(dist.entropy())
            actions.append(int(action.item()))
        return _RolloutBuffer(log_probs, rewards, entropies, actions)

    def _update(self, opt: optim.Optimizer, buf: _RolloutBuffer) -> None:
        returns = self.reward_to_go(buf.rewards, self._gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if self._use_baseline and returns_t.numel() > 1:
            returns_t = returns_t - returns_t.mean()
        log_probs_t = torch.stack(buf.log_probs)
        entropy_t = torch.stack(buf.entropies).mean() if buf.entropies else torch.tensor(0.0)
        policy_loss = -(log_probs_t * returns_t).sum()
        loss = policy_loss - self._entropy_bonus * entropy_t
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def reward_to_go(rewards: list[float], gamma: float) -> list[float]:
        """G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … computed in reverse."""
        out = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            out[t] = running
        return out

    @staticmethod
    def _make_metrics(ep: int, buf: _RolloutBuffer) -> EpisodeMetrics:
        from fitness_rl.shared.types import Action

        counts = np.zeros(Action.n(), dtype=np.int64)
        for a in buf.actions:
            counts[a] += 1
        mean_entropy = float(torch.stack(buf.entropies).mean().item()) if buf.entropies else 0.0
        return EpisodeMetrics(
            episode=int(ep),
            total_reward=float(sum(buf.rewards)),
            mean_entropy=mean_entropy,
            action_counts=counts,
        )
