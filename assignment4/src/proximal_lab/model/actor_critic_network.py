"""ActorCriticNet — separate actor + critic networks bundled for PPO.

The shared trunk is intentionally **not** used here. PRD_actor_critic.md
documents the rationale (avoids the trunk-double-step bug from Assignment 3
ADR-007).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from proximal_lab.model.actor import GaussianActor
from proximal_lab.model.critic import Critic


class ActorCriticNet(nn.Module):
    """Bundles a :class:`GaussianActor` and a :class:`Critic`."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.actor = GaussianActor(
            obs_dim, action_dim, hidden_sizes,
            log_std_init=log_std_init,
            log_std_min=log_std_min, log_std_max=log_std_max,
        )
        self.critic = Critic(obs_dim, hidden_sizes)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, value)`` — used during rollout collection."""
        dist = self.actor(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO updates: recompute (log_prob, entropy, value) under current θ."""
        dist = self.actor(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs)
        return log_prob, entropy, value

    def actor_params(self) -> list[nn.Parameter]:
        return list(self.actor.parameters())

    def critic_params(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "obs_dim": self.obs_dim, "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
        }, path)

    @classmethod
    def load(cls, path: Path) -> ActorCriticNet:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(obs_dim=int(blob["obs_dim"]), action_dim=int(blob["action_dim"]),
                   hidden_sizes=tuple(blob["hidden_sizes"]))
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net
