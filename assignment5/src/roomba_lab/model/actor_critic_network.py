"""Combined Actor + Critic + their Polyak target copies — the object the DDPG
training service operates on."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import torch
import torch.nn as nn

from roomba_lab.model.actor import Actor
from roomba_lab.model.critic import Critic


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, actor_hidden_sizes)
        self.critic = Critic(obs_dim, action_dim, critic_hidden_sizes)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(obs)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, obs_dim: int, action_dim: int,
             actor_hidden_sizes: Sequence[int] = (256, 256),
             critic_hidden_sizes: Sequence[int] = (256, 256)) -> ActorCriticNet:
        net = cls(obs_dim, action_dim, actor_hidden_sizes, critic_hidden_sizes)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return net
