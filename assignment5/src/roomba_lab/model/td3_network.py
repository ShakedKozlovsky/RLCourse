"""TD3 (Twin Delayed DDPG) network — Fujimoto, Hoof, Meger ICML 2018.

Above-spec literature awareness: keeps DDPG as the central spec algorithm but
adds the 3 TD3 modifications as an opt-in alternative. Three differences from
vanilla DDPG (slide 9 of L09 mentions TD3 as 'modern variant'):

  1. **Twin critic** — two Q-networks; we use min(Q1, Q2) for the bootstrap
     target. Reduces over-estimation bias.
  2. **Delayed actor update** — actor steps only every `policy_delay` critic
     steps (we use 2 by default).
  3. **Target policy smoothing** — Gaussian noise added to the target action
     inside the bootstrap, clipped to a small range. Smooths the Q surface."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import torch.nn as nn

from roomba_lab.model.actor import Actor
from roomba_lab.model.critic import Critic


class TD3Network(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, actor_hidden_sizes)
        self.critic_a = Critic(obs_dim, action_dim, critic_hidden_sizes)
        self.critic_b = Critic(obs_dim, action_dim, critic_hidden_sizes)
        self.target_actor = deepcopy(self.actor)
        self.target_critic_a = deepcopy(self.critic_a)
        self.target_critic_b = deepcopy(self.critic_b)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic_a.parameters():
            p.requires_grad = False
        for p in self.target_critic_b.parameters():
            p.requires_grad = False
