"""DQNAgent — owns the online/target networks, optimizer, and replay buffer.

Bellman target (Double DQN when ``double_dqn=True``):
    a* = argmax_{a'} Q_online(s', a')
    y  = r + γ · (1 − done) · Q_target(s', a*)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from dqn_trader.memory.prioritized_replay import PrioritizedReplay
from dqn_trader.memory.uniform_replay import Batch, UniformReplay
from dqn_trader.model.dueling_dqn import DuelingDQN, hard_update

ReplayBuffer = UniformReplay | PrioritizedReplay


class DQNAgent:
    """Single-agent training coordinator. Buffer-agnostic and Dueling-agnostic."""

    def __init__(
        self,
        window_size: int,
        n_features: int,
        n_actions: int,
        replay: ReplayBuffer,
        *,
        gamma: float = 0.99,
        lr: float = 5e-4,
        huber_delta: float = 1.0,
        grad_clip: float = 10.0,
        target_sync_every: int = 1000,
        dueling: bool = True,
        double_dqn: bool = True,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.online = DuelingDQN(window_size, n_features, n_actions, dueling=dueling).to(self.device)
        self.target = DuelingDQN(window_size, n_features, n_actions, dueling=dueling).to(self.device)
        hard_update(self.target, self.online)
        self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none")
        self.replay = replay
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)
        self._step = 0

    def act(self, state: np.ndarray, *, epsilon: float, rng: np.random.Generator) -> int:
        """ε-greedy over Q. Greedy when ``epsilon = 0``."""
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
            q = self.online(x).squeeze(0)
        return int(torch.argmax(q).item())

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.replay.add(s, a, r, s2, done)

    def optimize(self, batch_size: int, beta: float) -> float | None:
        """One gradient step. Returns the mean loss, or ``None`` if buffer is too small."""
        if len(self.replay) < batch_size:
            return None
        batch: Batch = self.replay.sample(batch_size, beta=beta)
        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device, dtype=torch.float32)
        is_w = torch.from_numpy(batch.is_weights).to(self.device)
        q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            y = rewards + self.gamma * (1.0 - dones) * next_q
        td = y - q_pred
        loss = (self.loss_fn(q_pred, y) * is_w).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        self.replay.update_priorities(batch.indices, td.detach().cpu().numpy())
        self._step += 1
        if self._step % self.target_sync_every == 0:
            hard_update(self.target, self.online)
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict(), "step": self._step}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(state["online"])
        hard_update(self.target, self.online)
        self._step = int(state.get("step", 0))
