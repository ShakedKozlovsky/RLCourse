"""Greedy evaluation rollout for a trained policy.

The trained policy is sampled stochastically *during training* (exploration);
at evaluation time we want the deterministic recommendation, so we take
argmax over logits. Works for both ``PolicyNet`` (logits) and
``ActorCriticNet`` (logits, value) via the ``actor_logits`` adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.shared.types import Action

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EvaluationResult:
    """One deterministic rollout's summary."""

    total_reward: float
    action_counts: np.ndarray  # shape (n_actions,)
    action_sequence: list[int]
    rewards: list[float]
    states: list[np.ndarray]


def actor_logits(net: PolicyNet | ActorCriticNet) -> LogitsFn:
    """Return a ``state_tensor -> logits_tensor`` callable for either network."""
    if isinstance(net, ActorCriticNet):
        return lambda s: net(s)[0]
    return net


class EvaluationService:
    """Run one greedy rollout; honour the env's action mask when present."""

    def __init__(self, respect_action_mask: bool = True):
        self._respect_mask = bool(respect_action_mask)

    def rollout(self, logits_fn: LogitsFn, env: WorldEnv) -> EvaluationResult:
        state_np, _ = env.reset()
        states = [state_np.copy()]
        rewards: list[float] = []
        actions: list[int] = []
        counts = np.zeros(Action.n(), dtype=np.int64)
        terminated = False
        # Use eval-mode forward; no gradients needed.
        with torch.no_grad():
            while not terminated:
                logits = logits_fn(torch.from_numpy(state_np).float())
                if self._respect_mask:
                    mask_np = env.get_mask()
                    if mask_np is not None:
                        logits = logits + torch.from_numpy(mask_np).float()
                action = int(torch.argmax(logits).item())
                state_np, reward, terminated, _, _ = env.step(action)
                states.append(state_np.copy())
                rewards.append(float(reward))
                actions.append(action)
                counts[action] += 1
        return EvaluationResult(
            total_reward=float(sum(rewards)),
            action_counts=counts,
            action_sequence=actions,
            rewards=rewards,
            states=states,
        )

    @staticmethod
    def action_distribution(result: EvaluationResult) -> np.ndarray:
        """Normalised action frequencies; shape (n_actions,)."""
        total = int(result.action_counts.sum())
        if total == 0:
            return np.zeros_like(result.action_counts, dtype=np.float32)
        return result.action_counts.astype(np.float32) / float(total)

    @staticmethod
    def collapsed(result: EvaluationResult, threshold: float = 0.8) -> bool:
        """``True`` if any single action makes up ≥ ``threshold`` of the rollout."""
        dist = EvaluationService.action_distribution(result)
        return bool(dist.max() >= threshold)
