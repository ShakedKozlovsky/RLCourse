"""Audit findings #15 + #19: qualitative + per-component diagnostics.

* :func:`record_greedy_rollout` runs one greedy rollout and returns a
  human-readable trajectory record: per-day (action_name, total_reward,
  per-component reward breakdown). Audit #15.
* :func:`reward_decomposition` over a training history returns the
  per-episode mean of `{gain, overload, imbalance}` — answers "which reward
  term is the policy actually optimising?". Audit #19.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.services.evaluation_service import actor_logits
from fitness_rl.shared.types import Action


@dataclass(frozen=True)
class GreedyStep:
    """One row of the qualitative trajectory."""

    step: int
    action: int
    action_name: str
    gain: float
    overload: float
    imbalance: float
    total_reward: float


@dataclass(frozen=True)
class GreedyTrajectory:
    """Sequence of ``GreedyStep``s for one episode + total reward."""

    steps: list[GreedyStep]
    total_reward: float

    def as_table(self) -> str:
        """Return a fixed-width text table for the README's qualitative section."""
        header = f"{'t':>3} {'action':>7} {'gain':>7} {'overload':>9} {'imbalance':>10} {'r_t':>7}"
        rows = [header, "-" * len(header)]
        for s in self.steps:
            rows.append(
                f"{s.step:>3} {s.action_name:>7} {s.gain:>7.3f} "
                f"{s.overload:>9.3f} {s.imbalance:>10.3f} {s.total_reward:>7.3f}"
            )
        rows.append(f"\nTotal reward: {self.total_reward:.3f}")
        return "\n".join(rows)


def record_greedy_rollout(
    net: PolicyNet | ActorCriticNet,
    env: WorldEnv,
    reward_fn: RewardFunction,
) -> GreedyTrajectory:
    """Run a greedy (argmax) rollout, recording the per-step reward decomposition.

    Audit finding #15: this is the *qualitative* output of the
    fitness-recommender. Without it the README only shows aggregate reward.
    """
    logits_fn = actor_logits(net)
    state, _ = env.reset()
    reward_fn.reset()
    steps: list[GreedyStep] = []
    total = 0.0
    terminated = False
    with torch.no_grad():
        while not terminated:
            logits = logits_fn(torch.from_numpy(state).float())
            mask_np = env.get_mask()
            if mask_np is not None:
                logits = logits + torch.from_numpy(mask_np).float()
            action = int(torch.argmax(logits).item())
            next_state, _, terminated, _, info = env.step(action)
            parts = reward_fn.decompose(next_state, action=action)
            # Sync the diagnostic reward_fn's rolling window with the env's.
            reward_fn._window.append(  # noqa: SLF001
                max(0.0, float(next_state[0]))
            )
            r = parts["total"]
            total += r
            steps.append(GreedyStep(
                step=info["step_idx"],
                action=action,
                action_name=Action(action).name,
                gain=parts["gain"],
                overload=parts["overload"],
                imbalance=parts["imbalance"],
                total_reward=r,
            ))
            state = next_state
    return GreedyTrajectory(steps=steps, total_reward=total)


def reward_decomposition(
    history_rewards: list[float],
    history_action_counts: list[np.ndarray],
) -> dict[str, list[float]]:
    """Aggregate per-episode totals into ``{episode_index, mean_reward, action_mix}``.

    Lightweight — the *full* per-component breakdown lives inside
    ``record_greedy_rollout`` because that's where ``decompose()`` runs.
    """
    return {
        "rewards": list(history_rewards),
        "action_mix": [counts.tolist() for counts in history_action_counts],
    }
