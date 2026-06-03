"""Workout recommender — the user-facing product around the trained policy.

Takes the user's recent action history and recommends the next N days as a
greedy rollout from the env's initial state, optionally honouring the action
mask (no 3-in-a-row, no 3-rest-in-a-row). Returns a structured plan with
per-day reward decomposition so the user knows *why* each action was picked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.services.evaluation_service import actor_logits
from fitness_rl.shared.types import Action


@dataclass(frozen=True)
class RecommendedDay:
    """One day of the recommended schedule."""

    day_index: int
    action: int
    action_name: str
    gain: float
    overload: float
    imbalance: float
    expected_reward: float


@dataclass(frozen=True)
class WorkoutPlan:
    """N-day plan + summary statistics."""

    days: list[RecommendedDay]
    total_expected_reward: float

    def as_table(self) -> str:
        """Render as a human-readable fixed-width table."""
        header = f"{'Day':>3} {'Workout':>8} {'Gain':>6} {'Overload':>9} {'Imbalance':>10} {'Reward':>7}"
        rows = [header, "-" * len(header)]
        for d in self.days:
            rows.append(
                f"{d.day_index:>3} {d.action_name:>8} {d.gain:>6.3f} "
                f"{d.overload:>9.3f} {d.imbalance:>10.3f} {d.expected_reward:>7.3f}"
            )
        rows.append(f"\nExpected total reward: {self.total_expected_reward:.3f}")
        return "\n".join(rows)

    def actions(self) -> list[str]:
        return [d.action_name for d in self.days]


class WorkoutRecommender:
    """Convert a trained policy + recent history into a forward-looking workout plan."""

    def __init__(self, action_mask: ActionMask | None = None):
        self._action_mask = action_mask

    def recommend(
        self,
        net: PolicyNet | ActorCriticNet,
        env: WorldEnv,
        reward_fn: RewardFunction,
        n_days: int = 7,
        recent_actions: list[int] | None = None,
    ) -> WorkoutPlan:
        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        logits_fn = actor_logits(net)
        state, _ = env.reset()
        reward_fn.reset()
        recent: list[int] = list(recent_actions or [])
        days: list[RecommendedDay] = []
        total = 0.0
        with torch.no_grad():
            for i in range(n_days):
                logits = logits_fn(torch.from_numpy(state).float())
                # Honour either the env's mask or our user-provided history.
                env_mask = env.get_mask()
                history_mask = (self._action_mask.mask(recent)
                                 if self._action_mask is not None else None)
                if env_mask is not None:
                    logits = logits + torch.from_numpy(env_mask).float()
                if history_mask is not None:
                    logits = logits + torch.from_numpy(history_mask).float()
                action = int(torch.argmax(logits).item())
                next_state, _, terminated, _, _ = env.step(action)
                parts = reward_fn.decompose(next_state, action=action)
                reward_fn._window.append(  # noqa: SLF001 - keep window in sync
                    max(0.0, float(next_state[0]))
                )
                days.append(RecommendedDay(
                    day_index=i + 1, action=action,
                    action_name=Action(action).name,
                    gain=parts["gain"], overload=parts["overload"],
                    imbalance=parts["imbalance"],
                    expected_reward=parts["total"],
                ))
                total += parts["total"]
                recent.append(action)
                state = next_state
                if terminated:
                    state, _ = env.reset()
        return WorkoutPlan(days=days, total_expected_reward=total)

    @staticmethod
    def parse_history(history_str: str) -> list[int]:
        """Parse a comma-separated string like ``'PUSH,PULL,REST'`` into action ints."""
        if not history_str.strip():
            return []
        out = []
        for token in history_str.split(","):
            name = token.strip().upper()
            try:
                out.append(int(Action[name]))
            except KeyError as exc:
                raise ValueError(
                    f"unknown action {token!r}; valid: "
                    + ", ".join(a.name for a in Action)
                ) from exc
        return out
