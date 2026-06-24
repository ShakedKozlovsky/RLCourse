"""Discrete action enum for Cops-and-Robbers. Cop has +1 action (PLACE_BARRIER).

Spec § 3.3 — the cop may place a barrier instead of moving (advanced option,
toggleable via `configs/setup.yaml::game.enable_barriers`)."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Action set for both agents. THIEF only uses indices 0–4."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BARRIER = 5    # cop only


COP_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
                Action.STAY, Action.PLACE_BARRIER)
THIEF_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY)


def n_actions(role: str, enable_barriers: bool) -> int:
    """Number of discrete actions available to `role` under the current config."""
    if role == "cop" and enable_barriers:
        return 6
    return 5


DELTA: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
    Action.PLACE_BARRIER: (0, 0),     # barrier placement doesn't move the cop
}
