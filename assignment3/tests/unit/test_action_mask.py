"""ActionMask — pre-softmax guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.shared.types import Action


def test_invalid_init_args_raise() -> None:
    with pytest.raises(ValueError):
        ActionMask(max_same_group=0)
    with pytest.raises(ValueError):
        ActionMask(max_rest=0)


def test_empty_history_no_mask() -> None:
    m = ActionMask().mask([])
    assert m.shape == (Action.n(),)
    assert np.all(m == 0.0)
    assert m.dtype == np.float32


def test_short_history_no_mask() -> None:
    m = ActionMask(max_same_group=2).mask([int(Action.PUSH)])
    assert np.all(m == 0.0)


def test_two_consecutive_push_forbids_push() -> None:
    m = ActionMask(max_same_group=2).mask([int(Action.PUSH), int(Action.PUSH)])
    assert m[Action.PUSH] == -np.inf
    # other actions allowed
    for a in (Action.PULL, Action.LEGS, Action.CARDIO, Action.REST):
        assert m[a] == 0.0


def test_two_consecutive_rest_forbids_rest() -> None:
    m = ActionMask(max_rest=2).mask([int(Action.REST), int(Action.REST)])
    assert m[Action.REST] == -np.inf


def test_mixed_recent_actions_no_mask() -> None:
    m = ActionMask().mask([int(Action.PUSH), int(Action.PULL)])
    assert np.all(m == 0.0)


def test_history_longer_than_window_uses_tail() -> None:
    # PUSH, PUSH, PULL → last 2 are PUSH+PULL → no mask
    m = ActionMask(max_same_group=2).mask(
        [int(Action.PUSH), int(Action.PUSH), int(Action.PULL)]
    )
    assert np.all(m == 0.0)


def test_dtype_is_float32() -> None:
    m = ActionMask().mask([int(Action.PUSH), int(Action.PUSH)])
    assert m.dtype == np.float32
