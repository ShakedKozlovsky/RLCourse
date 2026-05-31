"""ActionSpace — 5-action discrete contract."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_rl.environment.action_space import ActionSpace
from fitness_rl.shared.types import Action


def test_n_is_five() -> None:
    assert ActionSpace().n == 5


def test_actions_tuple_matches_enum() -> None:
    space = ActionSpace()
    assert space.actions == tuple(Action)


def test_sample_in_range() -> None:
    rng = np.random.default_rng(0)
    space = ActionSpace()
    for _ in range(50):
        a = space.sample(rng)
        assert 0 <= a < space.n


def test_contains_valid_indices() -> None:
    space = ActionSpace()
    for i in range(space.n):
        assert space.contains(i)


def test_contains_rejects_invalid() -> None:
    space = ActionSpace()
    assert not space.contains(-1)
    assert not space.contains(space.n)


def test_name_returns_enum_name() -> None:
    space = ActionSpace()
    assert space.name(int(Action.PUSH)) == "PUSH"
    assert space.name(int(Action.REST)) == "REST"


def test_name_invalid_raises() -> None:
    with pytest.raises(ValueError):
        ActionSpace().name(99)
