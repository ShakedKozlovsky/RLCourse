"""RewardFunction — gain − λ_1·overload − λ_2·imbalance."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_rl.environment.reward import RewardFunction


def _state(volume: float, muscle_dist: list[float], rest: bool = False) -> np.ndarray:
    s = np.zeros(16, dtype=np.float32)
    s[0] = volume
    s[1:6] = muscle_dist
    s[15] = 1.0 if rest else 0.0
    return s


def test_invalid_state_shape_raises() -> None:
    rf = RewardFunction()
    with pytest.raises(ValueError):
        rf.compute(np.zeros(10, dtype=np.float32))


def test_negative_weights_rejected() -> None:
    with pytest.raises(ValueError):
        RewardFunction(gain_weight=-1.0)


def test_zero_rolling_window_rejected() -> None:
    with pytest.raises(ValueError):
        RewardFunction(rolling_window=0)


def test_uniform_distribution_has_zero_imbalance() -> None:
    rf = RewardFunction(gain_weight=0.0, overload_lambda=0.0, imbalance_lambda=1.0)
    rf.reset()
    s = _state(0.0, [0.2, 0.2, 0.2, 0.2, 0.2])
    r = rf.compute(s)
    # uniform → imbalance ≈ 0 → reward ≈ 0
    assert abs(r) < 1e-4


def test_concentrated_distribution_has_high_imbalance() -> None:
    rf = RewardFunction(gain_weight=0.0, overload_lambda=0.0, imbalance_lambda=1.0)
    rf.reset()
    s = _state(0.0, [1.0, 0.0, 0.0, 0.0, 0.0])
    r = rf.compute(s)
    # fully concentrated → imbalance ≈ 1 → reward ≈ -1
    assert r < -0.9


def test_rest_action_skips_imbalance_penalty() -> None:
    """Layer 11 fix: imbalance is zeroed by the action, not the state's rest_indicator."""
    from fitness_rl.shared.types import Action

    rf = RewardFunction(gain_weight=0.0, overload_lambda=0.0, imbalance_lambda=1.0)
    rf.reset()
    s = _state(0.0, [1.0, 0.0, 0.0, 0.0, 0.0])  # very concentrated
    # Without REST action: imbalance penalty applies.
    assert rf.compute(s, action=int(Action.PUSH)) < -0.9
    # With REST action: imbalance bypassed.
    rf.reset()
    assert rf.compute(s, action=int(Action.REST)) == 0.0


def test_rest_action_earns_zero_gain() -> None:
    """Layer 15 fix: REST earns no volume reward even when LSTM predicts high volume."""
    from fitness_rl.shared.types import Action

    rf = RewardFunction(gain_weight=1.0, overload_lambda=0.0, imbalance_lambda=0.0)
    rf.reset()
    s = _state(0.8, [0.2] * 5)  # LSTM predicts high volume
    # PUSH: full volume reward
    assert rf.compute(s, action=int(Action.PUSH)) == pytest.approx(0.8)
    # REST: zero gain even though volume is 0.8
    rf.reset()
    assert rf.compute(s, action=int(Action.REST)) == 0.0


def test_rest_action_still_pays_overload_penalty() -> None:
    """REST does not earn gain, but the rolling overload window still counts the
    volume so resting after heavy days still triggers the penalty."""
    from fitness_rl.shared.types import Action

    rf = RewardFunction(gain_weight=1.0, overload_lambda=1.0, imbalance_lambda=0.0,
                         rolling_window=3)
    rf.reset()
    rf.compute(_state(1.0, [0.2] * 5), action=int(Action.PUSH))   # window [1.0]
    rf.compute(_state(1.0, [0.2] * 5), action=int(Action.PUSH))   # window [1.0, 1.0]
    r = rf.compute(_state(0.5, [0.2] * 5), action=int(Action.REST))
    # gain = 0 (REST), overload = mean([1, 1, 0.5]) = 0.833, imbalance = 0
    assert r == pytest.approx(-(1.0 + 1.0 + 0.5) / 3.0)


def test_state_rest_indicator_no_longer_zeroes_imbalance() -> None:
    """Audit finding #10: state-based zeroing was an exploit; action-based is correct."""
    from fitness_rl.shared.types import Action

    rf = RewardFunction(gain_weight=0.0, overload_lambda=0.0, imbalance_lambda=1.0)
    rf.reset()
    # Concentrated muscle dist AND rest_indicator=1 — but action is PUSH.
    s = _state(0.0, [1.0, 0.0, 0.0, 0.0, 0.0], rest=True)
    assert rf.compute(s, action=int(Action.PUSH)) < -0.9


def test_decompose_returns_components() -> None:
    rf = RewardFunction(gain_weight=1.0, overload_lambda=0.5, imbalance_lambda=1.0)
    rf.reset()
    s = _state(0.4, [0.4, 0.2, 0.2, 0.1, 0.1])
    parts = rf.decompose(s, action=0)
    assert set(parts) == {"gain", "overload", "imbalance", "total"}
    assert parts["gain"] == pytest.approx(0.4)
    assert parts["total"] == pytest.approx(parts["gain"] - parts["overload"] - parts["imbalance"])


def test_decompose_does_not_mutate_state() -> None:
    rf = RewardFunction()
    rf.reset()
    s = _state(0.4, [0.2] * 5)
    snapshot = list(rf._window)  # noqa: SLF001
    rf.decompose(s, action=0)
    assert list(rf._window) == snapshot  # noqa: SLF001


def test_negative_volume_clamped_in_compute() -> None:
    """LSTM can predict negative volume; reward should treat it as 0."""
    rf = RewardFunction(gain_weight=1.0, overload_lambda=0.0, imbalance_lambda=0.0)
    rf.reset()
    s = _state(-0.5, [0.2] * 5)
    assert rf.compute(s, action=0) == 0.0


def test_gain_is_volume_weighted() -> None:
    rf = RewardFunction(gain_weight=2.0, overload_lambda=0.0, imbalance_lambda=0.0)
    rf.reset()
    s = _state(0.5, [0.2, 0.2, 0.2, 0.2, 0.2])
    assert rf.compute(s) == pytest.approx(1.0)


def test_overload_uses_rolling_mean() -> None:
    rf = RewardFunction(gain_weight=0.0, overload_lambda=1.0, imbalance_lambda=0.0,
                       rolling_window=3)
    rf.reset()
    rf.compute(_state(1.0, [0.2] * 5))  # window: [1.0]
    rf.compute(_state(0.0, [0.2] * 5))  # window: [1.0, 0.0]
    r = rf.compute(_state(0.0, [0.2] * 5))  # window: [1.0, 0.0, 0.0]
    # gain=0, overload = mean=1/3, imbalance=0 → r = -1/3
    assert r == pytest.approx(-1.0 / 3.0)


def test_reset_clears_window() -> None:
    rf = RewardFunction(gain_weight=0.0, overload_lambda=1.0, imbalance_lambda=0.0)
    rf.reset()
    rf.compute(_state(1.0, [0.2] * 5))
    rf.reset()
    r = rf.compute(_state(0.0, [0.2] * 5))
    assert r == 0.0


def test_zero_total_muscle_distribution_returns_zero_imbalance() -> None:
    rf = RewardFunction(gain_weight=0.0, overload_lambda=0.0, imbalance_lambda=1.0)
    rf.reset()
    s = _state(0.0, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert rf.compute(s) == 0.0
