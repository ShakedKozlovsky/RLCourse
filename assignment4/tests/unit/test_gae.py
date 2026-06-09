"""GAE math battery — the **headline** correctness tests of the project.

Four tests proving the slide-16 claims hold:
1. λ=0  ⇒  GAE collapses to per-step TD error.
2. λ=1  ⇒  GAE collapses to (discounted Monte-Carlo return) − V(s_t).
3. Closed-form check on a 3-step trajectory with hand-computed expected values.
4. Terminal handling: ``done`` at step k zeroes the bootstrap through that step.
"""

from __future__ import annotations

import numpy as np
import pytest

from proximal_lab.services.gae import compute_gae


def test_invalid_gamma_raises() -> None:
    with pytest.raises(ValueError):
        compute_gae(np.zeros(2), np.zeros(2), 0.0, np.zeros(2, dtype=bool),
                    gamma=0.0, lam=0.5)


def test_invalid_lambda_raises() -> None:
    with pytest.raises(ValueError):
        compute_gae(np.zeros(2), np.zeros(2), 0.0, np.zeros(2, dtype=bool),
                    gamma=0.99, lam=1.5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_gae(np.zeros(3), np.zeros(4), 0.0, np.zeros(3, dtype=bool),
                    gamma=0.99, lam=0.95)


def test_lambda_zero_reduces_to_td_error() -> None:
    """Slide-15 claim: λ=0 GAE = TD error."""
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    dones = np.zeros(3, dtype=bool)
    last_value = 1.5
    gamma = 0.99
    out = compute_gae(rewards, values, last_value, dones, gamma=gamma, lam=0.0)
    # δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
    # value chain: V(s_{t+1}) at step t is values[t+1] (or last_value at the tail)
    expected = np.array([
        rewards[0] + gamma * values[1] - values[0],
        rewards[1] + gamma * values[2] - values[1],
        rewards[2] + gamma * last_value - values[2],
    ], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_lambda_one_reduces_to_mc_minus_v() -> None:
    """Slide-15 claim: λ=1 GAE = MC return − V(s_t)."""
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    dones = np.zeros(3, dtype=bool)
    last_value = 1.5
    gamma = 0.9
    out = compute_gae(rewards, values, last_value, dones, gamma=gamma, lam=1.0)
    # Â_t = (Σ γ^l r_{t+l}) + γ^(T-t) · last_value − V(s_t)
    expected_a0 = (rewards[0] + gamma * rewards[1] + gamma**2 * rewards[2]
                    + gamma**3 * last_value - values[0])
    expected_a1 = (rewards[1] + gamma * rewards[2]
                    + gamma**2 * last_value - values[1])
    expected_a2 = rewards[2] + gamma * last_value - values[2]
    expected = np.array([expected_a0, expected_a1, expected_a2], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_closed_form_three_step_gae() -> None:
    """Hand-computed reference for a 3-step trajectory with γ=0.99, λ=0.95."""
    gamma, lam = 0.99, 0.95
    rewards = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    values = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    dones = np.zeros(3, dtype=bool)
    last_value = 0.6
    # Reverse compute by hand
    d2 = rewards[2] + gamma * last_value - values[2]
    a2 = d2
    d1 = rewards[1] + gamma * values[2] - values[1]
    a1 = d1 + gamma * lam * a2
    d0 = rewards[0] + gamma * values[1] - values[0]
    a0 = d0 + gamma * lam * a1
    expected = np.array([a0, a1, a2], dtype=np.float32)
    out = compute_gae(rewards, values, last_value, dones, gamma=gamma, lam=lam)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_terminal_truncates_bootstrap() -> None:
    """done at step k zeroes the bootstrap *for that step*: no contribution from V(s_{k+1})."""
    gamma, lam = 0.99, 0.95
    rewards = np.array([1.0, 0.5, 0.5], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, True, False], dtype=bool)  # done at step 1
    last_value = 99.0
    out = compute_gae(rewards, values, last_value, dones, gamma=gamma, lam=lam)
    # Step 2: ordinary bootstrap from last_value
    a2 = rewards[2] + gamma * last_value - values[2]
    # Step 1: done → no bootstrap from V(s_2), GAE-tail also zeroed
    a1 = rewards[1] - values[1]
    # Step 0: normal recursion using V(s_1) and a1
    a0 = (rewards[0] + gamma * values[1] - values[0]
           + gamma * lam * a1)
    expected = np.array([a0, a1, a2], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)
