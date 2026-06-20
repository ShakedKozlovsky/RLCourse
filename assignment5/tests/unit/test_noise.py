"""Layer 6 — noise + schedule tests."""

from __future__ import annotations

import numpy as np
import pytest

from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.ou import OUNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule


def test_gaussian_validates_inputs() -> None:
    with pytest.raises(ValueError):
        GaussianNoise(action_dim=0, sigma=0.1)
    with pytest.raises(ValueError):
        GaussianNoise(action_dim=2, sigma=-0.1)


def test_gaussian_mean_zero_large_n() -> None:
    n = GaussianNoise(action_dim=2, sigma=0.2,
                      rng=np.random.default_rng(42))
    samples = np.stack([n.sample() for _ in range(10000)])
    assert np.abs(samples.mean(axis=0)).max() < 0.02


def test_gaussian_variance_matches_sigma_squared() -> None:
    n = GaussianNoise(action_dim=2, sigma=0.3,
                      rng=np.random.default_rng(7))
    samples = np.stack([n.sample() for _ in range(10000)])
    np.testing.assert_allclose(samples.std(axis=0), 0.3, rtol=0.05)


def test_gaussian_set_sigma() -> None:
    n = GaussianNoise(action_dim=2, sigma=0.1)
    n.set_sigma(0.05)
    assert n.sigma == 0.05
    with pytest.raises(ValueError):
        n.set_sigma(-1.0)


def test_ou_lag_one_autocorrelation_positive() -> None:
    ou = OUNoise(action_dim=1, theta=0.15, mu=0.0, sigma=0.2,
                 rng=np.random.default_rng(0))
    seq = np.stack([ou.sample() for _ in range(1000)]).squeeze()
    corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
    assert corr > 0.6


def test_ou_mean_reverts_to_mu() -> None:
    ou = OUNoise(action_dim=1, theta=0.5, mu=0.7, sigma=0.05,
                 rng=np.random.default_rng(0))
    samples = np.stack([ou.sample() for _ in range(5000)])
    assert abs(samples[2000:].mean() - 0.7) < 0.05


def test_ou_reset_returns_to_mu() -> None:
    ou = OUNoise(action_dim=2, theta=0.15, mu=0.5, sigma=0.2,
                 rng=np.random.default_rng(0))
    for _ in range(50):
        ou.sample()
    ou.reset()
    np.testing.assert_allclose(ou._state, 0.5)  # noqa: SLF001


def test_schedule_initial_and_final() -> None:
    s = LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=1000)
    assert s.at(0) == pytest.approx(0.2)
    assert s.at(1000) == pytest.approx(0.05)
    assert s.at(5000) == pytest.approx(0.05)


def test_schedule_midpoint() -> None:
    s = LinearSigmaSchedule(initial=0.2, final=0.0, decay_steps=1000)
    assert s.at(500) == pytest.approx(0.1)


def test_schedule_validates_decay_steps() -> None:
    with pytest.raises(ValueError):
        LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=0)


def test_schedule_validates_sigma() -> None:
    with pytest.raises(ValueError):
        LinearSigmaSchedule(initial=-0.1, final=0.05, decay_steps=100)
