"""NormalisedEnv + RunningMeanStd contracts."""

from __future__ import annotations

import numpy as np
import pytest

from proximal_lab.environment.mujoco_env import RunningMeanStd, make_env


def test_running_stats_match_numpy_reference() -> None:
    rms = RunningMeanStd.for_shape((3,))
    samples = np.random.default_rng(0).normal(size=(500, 3))
    rms.update(samples)
    # Welford's initialiser count=1e-4 introduces ~1e-7 drift; that's tighter
    # than the float32 we end up using anyway.
    np.testing.assert_allclose(rms.mean, samples.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(rms.var, samples.var(axis=0), atol=1e-4)


def test_running_stats_handle_1d_input() -> None:
    rms = RunningMeanStd.for_shape((2,))
    rms.update(np.array([1.0, 2.0]))
    np.testing.assert_allclose(rms.mean, [1.0, 2.0], atol=0.01)


def test_running_stats_streaming_matches_batch() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(200, 4))
    rms_stream = RunningMeanStd.for_shape((4,))
    for batch_start in range(0, 200, 20):
        rms_stream.update(samples[batch_start : batch_start + 20])
    rms_batch = RunningMeanStd.for_shape((4,))
    rms_batch.update(samples)
    np.testing.assert_allclose(rms_stream.mean, rms_batch.mean, atol=1e-9)
    np.testing.assert_allclose(rms_stream.var, rms_batch.var, atol=1e-4)


def test_halfcheetah_step_shapes() -> None:
    env = make_env("HalfCheetah-v5", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (17,)
    assert env.action_space.shape == (6,)
    action = env.action_space.sample()
    obs2, r, term, trunc, _ = env.step(action)
    assert obs2.shape == (17,)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_walker2d_step_shapes() -> None:
    env = make_env("Walker2d-v5", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (17,)
    assert env.action_space.shape == (6,)


def test_set_training_freezes_rms() -> None:
    env = make_env("HalfCheetah-v5", seed=0)
    env.set_training(False)
    mean_before = env.rms.mean.copy()
    var_before = env.rms.var.copy()
    for _ in range(50):
        env.step(env.action_space.sample())
    np.testing.assert_array_equal(env.rms.mean, mean_before)
    np.testing.assert_array_equal(env.rms.var, var_before)


def test_set_training_thaws_rms() -> None:
    env = make_env("HalfCheetah-v5", seed=0)
    env.set_training(False)
    env.set_training(True)
    mean_before = env.rms.mean.copy()
    for _ in range(20):
        env.step(env.action_space.sample())
    assert not np.allclose(env.rms.mean, mean_before)


@pytest.mark.parametrize("env_id", ["HalfCheetah-v5", "Walker2d-v5"])
def test_normalised_obs_has_finite_floats(env_id: str) -> None:
    env = make_env(env_id, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
