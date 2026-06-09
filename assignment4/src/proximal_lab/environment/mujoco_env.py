"""MuJoCo env wrapper + running-mean / running-std observation normaliser.

The standard PPO recipe normalises observations to roughly unit variance using
streaming statistics updated during rollouts. At eval time the normaliser is
frozen so the eval distribution doesn't drift.

See ``docs/PRD_mujoco_env.md`` for the full theory + caveats.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class RunningMeanStd:
    """Welford's online algorithm — numerically stable streaming mean + variance."""

    mean: np.ndarray
    var: np.ndarray
    count: float = 1e-4

    @classmethod
    def for_shape(cls, shape: tuple[int, ...]) -> RunningMeanStd:
        return cls(mean=np.zeros(shape, dtype=np.float64),
                   var=np.ones(shape, dtype=np.float64))

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of observations (shape ``(N, *)``)."""
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Return ``(x − mean) / sqrt(var + eps)`` without mutating statistics."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class NormalisedEnv:
    """``gym.Env`` wrapper that maintains a ``RunningMeanStd`` over observations.

    The class is intentionally minimal — it does not subclass ``gym.Wrapper`` so
    we can vectorise easily and avoid the Gymnasium wrapper API surface.
    """

    def __init__(self, env: gym.Env, training: bool = True):
        self._env = env
        self._training = bool(training)
        obs_shape: tuple[int, ...] = env.observation_space.shape  # type: ignore[assignment]
        self.rms = RunningMeanStd.for_shape(obs_shape)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed)
        if self._training:
            self.rms.update(np.asarray(obs, dtype=np.float64))
        return self.rms.normalise(obs).astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._training:
            self.rms.update(np.asarray(obs, dtype=np.float64))
        return (self.rms.normalise(obs).astype(np.float32), float(reward),
                bool(terminated), bool(truncated), info)

    def close(self) -> None:
        self._env.close()

    def set_training(self, training: bool) -> None:
        """Freeze (False) or thaw (True) the running statistics."""
        self._training = bool(training)


def make_env(env_id: str, seed: int = 0, training: bool = True) -> NormalisedEnv:
    """Construct a single ``NormalisedEnv`` ready for PPO use."""
    env = gym.make(env_id)
    env.action_space.seed(seed)
    wrapped = NormalisedEnv(env, training=training)
    wrapped.reset(seed=seed)
    return wrapped
