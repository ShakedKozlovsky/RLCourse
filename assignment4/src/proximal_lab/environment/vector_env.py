"""Synchronous vectorised env — n parallel ``NormalisedEnv`` instances stepped in lockstep.

We deliberately avoid Gymnasium's ``SyncVectorEnv`` here because we want a
shared ``RunningMeanStd`` across all sub-envs (one consistent normaliser the
PPO actor sees regardless of which sub-env produced an observation).
"""

from __future__ import annotations

import numpy as np

from proximal_lab.environment.mujoco_env import NormalisedEnv, RunningMeanStd, make_env


class SyncVectorEnv:
    """Drives ``n`` ``NormalisedEnv`` instances in lockstep with a shared RMS."""

    def __init__(self, env_id: str, n_envs: int, seed: int = 0, training: bool = True):
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = int(n_envs)
        self._training = bool(training)
        # Disable per-env training so only the shared RMS updates.
        self._envs: list[NormalisedEnv] = [
            make_env(env_id, seed=seed + i, training=False) for i in range(self.n_envs)
        ]
        obs_shape = self._envs[0].observation_space.shape
        self.shared_rms = RunningMeanStd.for_shape(obs_shape)
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self._last_obs = np.zeros((self.n_envs, *obs_shape), dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        observations = []
        for i, env in enumerate(self._envs):
            obs, _ = env.reset(seed=None if seed is None else seed + i)
            # Un-normalise back to the raw obs to update the shared RMS.
            raw = self._unnormalise(obs, env)
            observations.append(raw)
        batch = np.stack(observations)
        if self._training:
            self.shared_rms.update(batch.astype(np.float64))
        normalised = self.shared_rms.normalise(batch).astype(np.float32)
        self._last_obs = normalised
        return normalised

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if actions.shape[0] != self.n_envs:
            raise ValueError(f"actions batch size {actions.shape[0]} != n_envs {self.n_envs}")
        raws, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self._envs):
            obs, r, term, trunc, info = env.step(actions[i])
            raw = self._unnormalise(obs, env)
            if term or trunc:
                obs_reset, _ = env.reset(seed=None)
                raw = self._unnormalise(obs_reset, env)
            raws.append(raw)
            rewards.append(r)
            dones.append(term or trunc)
            infos.append(info)
        batch = np.stack(raws)
        if self._training:
            self.shared_rms.update(batch.astype(np.float64))
        normalised = self.shared_rms.normalise(batch).astype(np.float32)
        self._last_obs = normalised
        return (normalised, np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    @staticmethod
    def _unnormalise(obs: np.ndarray, env: NormalisedEnv) -> np.ndarray:
        """Recover the raw obs by inverting the per-env normaliser (which is frozen)."""
        return obs.astype(np.float64) * np.sqrt(env.rms.var + 1e-8) + env.rms.mean

    def close(self) -> None:
        for env in self._envs:
            env.close()

    def set_training(self, training: bool) -> None:
        self._training = bool(training)
