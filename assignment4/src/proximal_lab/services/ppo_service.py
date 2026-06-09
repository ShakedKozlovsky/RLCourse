"""End-to-end PPO + GAE training loop (slide-18 pipeline).

Per iteration:
    1. Collect `steps_per_rollout` transitions per env using the current policy
    2. Bootstrap last value + run GAE to fill advantages / returns
    3. ``ppo_update`` runs the K-epoch minibatch update + diagnostics
    4. Log + record one ``IterationDiagnostics`` row
"""

from __future__ import annotations

import numpy as np
import torch
from torch import optim

from proximal_lab.environment.vector_env import SyncVectorEnv
from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.services.ppo_update import ppo_update
from proximal_lab.services.rollout_buffer import RolloutBuffer
from proximal_lab.shared.logger import get_logger
from proximal_lab.shared.types import IterationDiagnostics, TrainResult

_logger = get_logger(__name__)


class PPOService:
    """Train an :class:`ActorCriticNet` with PPO + GAE on a :class:`SyncVectorEnv`."""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        n_epochs_per_update: int = 10,
        minibatch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl_stop: float | None = None,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_eps <= 0 or n_epochs_per_update < 1 or minibatch_size < 1:
            raise ValueError("clip_eps, n_epochs_per_update, minibatch_size must be > 0")
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.lr = float(lr)
        self.clip_eps = float(clip_eps)
        self.n_epochs = int(n_epochs_per_update)
        self.minibatch_size = int(minibatch_size)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = target_kl_stop

    def fit(
        self,
        net: ActorCriticNet,
        env: SyncVectorEnv,
        total_timesteps: int,
        steps_per_rollout: int = 2048,
    ) -> TrainResult:
        """Train ``net`` for ``total_timesteps`` env steps."""
        if total_timesteps < 1 or steps_per_rollout < 1:
            raise ValueError("total_timesteps and steps_per_rollout must be >= 1")
        buf = RolloutBuffer(steps_per_rollout, env.n_envs, net.obs_dim, net.action_dim)
        opt = optim.Adam(net.parameters(), lr=self.lr)
        diagnostics: list[IterationDiagnostics] = []
        episode_rewards: list[float] = []
        obs = env.reset(seed=0)
        ep_rewards = np.zeros(env.n_envs, dtype=np.float32)
        iteration, timestep = 0, 0
        while timestep < total_timesteps:
            buf.reset()
            iter_episode_rewards: list[float] = []
            for _ in range(steps_per_rollout):
                obs_t = torch.from_numpy(obs).float()
                with torch.no_grad():
                    action, log_prob, value = net.act(obs_t)
                action_np = action.numpy()
                next_obs, reward, done, _ = env.step(action_np)
                buf.add(obs, action_np, log_prob.numpy(), value.numpy(), reward, done)
                ep_rewards += reward
                for env_i in range(env.n_envs):
                    if done[env_i]:
                        iter_episode_rewards.append(float(ep_rewards[env_i]))
                        ep_rewards[env_i] = 0.0
                obs = next_obs
                timestep += env.n_envs
            with torch.no_grad():
                _, _, last_values = net.act(torch.from_numpy(obs).float())
            buf.compute_advantages_and_returns(
                last_values.numpy(), gamma=self.gamma, lam=self.gae_lambda)
            diag = ppo_update(
                net, opt, buf,
                clip_eps=self.clip_eps, n_epochs=self.n_epochs,
                minibatch_size=self.minibatch_size, value_coef=self.value_coef,
                entropy_coef=self.entropy_coef, max_grad_norm=self.max_grad_norm,
                target_kl=self.target_kl,
            )
            mean_reward = (float(np.mean(iter_episode_rewards))
                           if iter_episode_rewards else float(ep_rewards.mean()))
            diagnostics.append(IterationDiagnostics(
                iteration=iteration, timestep=timestep,
                mean_episode_reward=mean_reward, **diag,
            ))
            episode_rewards.extend(iter_episode_rewards)
            _logger.info(
                "iter=%d ts=%d ep_reward=%.2f kl=%.4f clip_frac=%.3f ev=%.3f",
                iteration, timestep, mean_reward,
                diag["mean_kl"], diag["clip_fraction"], diag["explained_variance"],
            )
            iteration += 1
        final = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
        return TrainResult(
            diagnostics=diagnostics,
            episode_rewards=episode_rewards,
            final_mean_reward=final,
            total_timesteps=timestep,
        )
