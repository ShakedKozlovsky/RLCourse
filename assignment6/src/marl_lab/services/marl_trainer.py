"""CTDE end-to-end trainer — collects episodes, learns from a centralised buffer.

Composes layers 4 (Q-net) + 5/6 (mixers) + 8 (buffer) + 9 (ε-greedy) +
10/11 (updaters). Algorithm-agnostic: ``algo='qmix' | 'vdn' | 'iql'``.

Per training step:
  1. Run one full episode under ε-greedy (epoch-decayed)
  2. Push the episode into the centralised buffer
  3. If buffer.size >= warmup_episodes: sample a batch and apply the algo's
     update step (qmix/vdn/iql)
  4. Periodically: log mean episode reward / win rate / critic_loss"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv
from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.noise.epsilon_greedy import select_action
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.iql_update import apply_iql_update
from marl_lab.services.qmix_update import apply_qmix_update
from marl_lab.services.trainer_build import (
    build_mixer_pair,
    build_optimisers,
    build_q_nets,
)
from marl_lab.services.vdn_update import apply_vdn_update
from marl_lab.shared.types import EpisodeSequence, Transition

AGENTS = ("cop", "thief")


@dataclass
class TrainerConfig:
    """Hyperparameters for the trainer (mirrors yaml `marl` block)."""
    algo: str = "qmix"                 # 'qmix' | 'vdn' | 'iql'
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 1000
    warmup_episodes: int = 16
    max_seq_len: int = 25
    embed_dim: int = 32
    hyper_hidden: int = 64
    gru_hidden_size: int = 64
    hidden_sizes: tuple[int, ...] = (128, 128)


@dataclass
class TrainerDiagnostic:
    """Per-train-step metrics (returned by step + summarised by trainer)."""
    episode_reward_cop: float = 0.0
    episode_reward_thief: float = 0.0
    episode_steps: int = 0
    winner: str | None = None
    critic_loss: float = 0.0
    mean_q_cop: float = 0.0
    mean_q_thief: float = 0.0
    epsilon: float = 1.0
    history: list[dict] = field(default_factory=list)


class MarlTrainer:
    """CTDE MARL trainer: runs episodes, learns from a centralised buffer.

    The trainer owns: per-agent live + target Q-nets, the chosen mixer (+ target
    if not IQL), the centralised replay buffer, optimisers, and the ε schedule.
    Public methods: ``collect_episode()``, ``learn_step()``, ``train(n_episodes)``."""

    def __init__(self, env: DecPomdpEnv, cfg: TrainerConfig,
                 epsilon_schedule: LinearEpsilonSchedule,
                 device: torch.device | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.env = env
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self._rng = rng or np.random.default_rng(0)
        self.eps_schedule = epsilon_schedule
        n_cop, n_thief = 6, 5    # static for this env; thief uses sub-set
        n_actions = max(n_cop, n_thief)
        state_dim = env.global_state().shape[0]
        self.q_nets, self.target_q_nets = build_q_nets(env, cfg, self.device, n_actions)
        self.mixer, self.target_mixer = build_mixer_pair(cfg, state_dim, self.device)
        self.opts = build_optimisers(cfg, self.q_nets, self.mixer)
        # Centralised replay buffer
        self.buffer = CentralisedReplayBuffer(
            capacity=cfg.buffer_capacity, max_seq_len=cfg.max_seq_len,
            state_dim=state_dim, obs_dim=env.obs_dim,
            n_actions_per_agent={"cop": n_cop, "thief": n_thief},
            rng=self._rng,
        )
        self.global_step = 0
        self.episode_count = 0

    def _select_joint_action(self, joint_obs: dict[str, np.ndarray],
                              hidden: dict[str, torch.Tensor],
                              epsilon: float) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
        """Pick actions for both agents under ε-greedy on per-agent live Q-nets."""
        joint_action: dict[str, int] = {}
        new_hidden: dict[str, torch.Tensor] = {}
        for a in AGENTS:
            with torch.no_grad():
                obs_t = torch.as_tensor(joint_obs[a], device=self.device, dtype=torch.float32)
                q_seq, h_new = self.q_nets[a](obs_t.unsqueeze(0), hidden=hidden[a])
            q_vals = q_seq.squeeze(0).squeeze(0).cpu().numpy()
            new_hidden[a] = h_new
            n_legal = 6 if a == "cop" else 5
            mask = np.zeros(q_vals.shape[0], dtype=bool)
            mask[:n_legal] = True
            joint_action[a] = select_action(q_vals, epsilon=epsilon, rng=self._rng,
                                              action_mask=mask)
        return joint_action, new_hidden

    def collect_episode(self, seed: int | None = None) -> tuple[EpisodeSequence, TrainerDiagnostic]:
        """Run ONE episode in the env and return its full transition sequence."""
        joint_obs = self.env.reset(seed=seed)
        global_state = self.env.global_state()
        hidden = {a: self.q_nets[a].init_hidden(batch_size=1, device=self.device)
                  for a in AGENTS}
        eps = self.eps_schedule.at(self.episode_count)
        ep = EpisodeSequence()
        reward_acc = {"cop": 0.0, "thief": 0.0}
        winner: str | None = None
        steps = 0
        while True:
            joint_action, hidden = self._select_joint_action(joint_obs, hidden, eps)
            next_joint_obs, reward, done, info = self.env.step(joint_action)
            next_global_state = self.env.global_state()
            ep.transitions.append(Transition(
                global_state=global_state,
                joint_obs={a: joint_obs[a].copy() for a in AGENTS},
                joint_action=dict(joint_action),
                joint_reward=dict(reward),
                next_global_state=next_global_state,
                next_joint_obs={a: next_joint_obs[a].copy() for a in AGENTS},
                done=done,
            ))
            reward_acc["cop"] += reward["cop"]
            reward_acc["thief"] += reward["thief"]
            steps += 1
            joint_obs = next_joint_obs
            global_state = next_global_state
            if done:
                winner = info["winner"]
                break
        self.episode_count += 1
        diag = TrainerDiagnostic(
            episode_reward_cop=reward_acc["cop"],
            episode_reward_thief=reward_acc["thief"],
            episode_steps=steps,
            winner=winner,
            epsilon=eps,
        )
        return ep, diag

    def learn_step(self) -> dict:
        """One learning update from the buffer (if warmup is satisfied)."""
        if len(self.buffer) < self.cfg.warmup_episodes:
            return {"skipped": True, "reason": "warmup"}
        batch = self.buffer.sample(min(self.cfg.batch_size, len(self.buffer)))
        if self.cfg.algo == "qmix":
            assert self.mixer is not None and self.target_mixer is not None
            d = apply_qmix_update(
                q_nets=self.q_nets, target_q_nets=self.target_q_nets,
                mixer=self.mixer, target_mixer=self.target_mixer,  # type: ignore[arg-type]
                batch=batch, gamma=self.cfg.gamma, tau=self.cfg.tau,
                critic_opt=self.opts,  # type: ignore[arg-type]
                device=self.device,
            )
            return {"skipped": False, "critic_loss": d.critic_loss,
                    "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief,
                    "target_drift": d.target_drift}
        if self.cfg.algo == "vdn":
            assert self.mixer is not None and self.target_mixer is not None
            d = apply_vdn_update(
                q_nets=self.q_nets, target_q_nets=self.target_q_nets,
                mixer=self.mixer, target_mixer=self.target_mixer,  # type: ignore[arg-type]
                batch=batch, gamma=self.cfg.gamma, tau=self.cfg.tau,
                critic_opt=self.opts,  # type: ignore[arg-type]
                device=self.device,
            )
            return {"skipped": False, "critic_loss": d.critic_loss,
                    "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief}
        # iql
        d_iql = apply_iql_update(
            q_nets=self.q_nets, target_q_nets=self.target_q_nets,
            batch=batch, gamma=self.cfg.gamma, tau=self.cfg.tau,
            critic_opts=self.opts,  # type: ignore[arg-type]
            device=self.device,
        )
        return {"skipped": False,
                "critic_loss": (d_iql.critic_loss_cop + d_iql.critic_loss_thief) / 2.0,
                "mean_q_cop": d_iql.mean_q_cop, "mean_q_thief": d_iql.mean_q_thief}

    def train(self, n_episodes: int) -> list[TrainerDiagnostic]:
        """Train for ``n_episodes`` episodes; return the per-episode diagnostics."""
        history: list[TrainerDiagnostic] = []
        for _ in range(n_episodes):
            ep, diag = self.collect_episode()
            self.buffer.push(ep)
            learn_info = self.learn_step()
            if not learn_info["skipped"]:
                diag.critic_loss = learn_info["critic_loss"]
                diag.mean_q_cop = learn_info["mean_q_cop"]
                diag.mean_q_thief = learn_info["mean_q_thief"]
            history.append(diag)
        return history
