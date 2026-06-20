"""Algorithm comparison — DDPG (Gaussian) vs DDPG (OU) vs TD3 vs no-replay DDPG.

Addresses TA findings:
  M4 (TD3 unbenchmarked)  — DDPG vs TD3 head-to-head
  M5 (Q1 lacks evidence)  — DDPG vs no-replay-DDPG (≈ on-policy proxy)
  m6 (OU never benchmarked) — Gaussian vs OU

3 seeds × ~6 algorithm variants × 4 000 steps each on the primary apartment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.model.td3_network import TD3Network
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.ou import OUNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.services.ddpg_service import DDPGHyperparams, DDPGService
from roomba_lab.services.td3_update import apply_td3_update
from roomba_lab.shared.seed import set_global_seed
from roomba_lab.shared.types import StepDiagnostic, TrainResult, Transition

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 4000
N_SEEDS = 3


def _make_buffer(lab: RoombaLab, env, rng: np.random.Generator) -> ReplayBuffer:
    return ReplayBuffer(int(lab.config.get("ddpg.replay_capacity")),
                         env.obs_dim, env.action_dim, rng=rng)


def _make_hp(lab: RoombaLab) -> DDPGHyperparams:
    c = lab.config
    return DDPGHyperparams(
        gamma=float(c.get("ddpg.gamma")), tau=float(c.get("ddpg.tau")),
        actor_lr=float(c.get("ddpg.actor_lr")), critic_lr=float(c.get("ddpg.critic_lr")),
        batch_size=int(c.get("ddpg.batch_size")),
        warmup_steps=int(c.get("ddpg.warmup_steps")),
        max_grad_norm=float(c.get("ddpg.max_grad_norm")),
        log_interval=int(c.get("training.log_interval")),
    )


def run_ddpg(lab: RoombaLab, noise_kind: str, seed: int,
              no_replay: bool = False) -> TrainResult:
    set_global_seed(seed)
    env = lab.make_env()
    rng = np.random.default_rng(seed)
    net = ActorCriticNet(obs_dim=env.obs_dim, action_dim=env.action_dim,
                          actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
                          critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")))
    if noise_kind == "gaussian":
        noise = GaussianNoise(env.action_dim, sigma=0.2, rng=rng)
    else:
        noise = OUNoise(env.action_dim, theta=0.15, mu=0.0, sigma=0.2, rng=rng)
    schedule = LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=50000)
    hp = _make_hp(lab)
    # The "no_replay" variant uses a buffer-of-1 (≈ on-policy: only sees the most
    # recent transition)
    capacity = 1 if no_replay else int(lab.config.get("ddpg.replay_capacity"))
    buf = ReplayBuffer(capacity, env.obs_dim, env.action_dim, rng=rng)
    svc = DDPGService(net, env, buf, noise, schedule, hp)
    return svc.fit(total_timesteps=TOTAL_TS, seed=seed)


def run_td3(lab: RoombaLab, seed: int) -> TrainResult:
    """TD3 mini-loop — mirrors DDPGService.fit but uses TD3 network + update."""
    set_global_seed(seed)
    env = lab.make_env()
    rng = np.random.default_rng(seed)
    net = TD3Network(obs_dim=env.obs_dim, action_dim=env.action_dim,
                      actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
                      critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")))
    buf = _make_buffer(lab, env, rng)
    noise = GaussianNoise(env.action_dim, sigma=0.2, rng=rng)
    schedule = LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=50000)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=float(lab.config.get("ddpg.actor_lr")))
    critic_params = list(net.critic_a.parameters()) + list(net.critic_b.parameters())
    critic_opt = torch.optim.Adam(critic_params, lr=float(lab.config.get("ddpg.critic_lr")))
    hp = _make_hp(lab)
    result = TrainResult()
    obs = env.reset(seed=seed)
    episode_reward = 0.0
    for step in range(TOTAL_TS):
        noise.set_sigma(schedule.at(step))
        if step < hp.warmup_steps:
            action = np.random.uniform(-1, 1, size=(env.action_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                a = net.actor(torch.as_tensor(obs).unsqueeze(0)).cpu().numpy()[0]
            action = np.clip(a + noise.sample(), -1, 1).astype(np.float32)
        next_obs, r, done, info = env.step(action)
        episode_reward += r
        buf.push(Transition(obs, action, float(r), next_obs, bool(done)))
        actor_l = critic_l = mean_q = 0.0
        if len(buf) >= max(hp.batch_size, hp.warmup_steps):
            batch = buf.sample(hp.batch_size)
            diag = apply_td3_update(net, batch, step, hp.gamma, hp.tau,
                                      policy_delay=2, target_policy_noise=0.2,
                                      target_noise_clip=0.5, actor_opt=actor_opt,
                                      critic_opt=critic_opt, max_grad_norm=hp.max_grad_norm)
            actor_l, critic_l, mean_q = diag.actor_loss, diag.critic_loss, diag.mean_q
        obs = next_obs
        if step % hp.log_interval == 0:
            result.diagnostics.append(StepDiagnostic(
                step=step, actor_loss=actor_l, critic_loss=critic_l, mean_q=mean_q,
                sigma=noise.sigma, episode_reward=episode_reward, coverage=info["coverage"]))
        if done:
            obs = env.reset(seed=seed + step)
            episode_reward = 0.0
            noise.reset()
    return result


def main() -> None:
    t0 = time.time()
    lab = RoombaLab()
    variants = [
        ("ddpg_gaussian", lambda s: run_ddpg(lab, "gaussian", s)),
        ("ddpg_ou", lambda s: run_ddpg(lab, "ou", s)),
        ("td3", lambda s: run_td3(lab, s)),
        ("ddpg_no_replay", lambda s: run_ddpg(lab, "gaussian", s, no_replay=True)),
    ]
    out_data = {"variants": {}, "n_seeds": N_SEEDS, "total_timesteps": TOTAL_TS}
    for name, runner in variants:
        per_seed = []
        for seed in range(N_SEEDS):
            t_v = time.time()
            r = runner(seed)
            last = r.diagnostics[-1] if r.diagnostics else None
            per_seed.append({
                "seed": seed,
                "final_reward": float(last.episode_reward) if last else 0.0,
                "final_coverage": float(last.coverage) if last else 0.0,
                "final_critic_loss": float(last.critic_loss) if last else 0.0,
            })
            print(f"  [{name} seed={seed}] reward={per_seed[-1]['final_reward']:7.1f} "
                   f"cov={per_seed[-1]['final_coverage']:.3f} ({time.time() - t_v:.1f}s)")
        out_data["variants"][name] = per_seed
    out = ROOT / "results" / "algorithms" / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
