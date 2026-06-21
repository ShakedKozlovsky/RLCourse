"""True on-policy DDPG ablation — Layer 28 closes TA M5.

The earlier "no-replay" variant used `capacity=1` + `batch_size=128`, which
caused the update step's gate (`len(buf) ≥ max(128, warmup_steps=1000)`) to
never fire — the agent never trained. The TA correctly called this tautological.

This script implements a REAL on-policy ablation:
  - Each step: push the single (s, a, r, s', d) transition
  - Then immediately train on JUST THAT ONE transition (batch_size=1)
  - No warmup, no replay sampling — pure online TD update

Comparable to single-sample TD learning. Should be unstable and learn poorly,
confirming Q1's claim that off-policy batched replay is what makes DDPG work."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.services.ddpg_update import apply_update
from roomba_lab.shared.seed import set_global_seed

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 4000
N_SEEDS = 3


def run_one(seed: int) -> dict:
    """Single seed: pure online DDPG (no replay, no batching)."""
    set_global_seed(seed)
    lab = RoombaLab()
    env = lab.make_env()
    rng = np.random.default_rng(seed)
    net = ActorCriticNet(obs_dim=env.obs_dim, action_dim=env.action_dim,
                          actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
                          critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
                          actor_head_gain=float(lab.config.get("ddpg.actor_head_gain", 0.1)))
    actor_opt = torch.optim.Adam(net.actor.parameters(),
                                    lr=float(lab.config.get("ddpg.actor_lr")))
    critic_opt = torch.optim.Adam(net.critic.parameters(),
                                     lr=float(lab.config.get("ddpg.critic_lr")))
    noise = GaussianNoise(env.action_dim, sigma=0.2, rng=rng)
    schedule = LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=TOTAL_TS)
    gamma = float(lab.config.get("ddpg.gamma"))
    tau = float(lab.config.get("ddpg.tau"))
    obs = env.reset(seed=seed)
    episode_reward = 0.0
    final_reward = final_cov = final_q = 0.0
    for step in range(TOTAL_TS):
        noise.set_sigma(schedule.at(step))
        with torch.no_grad():
            a = net.actor(torch.as_tensor(obs).unsqueeze(0)).cpu().numpy()[0]
        action = np.clip(a + noise.sample(), -1, 1).astype(np.float32)
        next_obs, r, done, info = env.step(action)
        episode_reward += r
        # Train on JUST this one transition (batch_size = 1 = on-policy)
        batch_single = {
            "state": np.expand_dims(obs, 0),
            "action": np.expand_dims(action, 0),
            "reward": np.array([float(r)], dtype=np.float32),
            "next_state": np.expand_dims(next_obs, 0),
            "done": np.array([float(done)], dtype=np.float32),
        }
        diag = apply_update(net, batch_single, gamma, tau, actor_opt, critic_opt,
                              max_grad_norm=1.0)
        final_reward = episode_reward
        final_cov = info["coverage"]
        final_q = diag.mean_q
        obs = next_obs
        if done:
            obs = env.reset(seed=seed + step)
            episode_reward = 0.0
            noise.reset()
    return {"seed": seed, "final_reward": float(final_reward),
            "final_coverage": float(final_cov), "final_q": float(final_q)}


def main() -> None:
    t0 = time.time()
    rows = []
    for seed in range(N_SEEDS):
        t = time.time()
        r = run_one(seed)
        rows.append(r)
        print(f"  [on-policy DDPG seed={seed}] "
               f"reward={r['final_reward']:7.1f}  cov={r['final_coverage']:.4f}  "
               f"q={r['final_q']:.2f}  ({time.time()-t:.1f}s)")
    payload = {"variant": "ddpg_true_on_policy",
                "description": "batch_size=1, no replay sampling — each step trains on just the latest transition",
                "n_seeds": N_SEEDS, "total_timesteps": TOTAL_TS, "rows": rows}
    out = ROOT / "results" / "algorithms" / "true_on_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    rewards = [r["final_reward"] for r in rows]
    covs = [r["final_coverage"] for r in rows]
    print(f"\nwrote {out}  ({time.time()-t0:.1f}s)")
    print(f"  mean reward = {sum(rewards)/N_SEEDS:.1f}, "
           f"mean coverage = {sum(covs)/N_SEEDS:.4f}")


if __name__ == "__main__":
    main()
