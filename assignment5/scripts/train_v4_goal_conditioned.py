"""Layer 30 — goal-conditioned DDPG: nearest-unvisited direction added to obs.

After 2 negative results (§ 9a boosted reward, § 9b smaller net), the TA
diagnosis is that the LIDAR-only observation cannot represent "where to go"
for a cleaning task — only "where the walls are right here."

This script adds 2 frontier-direction features (dx, dy to nearest-unvisited
cell, normalised to bbox span) → obs_dim 29 → 31. Same DDPG hyperparameters
as v1.20; same 20k training budget.

Target: cov > 0.10 (substantive improvement over v1.20's 0.045)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.environment.goal_obs import GoalConditionedEnv
from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.services.ddpg_service import DDPGHyperparams, DDPGService
from roomba_lab.tools.viz.plots import (
    plot_coverage_heatmap,
    plot_critic_loss,
    plot_learning_curve,
    plot_trajectory_overlay,
)

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 20000


def main() -> None:
    t0 = time.time()
    lab = RoombaLab()
    base_env = lab.make_env()
    env = GoalConditionedEnv(base_env)
    import numpy as np
    rng = np.random.default_rng(int(lab.config.get("seed")))
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
        actor_head_gain=float(lab.config.get("ddpg.actor_head_gain", 0.1)),
    )
    buf = ReplayBuffer(int(lab.config.get("ddpg.replay_capacity")),
                       env.obs_dim, env.action_dim, rng=rng)
    noise = GaussianNoise(env.action_dim,
                           sigma=float(lab.config.get("noise.sigma_initial")), rng=rng)
    schedule = LinearSigmaSchedule(
        initial=float(lab.config.get("noise.sigma_initial")),
        final=float(lab.config.get("noise.sigma_final")),
        decay_steps=TOTAL_TS,
    )
    hp = DDPGHyperparams(
        gamma=float(lab.config.get("ddpg.gamma")),
        tau=float(lab.config.get("ddpg.tau")),
        actor_lr=float(lab.config.get("ddpg.actor_lr")),
        critic_lr=float(lab.config.get("ddpg.critic_lr")),
        batch_size=int(lab.config.get("ddpg.batch_size")),
        warmup_steps=int(lab.config.get("ddpg.warmup_steps")),
        max_grad_norm=float(lab.config.get("ddpg.max_grad_norm")),
        log_interval=int(lab.config.get("training.log_interval")),
    )
    svc = DDPGService(net, env, buf, noise, schedule, hp)
    print(f"obs_dim with goal-conditioning: {env.obs_dim}  (default = 29)")
    result = svc.fit(total_timesteps=TOTAL_TS, seed=0)
    print(f"trained {TOTAL_TS} steps in {time.time()-t0:.1f}s")

    ckpt = ROOT / "saved_models" / "headline_policy_v4_goal.pt"
    torch.save(svc.net.state_dict(), ckpt)
    print(f"saved {ckpt}")
    plot_learning_curve(result, ROOT / "assets/plots/learning_curve_v4_goal.png",
                         title="DDPG Learning Curve — 20k, goal-conditioned (Layer 30)")
    plot_critic_loss(result, ROOT / "assets/plots/critic_loss_v4_goal.png",
                      title="Critic Loss — 20k, goal-conditioned (Layer 30)")
    loader = HouseExpoLoader(ROOT / "data/raw/sample_maps")
    verts = loader.load(loader.map_ids()[0]).verts
    plot_trajectory_overlay([(p.x, p.y) for p in env.robot.trajectory], verts,
                             ROOT / "assets/plots/trajectory_overlay_v4_goal.png",
                             title="20k goal-conditioned policy trajectory")
    plot_coverage_heatmap(env.world.grid, ROOT / "assets/plots/coverage_heatmap_v4_goal.png",
                           title="Final coverage — 20k goal-conditioned training")
    last = result.diagnostics[-1]
    print(f"final coverage={last.coverage:.3f}  episode_reward={last.episode_reward:.1f}")


if __name__ == "__main__":
    main()
