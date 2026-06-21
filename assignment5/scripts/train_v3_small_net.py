"""Layer 29 — third M1 attempt: smaller network + 50k training + proper LR schedule.

v2 (Layer 28) tried boosting reward — backfired. The TA suggested the v1.20
network ([256, 256]) might be over-parameterised for our small data budget.

This script:
  - Drops hidden sizes [256, 256] → [64, 64] (16× fewer params)
  - Keeps the v1.20 reward (proven well-tuned in Layer 28's negative result)
  - Trains 50 000 steps with cosine LR schedule
  - Target: cov > 0.10 (substantive improvement vs v1.20's 0.045)"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import time
from pathlib import Path

import torch

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.sdk.trainers import build_ddpg_service
from roomba_lab.services.ddpg_update import apply_update
from roomba_lab.shared.types import StepDiagnostic, TrainResult, Transition
from roomba_lab.tools.viz.plots import (
    plot_coverage_heatmap,
    plot_critic_loss,
    plot_learning_curve,
    plot_trajectory_overlay,
)

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 50000


def _cosine_lr(initial: float, step: int, total: int) -> float:
    """Cosine decay from initial → initial × 0.1 over `total` steps."""
    frac = min(1.0, step / max(1, total))
    return initial * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "setup.json").read_text())
    tuned = copy.deepcopy(base_cfg)
    tuned["ddpg"]["actor_hidden_sizes"] = [64, 64]
    tuned["ddpg"]["critic_hidden_sizes"] = [64, 64]
    tuned["noise"]["decay_steps"] = TOTAL_TS
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cfg_path = Path(f.name)
    cfg_path.write_text(json.dumps(tuned, indent=2))

    t0 = time.time()
    lab = RoombaLab(config_path=cfg_path)
    env = lab.make_env()
    svc = build_ddpg_service(lab.config, env)
    initial_actor_lr = float(lab.config.get("ddpg.actor_lr"))
    initial_critic_lr = float(lab.config.get("ddpg.critic_lr"))

    result = TrainResult()
    obs = env.reset(seed=0)
    episode_reward = 0.0
    for step in range(TOTAL_TS):
        # Cosine LR decay
        for g in svc.actor_opt.param_groups:
            g["lr"] = _cosine_lr(initial_actor_lr, step, TOTAL_TS)
        for g in svc.critic_opt.param_groups:
            g["lr"] = _cosine_lr(initial_critic_lr, step, TOTAL_TS)
        svc.noise.set_sigma(svc.schedule.at(step))
        action = svc._select_action(obs, step)  # noqa: SLF001
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        svc.buffer.push(Transition(state=obs, action=action,
                                     reward=float(reward), next_state=next_obs,
                                     done=bool(done)))
        obs = next_obs
        actor_l = critic_l = mean_q = 0.0
        if len(svc.buffer) >= max(svc.hp.batch_size, svc.hp.warmup_steps):
            batch = svc.buffer.sample(svc.hp.batch_size)
            diag = apply_update(svc.net, batch, svc.hp.gamma, svc.hp.tau,
                                 svc.actor_opt, svc.critic_opt,
                                 max_grad_norm=svc.hp.max_grad_norm)
            actor_l, critic_l, mean_q = diag.actor_loss, diag.critic_loss, diag.mean_q
        if step % svc.hp.log_interval == 0:
            result.diagnostics.append(StepDiagnostic(
                step=step, actor_loss=actor_l, critic_loss=critic_l,
                mean_q=mean_q, sigma=svc.noise.sigma,
                episode_reward=episode_reward, coverage=info["coverage"]))
        if done:
            obs = env.reset(seed=step)
            episode_reward = 0.0
            svc.noise.reset()
    print(f"trained {TOTAL_TS} steps ([64,64] net) in {time.time()-t0:.1f}s")

    ckpt = ROOT / "saved_models" / "headline_policy_v3_small.pt"
    torch.save(svc.net.state_dict(), ckpt)
    print(f"saved {ckpt}")

    plot_learning_curve(result, ROOT / "assets/plots/learning_curve_v3_small.png",
                         title="DDPG Learning Curve — 50k, [64,64] net, cosine LR")
    plot_critic_loss(result, ROOT / "assets/plots/critic_loss_v3_small.png",
                      title="Critic Loss — 50k, [64,64] net, cosine LR")
    loader = HouseExpoLoader(ROOT / "data/raw/sample_maps")
    verts = loader.load(loader.map_ids()[0]).verts
    plot_trajectory_overlay([(p.x, p.y) for p in env.robot.trajectory], verts,
                             ROOT / "assets/plots/trajectory_overlay_v3_small.png",
                             title="50k [64,64]-net policy trajectory")
    plot_coverage_heatmap(env.world.grid, ROOT / "assets/plots/coverage_heatmap_v3_small.png",
                           title="Final coverage — 50k [64,64]-net training")
    last = result.diagnostics[-1]
    print(f"final coverage={last.coverage:.3f}  episode_reward={last.episode_reward:.1f}")


if __name__ == "__main__":
    main()
