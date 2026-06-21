"""Layer 28 — push coverage above 0.20 (TA M1 final closure).

Strategy:
  - new_cell_bonus 1.0 → 3.0 (3× the carrot per fresh cell)
  - step_penalty -0.05 → -0.02 (less harsh on movement)
  - decay_steps 50000 → 30000 (σ tapers exactly when training ends)
  - actor_lr 1e-4 → 5e-5 mid-training (manual halve at step 20k for stability)
  - 30 000 timesteps — long enough to learn, short enough to avoid the 50k-policy
    catastrophic forgetting we saw earlier"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path

import torch

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.sdk.experiments import _write_temp_cfg  # noqa: F401  (re-uses pattern)
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.sdk.trainers import build_ddpg_service
from roomba_lab.tools.viz.plots import (
    plot_coverage_heatmap,
    plot_critic_loss,
    plot_learning_curve,
    plot_trajectory_overlay,
)

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 30000
LR_DECAY_AT = 20000
LR_DECAY_FACTOR = 0.5


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "setup.json").read_text())
    tuned = copy.deepcopy(base_cfg)
    tuned["reward"]["new_cell_bonus"] = 3.0
    tuned["reward"]["step_penalty"] = -0.02
    tuned["noise"]["decay_steps"] = 30000
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cfg_path = Path(f.name)
    cfg_path.write_text(json.dumps(tuned, indent=2))

    t0 = time.time()
    lab = RoombaLab(config_path=cfg_path)
    env = lab.make_env()
    svc = build_ddpg_service(lab.config, env)

    # Custom training with mid-run LR decay

    from roomba_lab.shared.types import StepDiagnostic, TrainResult, Transition
    result = TrainResult()
    obs = env.reset(seed=0)
    episode_reward = 0.0
    for step in range(TOTAL_TS):
        if step == LR_DECAY_AT:
            for g in svc.actor_opt.param_groups:
                g["lr"] *= LR_DECAY_FACTOR
            for g in svc.critic_opt.param_groups:
                g["lr"] *= LR_DECAY_FACTOR
            print(f"step {step}: LR halved")
        svc.noise.set_sigma(svc.schedule.at(step))
        action = svc._select_action(obs, step)  # noqa: SLF001 — internal helper
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        svc.buffer.push(Transition(state=obs, action=action,
                                     reward=float(reward), next_state=next_obs,
                                     done=bool(done)))
        obs = next_obs
        actor_l = critic_l = mean_q = 0.0
        if len(svc.buffer) >= max(svc.hp.batch_size, svc.hp.warmup_steps):
            batch = svc.buffer.sample(svc.hp.batch_size)
            from roomba_lab.services.ddpg_update import apply_update
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
    print(f"trained {TOTAL_TS} steps in {time.time()-t0:.1f}s")

    ckpt = ROOT / "saved_models" / "headline_policy_v2.pt"
    torch.save(svc.net.state_dict(), ckpt)
    print(f"saved {ckpt}")

    plot_learning_curve(result, ROOT / "assets/plots/learning_curve_v2.png",
                         title="DDPG Learning Curve — 30k, boosted reward + LR decay")
    plot_critic_loss(result, ROOT / "assets/plots/critic_loss_v2.png",
                      title="Critic Loss — 30k, boosted reward + LR decay")
    loader = HouseExpoLoader(ROOT / "data/raw/sample_maps")
    verts = loader.load(loader.map_ids()[0]).verts
    plot_trajectory_overlay([(p.x, p.y) for p in env.robot.trajectory], verts,
                             ROOT / "assets/plots/trajectory_overlay_v2.png",
                             title="30k boosted-reward policy trajectory")
    plot_coverage_heatmap(env.world.grid, ROOT / "assets/plots/coverage_heatmap_v2.png",
                           title="Final coverage — 30k boosted-reward training")
    last = result.diagnostics[-1]
    print(f"final coverage={last.coverage:.3f}  episode_reward={last.episode_reward:.1f}")


if __name__ == "__main__":
    main()
