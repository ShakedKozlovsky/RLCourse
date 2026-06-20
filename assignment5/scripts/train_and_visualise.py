"""End-to-end: train the headline policy, save it, and emit every visualisation."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.sdk.trainers import build_ddpg_service
from roomba_lab.tools.viz.plots import (
    plot_coverage_heatmap,
    plot_critic_loss,
    plot_learning_curve,
    plot_trajectory_overlay,
)

ROOT = Path(__file__).resolve().parents[1]


def main(total_timesteps: int = 4000, seed: int = 0) -> None:
    started = time.time()
    lab = RoombaLab(config_path=ROOT / "configs" / "setup.json")
    env = lab.make_env()
    svc = build_ddpg_service(lab.config, env)
    result = svc.fit(total_timesteps=total_timesteps, seed=seed)
    print(f"trained {total_timesteps} steps in {time.time() - started:.1f}s")

    ckpt = ROOT / "saved_models" / "headline_policy.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(svc.net.state_dict(), ckpt)
    print(f"saved {ckpt}")

    plots_dir = ROOT / "assets" / "plots"
    plot_learning_curve(result, plots_dir / "learning_curve.png")
    plot_critic_loss(result, plots_dir / "critic_loss.png")

    # Trajectory overlay + coverage heatmap from a deterministic rollout
    map_id = lab.config.get("env.primary_map_id")
    loader = HouseExpoLoader(ROOT / "data" / "raw" / "sample_maps")
    if map_id not in loader.map_ids():
        map_id = loader.map_ids()[0]
    polygon_verts = loader.load(map_id).verts
    world = env.world
    plot_trajectory_overlay([(p.x, p.y) for p in env.robot.trajectory],
                              polygon_verts, plots_dir / "trajectory_overlay.png")
    plot_coverage_heatmap(world.grid, plots_dir / "coverage_heatmap.png")
    print(f"wrote 4 plots to {plots_dir}")


if __name__ == "__main__":
    main()
