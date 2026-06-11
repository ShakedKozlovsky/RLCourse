"""Render trained policies (Walker2d + HalfCheetah) as animated GIFs.

Requires ``MUJOCO_GL=egl`` for headless rendering. The trained checkpoints
come from Layer 11's cross-env run.
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set BEFORE importing mujoco / gymnasium
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from proximal_lab.model.actor_critic_network import ActorCriticNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def render_one(env_id: str, ckpt: Path, out_gif: Path, n_steps: int = 400,
                fps: int = 30) -> None:
    """Roll out the trained policy and save the rendered frames as a GIF."""
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    net = ActorCriticNet.load(ckpt)
    net.eval()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    for step in range(n_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
            action, _, _ = net.act(obs_t, deterministic=True)
        next_obs, r, term, trunc, _ = env.step(action.squeeze(0).numpy())
        total_reward += float(r)
        frames.append(env.render())
        if term or trunc:
            obs, _ = env.reset(seed=step + 1)
        else:
            obs = next_obs
    env.close()
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"wrote {out_gif}  ({n_steps} frames, total_reward≈{total_reward:.1f})")


def main() -> None:
    out = ROOT / "assets" / "gifs"
    out.mkdir(parents=True, exist_ok=True)
    # Walker2d — the headline (positive reward, genuinely walking).
    walker_ckpt = ROOT / "saved_models" / "Walker2d-v5.pt"
    if walker_ckpt.exists():
        render_one("Walker2d-v5", walker_ckpt, out / "walker2d_trained.gif",
                    n_steps=300)
    # HalfCheetah — running ahead.
    cheetah_ckpt = ROOT / "saved_models" / "HalfCheetah-v5.pt"
    if cheetah_ckpt.exists():
        render_one("HalfCheetah-v5", cheetah_ckpt, out / "halfcheetah_trained.gif",
                    n_steps=300)


if __name__ == "__main__":
    main()
