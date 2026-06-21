"""Record a GIF of a cleaning episode — spec § 'find a visual way to present
the robot's cleaning behaviour'."""

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from roomba_lab.model.actor_critic_network import ActorCriticNet  # noqa: E402
from roomba_lab.sdk.sdk import RoombaLab  # noqa: E402


def _render_frame(env, polygon_verts) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 5))
    poly = np.array(polygon_verts)
    ax.fill(poly[:, 0], poly[:, 1], color="#eaeaea", alpha=0.5,
             edgecolor="black", linewidth=1.0)
    traj = [(p.x, p.y) for p in env.robot.trajectory]
    if traj:
        ax.plot([p[0] for p in traj], [p[1] for p in traj], color="#4477aa",
                 linewidth=1.2)
    p = env.robot.pose
    ax.scatter([p.x], [p.y], color="#cc6677", s=80, zorder=5)
    ax.set_aspect("equal")
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"step={env.step_count}  cov={env.world.coverage_fraction():.2f}")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return frame.copy()


def record_cleaning_gif(config_path: Path | None, checkpoint: Path, out: Path,
                         seed: int = 0, map_id: str | None = None,
                         max_steps: int = 200, frame_every: int = 4) -> Path:
    """Replay one cleaning episode of the loaded policy and write an imageio GIF.

    Captures one frame every `frame_every` steps (default ≈ 12 fps for a 200-
    step episode). Frame shows the apartment polygon, the trajectory so far,
    and the robot's current pose."""
    lab = RoombaLab(config_path=config_path)
    env = lab.make_env(map_id=map_id, max_episode_steps=max_steps)
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
    )
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    # Replay one episode and capture frames
    from roomba_lab.data.houseexpo_loader import HouseExpoLoader
    loader = HouseExpoLoader(lab.config.path("data_dir") / "raw" / "sample_maps")
    polygon_verts = loader.load(map_id or loader.map_ids()[0]).verts
    obs = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    for step in range(max_steps):
        with torch.no_grad():
            action = net.actor(torch.as_tensor(obs).unsqueeze(0)).cpu().numpy()[0]
        obs, _, done, _ = env.step(np.clip(action, -1, 1).astype(np.float32))
        if step % frame_every == 0:
            frames.append(_render_frame(env, polygon_verts))
        if done:
            frames.append(_render_frame(env, polygon_verts))
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=1.0 / 12.0)
    return out
