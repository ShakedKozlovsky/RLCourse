"""σ=0 vs σ=0.2 side-by-side coverage heatmap — LONGER training (Mod4 fix).

The v1.10 sigma_comparison.png was from 4 000-step training; TA Mod4 flagged
that as too short for either agent to have converged. This re-runs both
agents at 15 000 steps for a fairer asymptotic comparison."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomba_lab.memory.replay_buffer import ReplayBuffer  # noqa: E402
from roomba_lab.model.actor_critic_network import ActorCriticNet  # noqa: E402
from roomba_lab.noise.gaussian import GaussianNoise  # noqa: E402
from roomba_lab.noise.schedule import LinearSigmaSchedule  # noqa: E402
from roomba_lab.sdk.sdk import RoombaLab  # noqa: E402
from roomba_lab.services.ddpg_service import DDPGHyperparams, DDPGService  # noqa: E402
from roomba_lab.shared.seed import set_global_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TS = 15000


def _train_one(sigma: float, seed: int = 0) -> np.ndarray:
    set_global_seed(seed)
    lab = RoombaLab()
    env = lab.make_env()
    rng = np.random.default_rng(seed)
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
        actor_head_gain=float(lab.config.get("ddpg.actor_head_gain", 0.1)),
    )
    buf = ReplayBuffer(int(lab.config.get("ddpg.replay_capacity")),
                       env.obs_dim, env.action_dim, rng=rng)
    noise = GaussianNoise(env.action_dim, sigma=sigma, rng=rng)
    schedule = LinearSigmaSchedule(initial=sigma, final=sigma, decay_steps=1)
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
    svc.fit(total_timesteps=TOTAL_TS, seed=seed)
    return env.world.grid.copy()


def main() -> None:
    t0 = time.time()
    grid_zero = _train_one(sigma=0.0)
    grid_default = _train_one(sigma=0.2)
    print(f"trained both ({TOTAL_TS} steps each) in {time.time() - t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, grid, label in zip(
        axes, [grid_zero, grid_default],
        [f"σ = 0 (no exploration) — {TOTAL_TS} steps",
         f"σ = 0.2 (default) — {TOTAL_TS} steps"], strict=True,
    ):
        display = grid.astype(np.float32)
        display[grid == 255] = np.nan
        im = ax.imshow(display, origin="lower", cmap="viridis",
                        vmin=0, vmax=1, interpolation="nearest")
        cov = float((grid == 1).sum() / max(1, ((grid == 0) | (grid == 1)).sum()))
        ax.set_title(f"{label}\n(coverage = {cov:.3f})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reflection-Q2 long-training evidence — "
                  f"σ=0 vs σ=0.2 at {TOTAL_TS} steps (Layer 27)", fontsize=13)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(),
                  label="visited (1) / unvisited (0)", shrink=0.7)
    out = ROOT / "assets" / "plots" / "sigma_comparison_15k.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
