"""Live Tkinter widget for real-time board visualisation (spec § 5.4).

Wraps the headless ``GameGuiCore`` (which is the tested part) with a
Tkinter Canvas that redraws on every step. Runs the game with a
configurable delay so a human observer can follow the cop chasing the thief.

Skipped on headless environments (no DISPLAY): the widget won't construct
because tkinter itself won't import — the module still imports cleanly."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.interface.board_renderer import BARRIER, COP, EMPTY, THIEF, render
from marl_lab.interface.game_gui import GameGuiCore, PolicyFn

CELL_PX = 60
GRID_COLOUR = "black"
CELL_COLOURS = {
    EMPTY: "white",
    COP: "#3060ff",
    THIEF: "#ff3050",
    BARRIER: "#404040",
}


def _load_greedy_policy(checkpoint_path: str, role: str,
                          observation_radius: int) -> PolicyFn:
    """Load a trained Q-net from a checkpoint and return a greedy policy."""
    import torch

    from marl_lab.model.recurrent_q import QPerAgent
    from marl_lab.sensor.partial_observation import obs_dim
    o = obs_dim(observation_radius)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128),
                        gru_hidden_size=64)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    q_net.load_state_dict(ckpt["q_nets"][role])
    q_net.eval()
    hidden_state = {"h": q_net.init_hidden(batch_size=1)}

    def policy(r: str, obs: np.ndarray) -> int:
        n_legal = 6 if r == "cop" else 5
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_seq, hidden_state["h"] = q_net(obs_t, hidden=hidden_state["h"])
            q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
        q_masked = q.copy()
        q_masked[n_legal:] = -np.inf
        return int(np.argmax(q_masked))
    return policy


class LiveTkGui:
    """Real-time Tkinter window driving a sub-game via ``GameGuiCore``.

    Constructor imports tkinter lazily so ``import interface.tk_gui`` does
    not crash on headless environments (they only crash when actually
    trying to instantiate the widget)."""

    def __init__(self, env: DecPomdpEnv, cop_policy: PolicyFn,
                 thief_policy: PolicyFn, *, delay_ms: int = 500,
                 cell_px: int = CELL_PX, title: str = "marl_lab live") -> None:
        import tkinter as tk  # noqa: PLC0415 — intentional lazy import
        self.core = GameGuiCore(env=env, cop_policy=cop_policy,
                                    thief_policy=thief_policy)
        h, w = env.env_cfg.grid_size
        self._delay_ms = int(delay_ms)
        self._cell_px = int(cell_px)
        self.root = tk.Tk()
        self.root.title(title)
        self.status = tk.StringVar(value="click Start")
        tk.Label(self.root, textvariable=self.status,
                   font=("Menlo", 12)).pack(pady=4)
        self.canvas = tk.Canvas(self.root, width=w * cell_px,
                                  height=h * cell_px,
                                  bg="white", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Start",
                   command=self._start).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Reset",
                   command=self._reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Quit",
                   command=self.root.destroy).pack(side=tk.LEFT, padx=4)
        self._reset()

    def _reset(self) -> None:
        self.core.reset(seed=0)
        self._redraw()
        self.status.set("ready — click Start")

    def _start(self) -> None:
        self.status.set("running…")
        self.root.after(self._delay_ms, self._tick)

    def _tick(self) -> None:
        snap = self.core.step()
        self._redraw(snap["board_grid"])
        if snap["done"]:
            self.status.set(f"done — winner: {snap['winner']} "
                              f"after {snap['steps']} moves")
            return
        self.status.set(f"step {snap['steps']}")
        self.root.after(self._delay_ms, self._tick)

    def _redraw(self, grid_list: list | None = None) -> None:
        board = self.core.env.board()
        grid = render(board) if grid_list is None else np.array(grid_list)
        self.canvas.delete("all")
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                x0 = c * self._cell_px
                y0 = r * self._cell_px
                x1 = x0 + self._cell_px
                y1 = y0 + self._cell_px
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=CELL_COLOURS[int(grid[r, c])],
                    outline=GRID_COLOUR, width=1,
                )
                if grid[r, c] == COP:
                    self.canvas.create_text(
                        (x0 + x1) // 2, (y0 + y1) // 2,
                        text="C", fill="white",
                        font=("Menlo", int(self._cell_px * 0.4), "bold"),
                    )
                elif grid[r, c] == THIEF:
                    self.canvas.create_text(
                        (x0 + x1) // 2, (y0 + y1) // 2,
                        text="T", fill="white",
                        font=("Menlo", int(self._cell_px * 0.4), "bold"),
                    )

    def run(self) -> None:
        """Enter the Tk event loop. Blocks until the window is closed."""
        self.root.mainloop()


def launch_live_gui(checkpoint_path: str, *,
                    grid_size: tuple[int, int] = (5, 5),
                    max_moves: int = 25,
                    observation_radius: int = 2,
                    delay_ms: int = 500) -> None:
    """Convenience one-liner: load a checkpoint, open a window, start playing.

    Called by the ``marl gui`` CLI subcommand."""
    if not os.environ.get("DISPLAY") and os.name != "nt" and os.uname().sysname != "Darwin":
        raise SystemExit(
            "no DISPLAY available — the live Tk widget requires a real "
            "display server. On Linux headless: use `xvfb-run marl gui ...` "
            "or run in a desktop session. Alternatively, view "
            "assets/figures/sub_game.gif for a pre-rendered game."
        )
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=grid_size, max_moves=max_moves,
                          max_barriers=5, enable_barriers=True,
                          observation_radius=observation_radius),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )
    cop_policy = _load_greedy_policy(checkpoint_path, "cop", observation_radius)
    thief_policy = _load_greedy_policy(checkpoint_path, "thief", observation_radius)
    gui = LiveTkGui(env=env, cop_policy=cop_policy,
                      thief_policy=thief_policy, delay_ms=delay_ms)
    gui.run()
