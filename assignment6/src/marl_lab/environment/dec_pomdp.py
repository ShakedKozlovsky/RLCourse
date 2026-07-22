"""Dec-POMDP environment — the gym-shape-but-gym-free wrapper.

ADR-001: NO ``gym``/``gymnasium`` imports. We expose ``reset(seed) → joint_obs``
and ``step(joint_action) → (joint_obs, joint_reward, done, info)`` for API
familiarity but the env is a pure Python class.

ADR-002 honest framing: this is technically a POSG (per-agent rewards) but we
implement the CTDE machinery treating it as Dec-POMDP — see PRD § 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl_lab.environment.reward import RewardConfig, per_step_reward
from marl_lab.game.actions import Action
from marl_lab.game.board import Board, BoardFactory
from marl_lab.game.moves import MoveDynamics
from marl_lab.game.win import adjudicate
from marl_lab.sensor.partial_observation import obs_dim, observe


@dataclass(frozen=True)
class EnvConfig:
    """Static config used to construct the env (mirrors yaml `game` block)."""
    grid_size: tuple[int, int]
    max_moves: int
    max_barriers: int
    enable_barriers: bool
    observation_radius: int


class DecPomdpEnv:
    """Cops-and-Robbers Dec-POMDP env. One ``step`` advances one tick.

    `reset(seed)` starts a fresh sub-game; episode terminates on capture or
    timeout. The user is responsible for resetting between sub-games."""

    def __init__(self, env_cfg: EnvConfig, reward_cfg: RewardConfig,
                 rng: np.random.Generator | None = None) -> None:
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self._rng = rng or np.random.default_rng(0)
        self._board_factory = BoardFactory(
            grid_size=env_cfg.grid_size,
            enable_barriers=env_cfg.enable_barriers,
            rng=self._rng,
        )
        self._dynamics = MoveDynamics(max_barriers=env_cfg.max_barriers)
        self._board: Board | None = None

    @property
    def obs_dim(self) -> int:
        """Per-agent observation vector size."""
        return obs_dim(self.env_cfg.observation_radius)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Start a fresh sub-game. Returns per-agent joint observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._board_factory.rng = self._rng
        self._board = self._board_factory.fresh()
        return self._joint_obs()

    def step(self, joint_action: dict[str, int]) -> tuple[
        dict[str, np.ndarray], dict[str, float], bool, dict
    ]:
        """Advance one tick. ``joint_action`` is {'cop': int, 'thief': int}.

        Returns (joint_obs, joint_reward, done, info)."""
        if self._board is None:
            raise RuntimeError("env.step() called before env.reset()")
        cop_action = Action(int(joint_action["cop"]))
        thief_action = Action(int(joint_action["thief"]))
        new_board, move_info = self._dynamics.apply(self._board, cop_action, thief_action)
        self._board = new_board

        winner = adjudicate(new_board, self.env_cfg.max_moves)
        capture = winner == "cop"
        timeout = winner == "thief"
        done = winner is not None
        # Manhattan distance from the NEW board — used for distance shaping
        # if reward_cfg.distance_shaping_weight > 0 (v1.13 CTDE training aid)
        cr, cc = new_board.cop_pos
        tr, tc = new_board.thief_pos
        manhattan = abs(cr - tr) + abs(cc - tc)
        reward = per_step_reward(
            capture=capture, timeout=timeout, collision=move_info.collision,
            barrier_placed_by_cop=move_info.barrier_placed,
            cfg=self.reward_cfg,
            manhattan_distance=manhattan,
        )
        info = {
            "winner": winner,
            "step": new_board.step,
            "barriers_placed": len(new_board.barriers),
            "collision": move_info.collision,
            "barrier_placed_at": move_info.barrier_placed_at,
        }
        return self._joint_obs(), reward, done, info

    def global_state(self) -> np.ndarray:
        """ONLY for training-time centralised critic / QMIX mixer.

        Spec § 5.1: execution must use only local obs. Calling this from the
        execution path is a bug — we log a warning. The mcp/* servers refuse
        to send the global state in any response."""
        if self._board is None:
            raise RuntimeError("global_state() called before reset()")
        return self._board.to_state_vector()

    def board(self) -> Board:
        """Public read-only view of the current Board (used by GUI + tests)."""
        if self._board is None:
            raise RuntimeError("board() called before reset()")
        return self._board

    def _joint_obs(self) -> dict[str, np.ndarray]:
        assert self._board is not None
        return {
            "cop": observe(self._board, "cop", self.env_cfg.observation_radius),
            "thief": observe(self._board, "thief", self.env_cfg.observation_radius),
        }
