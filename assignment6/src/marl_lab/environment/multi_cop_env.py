"""Multi-cop variant of the Dec-POMDP env — swarm vs single, Reflection Q3.

Spec § 7 reflection Q3 asks: how does the swarm-vs-single-agent framing
change the optimal-policy story? To answer this empirically we need an env
where N ≥ 2 cops chase 1 thief. This module implements that variant while
keeping the base ``DecPomdpEnv`` unchanged (existing spec-conforming path
uses 1 cop + 1 thief exactly).

Simplifications from the full spec:
  - No barriers (multi-cop dynamics with N cops placing barriers explodes
    the state space; the point of Q3 is coordination, not barrier tactics)
  - Terminal condition: ANY cop landing on the thief = capture (team win)
  - Reward: shared team reward for capture (cop swarm is genuinely
    cooperative — no POSG framing tension)"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marl_lab.game.actions import Action
from marl_lab.game.board import Board
from marl_lab.sensor.partial_observation import obs_dim, observe


@dataclass(frozen=True)
class MultiCopEnvConfig:
    """Multi-cop env parameters."""
    n_cops: int = 2
    grid_size: tuple[int, int] = (5, 5)
    max_moves: int = 25
    observation_radius: int = 2


@dataclass
class MultiCopBoard:
    """Board with N cop positions + 1 thief. No barriers."""
    grid_size: tuple[int, int]
    cop_positions: list[tuple[int, int]] = field(default_factory=list)
    thief_pos: tuple[int, int] = (0, 0)
    step: int = 0
    capture_flag: bool = False

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        h, w = self.grid_size
        return 0 <= r < h and 0 <= c < w


DELTA = {
    Action.UP: (-1, 0), Action.DOWN: (1, 0),
    Action.LEFT: (0, -1), Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


class MultiCopEnv:
    """Multi-cop pursuit env.

    Interface mirrors DecPomdpEnv: ``reset()`` returns joint obs, ``step()``
    takes a dict of per-cop-and-thief actions.

    ``reset()`` sample: N cops uniformly on random distinct cells; thief on
    a remaining cell. All ``__pos`` are tuples.

    ``step(joint_action)`` where joint_action is
    ``{"cop_0": a, "cop_1": a, ..., "thief": a}``. Cop actions are picked
    from Actions.UP/DOWN/LEFT/RIGHT/STAY (5 legal; no PLACE_BARRIER).
    Thief the same. Simultaneous resolution: each agent moves to their
    target unless it goes off-grid, in which case they stay put.

    Terminal condition: any cop shares a cell with the thief after the
    move → cop team wins. If step == max_moves without capture → thief wins.

    Reward: shared team reward — +1 for cops on capture, 0 otherwise.
    Thief gets the negation. Small step penalty for both to encourage
    action."""

    def __init__(self, cfg: MultiCopEnvConfig,
                 rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self._rng = rng or np.random.default_rng(0)
        self._board: MultiCopBoard | None = None

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.cfg.observation_radius)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        h, w = self.cfg.grid_size
        cells = [(r, c) for r in range(h) for c in range(w)]
        idx = self._rng.choice(len(cells), size=self.cfg.n_cops + 1, replace=False)
        positions = [cells[i] for i in idx]
        cop_positions = positions[:self.cfg.n_cops]
        thief_pos = positions[self.cfg.n_cops]
        self._board = MultiCopBoard(
            grid_size=self.cfg.grid_size,
            cop_positions=cop_positions, thief_pos=thief_pos,
        )
        return self._joint_obs()

    def step(self, joint_action: dict[str, int]) -> tuple[
        dict[str, np.ndarray], dict[str, float], bool, dict
    ]:
        if self._board is None:
            raise RuntimeError("step() called before reset()")
        b = self._board
        # Resolve new positions (target if in-bounds else stay)
        new_cops: list[tuple[int, int]] = []
        for i, cp in enumerate(b.cop_positions):
            a = Action(int(joint_action[f"cop_{i}"]))
            dr, dc = DELTA[a]
            target = (cp[0] + dr, cp[1] + dc)
            new_cops.append(target if b.in_bounds(target) else cp)
        thief_a = Action(int(joint_action["thief"]))
        dr, dc = DELTA[thief_a]
        thief_target = (b.thief_pos[0] + dr, b.thief_pos[1] + dc)
        new_thief = thief_target if b.in_bounds(thief_target) else b.thief_pos

        capture = any(cp == new_thief for cp in new_cops)
        new_step = b.step + 1
        timeout = (new_step >= self.cfg.max_moves) and not capture
        done = capture or timeout
        self._board = MultiCopBoard(
            grid_size=b.grid_size, cop_positions=new_cops,
            thief_pos=new_thief, step=new_step, capture_flag=capture,
        )
        # Shared team reward for cops; negation for thief
        r_cops = 1.0 if capture else (-0.05)
        r_thief = -1.0 if capture else 0.05
        # Timeout: thief wins
        if timeout:
            r_thief += 0.5
        reward: dict[str, float] = {f"cop_{i}": r_cops
                                     for i in range(self.cfg.n_cops)}
        reward["thief"] = r_thief
        winner = "cops" if capture else ("thief" if timeout else None)
        info = {"winner": winner, "step": new_step,
                "capture": capture, "timeout": timeout}
        return self._joint_obs(), reward, done, info

    def board(self) -> MultiCopBoard:
        if self._board is None:
            raise RuntimeError("board() called before reset()")
        return self._board

    def _joint_obs(self) -> dict[str, np.ndarray]:
        """Each cop sees its own Manhattan-radius view; thief sees hers.

        Each cop's view marks the nearest visible other cop as "OPPONENT"
        channel bit; this is a simplification — proper multi-cop obs
        would carry ALL other cops' positions. Kept simple because Q3 is
        about the coordination dynamic, not observation richness."""
        assert self._board is not None
        h, w = self._board.grid_size
        obs_dict: dict[str, np.ndarray] = {}
        # Each cop sees the board through the same encoding as the base env,
        # with the thief as OPPONENT. Other cops are ignored in the obs
        # (they'd take another channel; we simplify).
        for i, cp in enumerate(self._board.cop_positions):
            fake_board = Board(grid_size=(h, w), cop_pos=cp,
                                 thief_pos=self._board.thief_pos)
            obs_dict[f"cop_{i}"] = observe(fake_board, "cop",
                                             radius=self.cfg.observation_radius)
        # Thief sees the NEAREST cop as opponent (again — simplification).
        nearest_cop = min(self._board.cop_positions,
                            key=lambda cp: abs(cp[0] - self._board.thief_pos[0])
                                            + abs(cp[1] - self._board.thief_pos[1]))
        fake_board = Board(grid_size=(h, w), cop_pos=nearest_cop,
                             thief_pos=self._board.thief_pos)
        obs_dict["thief"] = observe(fake_board, "thief",
                                       radius=self.cfg.observation_radius)
        return obs_dict
