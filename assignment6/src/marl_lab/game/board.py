"""Board — the global state S of the Dec-POMDP (PRD_dec_pomdp.md § 1).

Pure dataclass. Updated by `moves.py::MoveDynamics.apply`. Read by
`sensor/partial_observation.py` (to compute per-agent local views) and by
`environment/reward.py`."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Board:
    """Frozen global state. ``barriers`` is a frozenset so the whole Board is hashable."""
    grid_size: tuple[int, int]
    cop_pos: tuple[int, int]
    thief_pos: tuple[int, int]
    barriers: frozenset[tuple[int, int]] = frozenset()
    step: int = 0
    capture_flag: bool = False
    timeout_flag: bool = False

    def with_(self, **kw) -> Board:  # noqa: ANN003 — generic patch
        """Return a new Board with selected fields overridden (frozen-replace)."""
        return replace(self, **kw)

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        """True if ``pos`` lies on the grid."""
        r, c = pos
        h, w = self.grid_size
        return 0 <= r < h and 0 <= c < w

    def is_blocked(self, pos: tuple[int, int]) -> bool:
        """True if ``pos`` is off-grid OR contains a barrier (≠ agent positions)."""
        return (not self.in_bounds(pos)) or (pos in self.barriers)

    def cell_kind(self, pos: tuple[int, int]) -> Literal["empty", "barrier", "cop", "thief", "offgrid"]:
        """Symbolic content of a cell — used by the sensor encoding."""
        if not self.in_bounds(pos):
            return "offgrid"
        if pos == self.cop_pos:
            return "cop"
        if pos == self.thief_pos:
            return "thief"
        if pos in self.barriers:
            return "barrier"
        return "empty"

    def to_state_vector(self) -> np.ndarray:
        """Flat representation used by the centralised critic + mixer hypernet.

        Encoding: cop one-hot over HW + thief one-hot over HW + barriers
        bitmap over HW + scalar step + scalar barrier_remaining. dim = 3HW+2."""
        h, w = self.grid_size
        cop_oh = np.zeros(h * w, dtype=np.float32)
        thief_oh = np.zeros(h * w, dtype=np.float32)
        barr_bm = np.zeros(h * w, dtype=np.float32)
        cop_oh[self.cop_pos[0] * w + self.cop_pos[1]] = 1.0
        thief_oh[self.thief_pos[0] * w + self.thief_pos[1]] = 1.0
        for r, c in self.barriers:
            barr_bm[r * w + c] = 1.0
        scalars = np.array([self.step, len(self.barriers)], dtype=np.float32)
        return np.concatenate([cop_oh, thief_oh, barr_bm, scalars])

    @staticmethod
    def state_vector_dim(grid_size: tuple[int, int]) -> int:
        """Static helper — used by ConfigManager + Mixer hypernet sizing."""
        h, w = grid_size
        return 3 * h * w + 2


@dataclass
class BoardFactory:
    """Construct fresh Boards with random valid agent positions."""
    grid_size: tuple[int, int]
    enable_barriers: bool
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def fresh(self) -> Board:
        """Sample a new starting board: cop and thief on different cells, no barriers."""
        h, w = self.grid_size
        all_cells = [(r, c) for r in range(h) for c in range(w)]
        idx = self.rng.permutation(len(all_cells))
        cop = all_cells[int(idx[0])]
        thief = all_cells[int(idx[1])]
        return Board(grid_size=self.grid_size, cop_pos=cop, thief_pos=thief)
