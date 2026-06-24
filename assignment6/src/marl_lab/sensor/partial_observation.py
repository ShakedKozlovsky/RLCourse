"""Manhattan-radius partial observation — the Dec-POMDP ``O`` element.

PRD_partial_observation.md is the contract."""

from __future__ import annotations

import numpy as np

from marl_lab.game.board import Board

# Cell-encoding channels (PRD § 3 — one-hot per cell)
EMPTY, WALL, BARRIER, OPPONENT = 0, 1, 2, 3


def cells_in_radius(centre: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    """All grid coordinates within Manhattan distance ``radius`` of ``centre``.

    Returns the cells in raster order (row-major) for stable encoding."""
    cells = []
    for r in range(centre[0] - radius, centre[0] + radius + 1):
        for c in range(centre[1] - radius, centre[1] + radius + 1):
            if abs(r - centre[0]) + abs(c - centre[1]) <= radius:
                cells.append((r, c))
    return cells


def observe(board: Board, agent_role: str, radius: int) -> np.ndarray:
    """Per-agent local observation (Manhattan-radius mask + status entries).

    Layout (PRD § 2): [4 × n_visible_cells one-hot] + [x_norm, y_norm, step_norm,
    barriers_remaining_norm, role_one_hot_cop, role_one_hot_thief]."""
    self_pos = board.cop_pos if agent_role == "cop" else board.thief_pos
    opponent_pos = board.thief_pos if agent_role == "cop" else board.cop_pos

    visible = cells_in_radius(self_pos, radius)
    n_cells = len(visible)
    grid = np.zeros((n_cells, 4), dtype=np.float32)
    for k, (r, c) in enumerate(visible):
        if not board.in_bounds((r, c)):
            grid[k, WALL] = 1.0
            continue
        if (r, c) == opponent_pos:
            grid[k, OPPONENT] = 1.0
        elif (r, c) in board.barriers:
            grid[k, BARRIER] = 1.0
        else:
            grid[k, EMPTY] = 1.0

    h, w = board.grid_size
    x_norm = self_pos[0] / max(1, h - 1)
    y_norm = self_pos[1] / max(1, w - 1)
    # Max moves comes externally; here we encode step itself; downstream code
    # divides by max_moves if normalisation is needed.
    step_scaled = board.step / 25.0
    barr_remaining = max(0.0, 1.0 - len(board.barriers) / 5.0) if agent_role == "cop" else 0.0
    role_oh_cop = 1.0 if agent_role == "cop" else 0.0
    role_oh_thief = 1.0 - role_oh_cop

    status = np.array([x_norm, y_norm, step_scaled, barr_remaining,
                       role_oh_cop, role_oh_thief], dtype=np.float32)
    return np.concatenate([grid.flatten(), status])


def obs_dim(radius: int) -> int:
    """Return the observation vector length for a given Manhattan radius.

    Same formula used by ConfigManager + Q-net constructor."""
    n_visible = 2 * radius * (radius + 1) + 1   # Manhattan diamond size
    return 4 * n_visible + 6                     # 4 channels per cell + 6 status entries
