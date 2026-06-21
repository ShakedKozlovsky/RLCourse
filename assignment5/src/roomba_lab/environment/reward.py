"""Reward function — pure (state, prev_state, cfg) → (reward, info).

PRD § 4: new_cell_bonus per freshly cleaned cell, collision_penalty when the
candidate move would clip a wall (move cancelled), step_penalty every step,
completion_bonus once when coverage crosses the target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    new_cell_bonus: float
    collision_penalty: float
    step_penalty: float
    completion_bonus: float
    coverage_target: float
    coverage_progress_coef: float = 0.0


@dataclass(frozen=True)
class RewardInputs:
    new_cells: int
    collided: bool
    coverage_before: float
    coverage_after: float


def compute_reward(inputs: RewardInputs, cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """Pure reward function: per-step penalty + per-cell bonus + (dense) coverage-
    progress shaping + collision penalty + (one-shot) completion bonus."""
    r = cfg.step_penalty
    r += inputs.new_cells * cfg.new_cell_bonus
    r += cfg.coverage_progress_coef * max(0.0, inputs.coverage_after - inputs.coverage_before)
    if inputs.collided:
        r += cfg.collision_penalty
    bonus_fired = (
        inputs.coverage_before < cfg.coverage_target <= inputs.coverage_after
    )
    if bonus_fired:
        r += cfg.completion_bonus
    return float(r), {
        "new_cells": float(inputs.new_cells),
        "collided": float(inputs.collided),
        "bonus_fired": float(bonus_fired),
    }
