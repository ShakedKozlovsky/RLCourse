"""Curriculum learning for pursuit-evasion MARL (Lin et al. 2025, Electronics MDPI).

The standard fixed-grid training has two failure modes on the 5×5 target:
  1. The cop must learn to corner the thief in a much larger search space
     than the agent's local Manhattan-radius view exposes.
  2. Exploration is brittle — random ε-policies on a 5×5 rarely produce
     captures, so the learning signal is sparse for thousands of episodes.

Curriculum learning addresses both by training first on a small grid (where
captures happen often, so the cop learns a positive Q signal fast) and
gradually expanding the grid as performance crosses a threshold. The thief
inherits the policy at each stage, so the cop's growing competence pressures
the thief to learn evasion strategies that generalise to the larger grid.

This module implements the **CurriculumSchedule**: a state machine over
(grid_size, threshold) stages. The trainer queries ``maybe_advance(history)``
after every K episodes; if the recent cop win-rate exceeds the stage's
threshold, the schedule advances to the next stage and the trainer rebuilds
its env at the new grid size. The Q-net weights are PRESERVED across stages —
that's the curriculum transfer signal."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumStage:
    """One stage in the curriculum: train on this grid until the threshold."""
    grid_size: tuple[int, int]
    cop_win_rate_threshold: float    # advance when rolling win-rate >= this
    min_episodes_at_stage: int = 20  # don't advance before this many episodes


@dataclass
class CurriculumSchedule:
    """Stateful curriculum: tracks current stage + advancement decisions.

    Default stages mirror spec § 5.1 Table 2 (2x2 → 3x3 → 4x4 → 5x5).
    Each threshold is a cop_win_rate target: easier grids have a *lower*
    threshold to advance quickly, harder grids have a *higher* threshold
    so the policy is well-trained before going up. The final stage has
    threshold ≥ 1.0 so it never advances (i.e. terminal)."""

    stages: list[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(grid_size=(2, 2), cop_win_rate_threshold=0.60),
        CurriculumStage(grid_size=(3, 3), cop_win_rate_threshold=0.50),
        CurriculumStage(grid_size=(4, 4), cop_win_rate_threshold=0.45),
        CurriculumStage(grid_size=(5, 5), cop_win_rate_threshold=1.1),
    ])
    rolling_window: int = 20         # how many recent episodes feed the rate
    stage_idx: int = 0
    episodes_at_stage: int = 0
    advancements: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("at least one stage required")

    def current_grid_size(self) -> tuple[int, int]:
        return self.stages[self.stage_idx].grid_size

    def current_threshold(self) -> float:
        return self.stages[self.stage_idx].cop_win_rate_threshold

    def maybe_advance(self, episode_winners: list[str]) -> bool:
        """Inspect recent winners; advance to next stage if criterion met.

        Returns True if the schedule advanced (so the trainer should rebuild
        its env at the new grid size). Returns False otherwise."""
        self.episodes_at_stage += 1
        stage = self.stages[self.stage_idx]
        if self.episodes_at_stage < stage.min_episodes_at_stage:
            return False
        if self.stage_idx >= len(self.stages) - 1:
            return False                 # already at terminal stage
        recent = episode_winners[-self.rolling_window:]
        if len(recent) < self.rolling_window:
            return False
        cop_win_rate = sum(1 for w in recent if w == "cop") / len(recent)
        if cop_win_rate >= stage.cop_win_rate_threshold:
            self.stage_idx += 1
            self.episodes_at_stage = 0
            self.advancements += 1
            return True
        return False

    def is_terminal(self) -> bool:
        return self.stage_idx >= len(self.stages) - 1
