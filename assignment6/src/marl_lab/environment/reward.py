"""Per-agent reward functions — PRD_game.md § 6 + PRD_dec_pomdp.md § 1 R element.

Pure function on (board, move_info, sub_game_just_ended, winner). The
returned values are the **per-step** rewards used by the Q-learner; the
**per-sub-game** scoring (Table 1: 20/10/5/5) is computed separately by
``services/game_runner.py`` for the Gmail JSON."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Per-step shaping weights + per-sub-game scoring (mirror of yaml schema)."""
    # Per-step shaping (learning signal)
    step_penalty_cop: float = -0.05      # cop is incentivised to act fast
    step_penalty_thief: float = -0.01    # thief gets mild step cost
    capture_reward_cop: float = 1.0      # large positive on capture (cop)
    capture_penalty_thief: float = -1.0  # large negative on capture (thief)
    collision_penalty: float = -0.02     # both get a small penalty on collision
    barrier_placed_penalty_cop: float = -0.05  # opportunity cost of placing a barrier
    # Per-sub-game scoring (used for the Gmail JSON via game_runner)
    score_cop_win: int = 20
    score_thief_win: int = 10
    score_cop_loss: int = 5
    score_thief_loss: int = 5


def per_step_reward(
    capture: bool,
    timeout: bool,
    collision: bool,
    barrier_placed_by_cop: bool,
    cfg: RewardConfig,
) -> dict[str, float]:
    """Per-tick reward emitted to both agents. Pure function — no env mutation."""
    r_cop = cfg.step_penalty_cop
    r_thief = cfg.step_penalty_thief
    if capture:
        r_cop += cfg.capture_reward_cop
        r_thief += cfg.capture_penalty_thief
    if timeout and not capture:
        r_cop += -cfg.capture_reward_cop  # symmetric: cop's loss reward
        r_thief += -cfg.capture_penalty_thief
    if collision:
        r_cop += cfg.collision_penalty
        r_thief += cfg.collision_penalty
    if barrier_placed_by_cop:
        r_cop += cfg.barrier_placed_penalty_cop
    return {"cop": float(r_cop), "thief": float(r_thief)}


def sub_game_score(winner: str, cfg: RewardConfig) -> dict[str, int]:
    """Discrete score per sub-game per spec § 3.4 Table 1.

    winner ∈ {'cop', 'thief'}. Returns dict {cop: int, thief: int}."""
    if winner == "cop":
        return {"cop": cfg.score_cop_win, "thief": cfg.score_thief_loss}
    return {"cop": cfg.score_cop_loss, "thief": cfg.score_thief_win}
