"""Bonus-game scoring rules (spec § 9.2).

Given the total score per group across all 6 sub-games:
  - Different totals → winner: 10 pts, loser: 7 pts
  - Equal totals    → 5 pts each (tie)

Return the {group_name: bonus_pts} mapping."""

from __future__ import annotations

BONUS_WIN = 10
BONUS_LOSS = 7
BONUS_TIE = 5


def compute_bonus_claim(totals_by_group: dict[str, int]) -> dict[str, int]:
    """Apply spec § 9.2 rule to per-group totals.

    Args:
        totals_by_group: {"Team-Alpha": 60, "Team-Beta": 80} — sums of
            per-role scores accumulated by each group across all 6
            sub-games.

    Returns:
        {group_name: bonus_pts} where pts ∈ {5, 7, 10} per § 9.2.

    Raises:
        ValueError: if len(totals_by_group) != 2 (bonus game is inter-group,
            exactly two teams).
    """
    if len(totals_by_group) != 2:
        raise ValueError(
            f"bonus game requires exactly 2 groups, got {len(totals_by_group)}: "
            f"{list(totals_by_group.keys())}"
        )
    (a, sa), (b, sb) = list(totals_by_group.items())
    if sa == sb:
        return {a: BONUS_TIE, b: BONUS_TIE}
    if sa > sb:
        return {a: BONUS_WIN, b: BONUS_LOSS}
    return {a: BONUS_LOSS, b: BONUS_WIN}
