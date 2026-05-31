"""Side-by-side comparison of two training runs (REINFORCE vs A2C).

Produces a small JSON-serialisable summary plus the per-episode arrays
the GUI / README need to plot the comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fitness_rl.shared.types import EpisodeMetrics


@dataclass(frozen=True)
class ComparisonReport:
    """Per-algorithm scalar summary used in the README's comparison table."""

    name: str
    n_episodes: int
    mean_final_reward: float  # mean over the final 30 % of episodes
    overall_std: float
    final_std: float          # std over the final 30 %
    final_cv: float           # coefficient of variation = std / |mean|
    action_distribution: list[float]


@dataclass(frozen=True)
class ComparisonResult:
    """Bundles both reports + raw reward arrays for plotting."""

    reinforce: ComparisonReport
    a2c: ComparisonReport
    reinforce_rewards: list[float]
    a2c_rewards: list[float]
    winner: str  # "reinforce" | "a2c" | "tie"

    def to_dict(self) -> dict:
        return {
            "reinforce": asdict(self.reinforce),
            "a2c": asdict(self.a2c),
            "reinforce_rewards": list(self.reinforce_rewards),
            "a2c_rewards": list(self.a2c_rewards),
            "winner": self.winner,
        }


class ComparisonService:
    """Build a :class:`ComparisonResult` from two training histories."""

    def __init__(self, final_window_pct: float = 0.3):
        if not 0.0 < final_window_pct <= 1.0:
            raise ValueError("final_window_pct must be in (0, 1]")
        self._window_pct = float(final_window_pct)

    def compare(
        self,
        reinforce_history: list[EpisodeMetrics],
        a2c_history: list[EpisodeMetrics],
    ) -> ComparisonResult:
        if not reinforce_history or not a2c_history:
            raise ValueError("both training histories must be non-empty")
        r_report = self._make_report("reinforce", reinforce_history)
        a_report = self._make_report("a2c", a2c_history)
        winner = self._winner(r_report, a_report)
        return ComparisonResult(
            reinforce=r_report,
            a2c=a_report,
            reinforce_rewards=[m.total_reward for m in reinforce_history],
            a2c_rewards=[m.total_reward for m in a2c_history],
            winner=winner,
        )

    def _make_report(
        self, name: str, history: list[EpisodeMetrics]
    ) -> ComparisonReport:
        rewards = np.array([m.total_reward for m in history], dtype=np.float64)
        tail_n = max(1, int(self._window_pct * len(history)))
        tail = rewards[-tail_n:]
        final_mean = float(tail.mean())
        final_std = float(tail.std()) if tail.size > 1 else 0.0
        denom = max(abs(final_mean), 1e-9)
        # Sum action counts across all episodes for distribution.
        total_counts = np.zeros_like(history[0].action_counts, dtype=np.float64)
        for m in history:
            total_counts += m.action_counts
        total = float(total_counts.sum())
        dist = (total_counts / total).tolist() if total > 0 else [0.0] * len(total_counts)
        return ComparisonReport(
            name=name,
            n_episodes=len(history),
            mean_final_reward=final_mean,
            overall_std=float(rewards.std()) if rewards.size > 1 else 0.0,
            final_std=final_std,
            final_cv=final_std / denom,
            action_distribution=dist,
        )

    @staticmethod
    def _winner(r: ComparisonReport, a: ComparisonReport) -> str:
        """Higher mean wins; on a tie within 1%, lower CV wins."""
        if abs(r.mean_final_reward - a.mean_final_reward) / max(
            abs(r.mean_final_reward), abs(a.mean_final_reward), 1e-9
        ) < 0.01:
            if r.final_cv < a.final_cv:
                return "reinforce"
            if a.final_cv < r.final_cv:
                return "a2c"
            return "tie"
        return "reinforce" if r.mean_final_reward > a.mean_final_reward else "a2c"
