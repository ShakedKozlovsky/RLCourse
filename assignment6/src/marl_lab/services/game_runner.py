"""Game runner — orchestrates the 6 sub-games of one "game" (spec § 3.4).

A "game" is N sub-games of cops-and-robbers, alternating which trained policy
plays which role. Per spec § 3.5, the output is a single GameReport JSON
(timezone-stamped) that matches the spec's example shape EXACTLY:
  - group_name, group_code, students (list of StudentEntry), github_repo,
    timezone, sub_games (list of SubGameResult), totals.

This layer is **pure orchestration**: it does NOT do any learning — it loads
two trained policies and plays them against each other. Used by:
  - cli `play-game` subcommand (local)
  - mcp/* servers (over the network)
  - Gmail report generator"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig, sub_game_score
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.shared.types import GameReport, StudentEntry, SubGameResult

AGENTS = ("cop", "thief")


@dataclass(frozen=True)
class RunnerConfig:
    """One-off config for the game runner (mirrors yaml `game` + n_sub_games)."""
    n_sub_games: int = 6
    grid_size: tuple[int, int] = (5, 5)
    max_moves: int = 25
    max_barriers: int = 5
    enable_barriers: bool = True
    observation_radius: int = 2


class GameRunner:
    """Plays one game (N sub-games) between two trained policies.

    ``policy_a`` plays cop in EVEN sub-game indices, thief in ODD — fair
    alternation since the spec doesn't pin a specific pattern."""

    def __init__(self, runner_cfg: RunnerConfig, reward_cfg: RewardConfig,
                 rng: np.random.Generator | None = None) -> None:
        self.runner_cfg = runner_cfg
        self.reward_cfg = reward_cfg
        self._rng = rng or np.random.default_rng(0)
        env_cfg = EnvConfig(
            grid_size=runner_cfg.grid_size,
            max_moves=runner_cfg.max_moves,
            max_barriers=runner_cfg.max_barriers,
            enable_barriers=runner_cfg.enable_barriers,
            observation_radius=runner_cfg.observation_radius,
        )
        self.env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, rng=self._rng)

    def _greedy_action(self, q_net: QPerAgent, obs: np.ndarray,
                        hidden: torch.Tensor, n_legal: int) -> tuple[int, torch.Tensor]:
        """Greedy argmax over the first ``n_legal`` actions."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_seq, new_hidden = q_net(obs_t, hidden=hidden)
            q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
        q_masked = q.copy()
        q_masked[n_legal:] = -np.inf
        return int(np.argmax(q_masked)), new_hidden

    def play_sub_game(self, q_a: QPerAgent, q_b: QPerAgent,
                       a_role: str, sub_game_id: int, seed: int) -> SubGameResult:
        """Play one sub-game with policy_a in role ``a_role``."""
        start = datetime.now(tz=timezone.utc)
        joint_obs = self.env.reset(seed=seed)
        h_a = q_a.init_hidden(batch_size=1)
        h_b = q_b.init_hidden(batch_size=1)
        moves = 0
        while True:
            cop_obs = joint_obs["cop"]
            thief_obs = joint_obs["thief"]
            if a_role == "cop":
                cop_action, h_a = self._greedy_action(q_a, cop_obs, h_a, n_legal=6)
                thief_action, h_b = self._greedy_action(q_b, thief_obs, h_b, n_legal=5)
            else:
                cop_action, h_b = self._greedy_action(q_b, cop_obs, h_b, n_legal=6)
                thief_action, h_a = self._greedy_action(q_a, thief_obs, h_a, n_legal=5)
            joint_obs, _, done, info = self.env.step(
                {"cop": cop_action, "thief": thief_action})
            moves += 1
            if done:
                break
        end = datetime.now(tz=timezone.utc)
        winner = info["winner"] or "thief"
        scores = sub_game_score(winner, self.reward_cfg)
        return SubGameResult(
            id=sub_game_id,
            start=start,
            end=end,
            moves=moves,
            winner=winner,
            scores=scores,
        )

    def play_full_game(
        self,
        q_a: QPerAgent,
        q_b: QPerAgent,
        students: list[StudentEntry],
        group_name: str,
        group_code: str,
        github_repo: str,
        timezone_name: str = "UTC",
        seed: int = 0,
    ) -> GameReport:
        """Run all sub-games and assemble the spec § 3.5 GameReport.

        ``totals`` accumulates per-role scores across ALL sub-games (matching
        the spec's example: totals = {"cop": Σ score_cop, "thief": Σ score_thief}
        across the full game). Returns a fully-populated GameReport."""
        sub_games: list[SubGameResult] = []
        totals: dict[str, int] = {"cop": 0, "thief": 0}
        for k in range(self.runner_cfg.n_sub_games):
            a_role = "cop" if k % 2 == 0 else "thief"
            res = self.play_sub_game(q_a, q_b, a_role, sub_game_id=k, seed=seed + k)
            sub_games.append(res)
            totals["cop"] += res.scores["cop"]
            totals["thief"] += res.scores["thief"]
        return GameReport(
            group_name=group_name,
            group_code=group_code,
            students=list(students),
            github_repo=github_repo,
            timezone=timezone_name,
            sub_games=sub_games,
            totals=totals,
        )
