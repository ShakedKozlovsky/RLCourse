"""Orchestrator for the spec § 9 inter-group bonus match.

Runs 6 sub-games with role alternation (spec § 9.1):
  - Sub-games 1-3: local group = cop, peer group = thief
  - Sub-games 4-6: swap — local group = thief, peer group = cop

The peer policy is injected as a callable ``(role, obs) → action`` — for
real matches this wraps an MCP client pointing at the peer group's cop or
thief server; for tests / dry-runs it can be a local dummy policy. This
keeps the runner **transport-agnostic** — the same code drives an
in-process smoke run and a real cross-network bonus match."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig, sub_game_score
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.services.bonus_scoring import compute_bonus_claim
from marl_lab.shared.types import (
    BonusGameReport,
    BonusSubGameResult,
    StudentEntry,
)

PolicyFn = Callable[[str, np.ndarray], int]        # (role, obs) → action


@dataclass(frozen=True)
class BonusRunnerConfig:
    """Env + match parameters for the bonus game."""
    n_sub_games_per_side: int = 3        # 3 + 3 = 6 total per spec § 9.1
    grid_size: tuple[int, int] = (5, 5)
    max_moves: int = 25
    max_barriers: int = 5
    enable_barriers: bool = True
    observation_radius: int = 2
    timezone_name: str = "Asia/Jerusalem"


def make_local_policy_from_qnet(q_net: QPerAgent) -> PolicyFn:
    """Wrap a trained Q-net as a decentralised-execution greedy policy."""
    def policy(role: str, obs: np.ndarray) -> int:
        n_legal = 6 if role == "cop" else 5
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_seq, _ = q_net(obs_t, hidden=None)
            q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
        q_masked = q.copy()
        q_masked[n_legal:] = -np.inf
        return int(np.argmax(q_masked))
    return policy


class BonusGameRunner:
    """Play one spec § 9 inter-group bonus match.

    Args:
        cfg: env + match params
        reward_cfg: per-sub-game scoring (spec § 3.4 Table 1)
        rng: numpy Generator; each sub-game seeds off ``seed + k``
    """

    def __init__(self, cfg: BonusRunnerConfig, reward_cfg: RewardConfig,
                 rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self._rng = rng or np.random.default_rng(0)
        env_cfg = EnvConfig(
            grid_size=cfg.grid_size, max_moves=cfg.max_moves,
            max_barriers=cfg.max_barriers,
            enable_barriers=cfg.enable_barriers,
            observation_radius=cfg.observation_radius,
        )
        self.env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=reward_cfg,
                                 rng=self._rng)

    def _play_one(self, cop_policy: PolicyFn, thief_policy: PolicyFn,
                    sub_game_id: int, cop_group: str, thief_group: str,
                    seed: int) -> BonusSubGameResult:
        """Play a single sub-game with the two supplied policies."""
        self.env.reset(seed=seed)
        joint_obs = self.env.reset(seed=seed)
        moves = 0
        while True:
            cop_a = cop_policy("cop", joint_obs["cop"])
            thief_a = thief_policy("thief", joint_obs["thief"])
            joint_obs, _, done, info = self.env.step(
                {"cop": int(cop_a), "thief": int(thief_a)})
            moves += 1
            if done:
                break
        winner = info["winner"] or "thief"
        scores = sub_game_score(winner, self.reward_cfg)
        return BonusSubGameResult(
            id=sub_game_id, cop_group=cop_group, thief_group=thief_group,
            winner=winner, scores=scores,
        )

    def play_bonus_match(
        self, *,
        local_group_name: str,
        peer_group_name: str,
        local_students: list[StudentEntry],
        peer_students: list[StudentEntry],
        local_github_repo: str,
        peer_github_repo: str,
        local_policy: PolicyFn,
        peer_policy: PolicyFn,
        seed: int = 0,
    ) -> BonusGameReport:
        """Run the full 6-sub-game bonus match and return a spec § 9.4 report.

        Sub-games 1..N: local group = cop, peer group = thief.
        Sub-games N+1..2N: swap. N = n_sub_games_per_side (default 3).

        `bonus_claim` is computed via `compute_bonus_claim(totals_by_group)`.
        `mutual_agreement` is set to False by default — flip to True only
        after receiving + verifying the peer's report externally (see
        `gmail.bonus_formatter.verify_peer_agreement`)."""
        sub_games: list[BonusSubGameResult] = []
        totals: dict[str, int] = {local_group_name: 0, peer_group_name: 0}

        # Half 1: local = cop, peer = thief
        for k in range(self.cfg.n_sub_games_per_side):
            res = self._play_one(
                cop_policy=local_policy, thief_policy=peer_policy,
                sub_game_id=k + 1,
                cop_group=local_group_name, thief_group=peer_group_name,
                seed=seed + k,
            )
            sub_games.append(res)
            totals[local_group_name] += res.scores["cop"]
            totals[peer_group_name] += res.scores["thief"]

        # Half 2: swap
        for k in range(self.cfg.n_sub_games_per_side):
            res = self._play_one(
                cop_policy=peer_policy, thief_policy=local_policy,
                sub_game_id=self.cfg.n_sub_games_per_side + k + 1,
                cop_group=peer_group_name, thief_group=local_group_name,
                seed=seed + 1000 + k,
            )
            sub_games.append(res)
            totals[peer_group_name] += res.scores["cop"]
            totals[local_group_name] += res.scores["thief"]

        bonus_claim = compute_bonus_claim(totals)
        tz = ZoneInfo(self.cfg.timezone_name)
        _ = datetime.now(tz=tz)   # touch tz for possible future report-time stamp
        return BonusGameReport(
            groups={"group_1": local_group_name, "group_2": peer_group_name},
            github_repo_group_1=local_github_repo,
            github_repo_group_2=peer_github_repo,
            timezone=self.cfg.timezone_name,
            students_group_1=list(local_students),
            students_group_2=list(peer_students),
            sub_games=sub_games,
            totals_by_group=totals,
            bonus_claim=bonus_claim,
            mutual_agreement=False,
        )
