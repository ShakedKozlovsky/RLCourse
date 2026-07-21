"""Tiny dataclasses passed between layers — frozen where state can't change.

MARL-specific shapes:
  * `Obs` — per-agent observation (numpy)
  * `JointAction` — dict {agent_id: action_int} for the 2-agent game
  * `Transition` — one timestep (s, ō, ā, r̄, s', ō', done)
  * `EpisodeSequence` — variable-length list of Transitions (one sub-game)
  * `SubGameResult` / `GameReport` — for the Gmail JSON (spec § 3.5 shape)
  * `StepDiagnostic` / `TrainResult` — training-time logging"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

AgentRole = Literal["cop", "thief"]
ActionInt = int  # 0..5 (UP, DOWN, LEFT, RIGHT, STAY, PLACE_BARRIER); cop has 6, thief has 5
Winner = Literal["cop", "thief", "draw"]


@dataclass(frozen=True)
class Obs:
    """Per-agent local observation (Manhattan-radius view + agent-role one-hot)."""
    agent_role: AgentRole
    vector: np.ndarray         # shape (obs_dim,)


@dataclass(frozen=True)
class Transition:
    """One environment timestep, with FULL state for centralised training."""
    global_state: np.ndarray
    joint_obs: dict[AgentRole, np.ndarray]
    joint_action: dict[AgentRole, ActionInt]
    joint_reward: dict[AgentRole, float]
    next_global_state: np.ndarray
    next_joint_obs: dict[AgentRole, np.ndarray]
    done: bool


@dataclass
class EpisodeSequence:
    """Variable-length sequence of Transitions for one sub-game."""
    transitions: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True)
class SubGameResult:
    """One sub-game's outcome — matches spec § 3.5 JSON shape."""
    id: int
    start: datetime
    end: datetime
    moves: int
    winner: Winner
    scores: dict[AgentRole, int]    # e.g. {"cop": 20, "thief": 5}


@dataclass(frozen=True)
class StudentEntry:
    """Per-student record in the Gmail JSON body."""
    role: Literal["A", "B", "C"]
    full_name: str
    id: str


@dataclass
class GameReport:
    """One game (6 sub-games) — matches spec § 3.5 example exactly."""
    group_name: str
    group_code: str                  # 8-char unique
    students: list[StudentEntry]
    github_repo: str
    timezone: str
    sub_games: list[SubGameResult] = field(default_factory=list)
    totals: dict[AgentRole, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BonusSubGameResult:
    """One sub-game outcome in a spec § 9 inter-group bonus match.

    Differs from SubGameResult by naming the GROUP holding each role
    (cop_group, thief_group) instead of just tagging the role — the same
    (cop, thief) roles may swap between the two groups across sub-games."""
    id: int
    cop_group: str
    thief_group: str
    winner: Winner
    scores: dict[AgentRole, int]     # {"cop": X, "thief": Y}


@dataclass
class BonusGameReport:
    """Inter-group bonus match report — matches spec § 9.4 JSON example exactly.

    Two groups play 6 sub-games total with roles alternating (Group 1 = cop
    for sub-games 1–3, then swap for sub-games 4–6). BOTH groups must send
    ONE bonus email each; bonus counts only if `mutual_agreement=True` in
    both reports."""
    groups: dict[Literal["group_1", "group_2"], str]  # {"group_1": "Team-A", ...}
    github_repo_group_1: str
    github_repo_group_2: str
    timezone: str
    students_group_1: list[StudentEntry]
    students_group_2: list[StudentEntry]
    sub_games: list[BonusSubGameResult] = field(default_factory=list)
    totals_by_group: dict[str, int] = field(default_factory=dict)   # {group_name: total_score}
    bonus_claim: dict[str, int] = field(default_factory=dict)       # {group_name: bonus_pts (5/7/10)}
    mutual_agreement: bool = False


@dataclass(frozen=True)
class StepDiagnostic:
    """Per-update logging tuple (mirrors A4/A5)."""
    step: int
    episode: int
    actor_loss: float
    critic_loss: float
    mixer_loss: float
    mean_q_cop: float
    mean_q_thief: float
    epsilon: float
    episode_reward_cop: float
    episode_reward_thief: float


@dataclass
class TrainResult:
    """Aggregate of all per-step diagnostics + final-episode summary."""
    diagnostics: list[StepDiagnostic] = field(default_factory=list)
    final_subgame: SubGameResult | None = None
