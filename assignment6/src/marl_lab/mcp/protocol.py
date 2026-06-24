"""MCP tool contract — shared between cop and thief servers (PRD_mcp § 2).

The MCP protocol exposes ONE tool per server: ``select_action``. Both share
the same request/response shape so the client is uniform. Global state is
NEVER sent over the wire (ADR-001 / spec § 5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectActionRequest:
    """``select_action`` input — local observation only."""
    agent_role: str                # 'cop' or 'thief'
    observation: list[float]       # serialised obs vector (numpy → list[float])
    episode_step: int = 0          # for logging only; server doesn't trust it
    auth_token: str | None = None  # validated by AuthMiddleware

    @staticmethod
    def from_dict(d: dict) -> SelectActionRequest:
        return SelectActionRequest(
            agent_role=d["agent_role"],
            observation=list(d["observation"]),
            episode_step=int(d.get("episode_step", 0)),
            auth_token=d.get("auth_token"),
        )

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role,
            "observation": list(self.observation),
            "episode_step": int(self.episode_step),
            "auth_token": self.auth_token,
        }

    def observation_array(self) -> np.ndarray:
        return np.asarray(self.observation, dtype=np.float32)


@dataclass(frozen=True)
class SelectActionResponse:
    """``select_action`` output — discrete action only."""
    action: int                    # 0..5
    q_value_for_action: float      # the picked action's Q (NOT all of them)
    server_role: str               # 'cop' or 'thief' — for client-side validation

    def to_dict(self) -> dict:
        return {
            "action": int(self.action),
            "q_value_for_action": float(self.q_value_for_action),
            "server_role": self.server_role,
        }
