"""MCP client — talks to cop + thief MCP servers + a local adjudicator.

The client is injected with **transport functions** (one per role) that take
a JSON string and return a JSON string. In tests we inject in-process server
objects directly; in production these are FastMCP HTTP calls. This keeps
the adjudicator-over-MCP testable without spawning real servers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig, sub_game_score
from marl_lab.mcp.protocol import SelectActionRequest, SelectActionResponse
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import SubGameResult

LOG = get_logger("mcp.client")


Transport = Callable[[str], str]   # (json_request) → json_response


@dataclass(frozen=True)
class MCPClientConfig:
    """Auth + retry config (mirrors yaml `mcp` block)."""
    cop_token: str
    thief_token: str
    request_timeout_s: float = 10.0


class MCPClient:
    """Two-server MCP client with role-aware dispatch + a built-in adjudicator."""

    def __init__(self, cop_transport: Transport, thief_transport: Transport,
                 cfg: MCPClientConfig) -> None:
        self.cop_transport = cop_transport
        self.thief_transport = thief_transport
        self.cfg = cfg

    def _call(self, role: str, observation: np.ndarray, episode_step: int) -> SelectActionResponse:
        req = SelectActionRequest(
            agent_role=role,
            observation=observation.astype(float).tolist(),
            episode_step=int(episode_step),
            auth_token=self.cfg.cop_token if role == "cop" else self.cfg.thief_token,
        )
        transport = self.cop_transport if role == "cop" else self.thief_transport
        raw = transport(json.dumps(req.to_dict()))
        d = json.loads(raw)
        resp = SelectActionResponse(
            action=int(d["action"]),
            q_value_for_action=float(d["q_value_for_action"]),
            server_role=d["server_role"],
        )
        if resp.server_role != role:
            raise RuntimeError(
                f"server_role mismatch: asked={role!r}, got={resp.server_role!r}"
            )
        return resp

    def play_sub_game(self, env_cfg: EnvConfig, reward_cfg: RewardConfig,
                       sub_game_id: int, seed: int) -> SubGameResult:
        """Play ONE sub-game using the two remote policies. Returns SubGameResult."""
        from datetime import datetime, timezone
        env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=reward_cfg,
                            rng=np.random.default_rng(seed))
        joint_obs = env.reset(seed=seed)
        start = datetime.now(tz=timezone.utc)
        moves = 0
        while True:
            cop_resp = self._call("cop", joint_obs["cop"], moves)
            thief_resp = self._call("thief", joint_obs["thief"], moves)
            joint_obs, _, done, info = env.step(
                {"cop": cop_resp.action, "thief": thief_resp.action})
            moves += 1
            if done:
                break
        end = datetime.now(tz=timezone.utc)
        winner = info["winner"] or "thief"
        scores = sub_game_score(winner, reward_cfg)
        return SubGameResult(
            id=sub_game_id, start=start, end=end, moves=moves,
            winner=winner, scores=scores,
        )
