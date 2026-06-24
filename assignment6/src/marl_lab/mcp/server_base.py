"""Framework-agnostic MCP-server logic — testable without FastMCP runtime.

The actual FastMCP wiring lives in ``cop_server.py`` and ``thief_server.py``
(thin wrappers); this module is the brain. By keeping the policy logic +
auth in plain Python, the tests don't need a running server."""

from __future__ import annotations

import numpy as np
import torch

from marl_lab.auth.token_registry import TokenRegistry
from marl_lab.mcp.protocol import SelectActionRequest, SelectActionResponse
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.shared.logger import get_logger

LOG = get_logger("mcp.server")


class UnauthorizedError(RuntimeError):
    """Raised when the request's auth_token is not in the allowlist."""


class WrongRoleError(RuntimeError):
    """Raised when the request's agent_role mismatches this server's role."""


class BaseMCPServer:
    """Shared MCP server logic — wraps a single Q-net + token-registry guard.

    Each server **owns one role** (cop or thief). ``select_action`` does:
      1. Auth check (token allowlist)
      2. Role check (request must target this server's role)
      3. Greedy argmax restricted to legal actions"""

    def __init__(self, role: str, q_net: QPerAgent,
                 token_registry: TokenRegistry,
                 n_legal_actions: int) -> None:
        if role not in ("cop", "thief"):
            raise ValueError(f"role must be 'cop' or 'thief', got {role!r}")
        self.role = role
        self.q_net = q_net
        self.tokens = token_registry
        self.n_legal_actions = int(n_legal_actions)
        # Per-connection hidden state would require session tracking — for the
        # local test cycle a fresh hidden state per call is fine.
        self._hidden: torch.Tensor | None = None

    def reset_hidden(self) -> None:
        """Clear the GRU hidden state — called at the start of a new episode."""
        self._hidden = None

    def select_action(self, req: SelectActionRequest) -> SelectActionResponse:
        """Authorize, validate, run greedy argmax, return action."""
        if not self.tokens.is_authorised(req.auth_token):
            LOG.warning("denied: bad token (role=%s)", self.role)
            raise UnauthorizedError("invalid or missing auth_token")
        if req.agent_role != self.role:
            raise WrongRoleError(
                f"this server serves role={self.role!r}, "
                f"but request asked for {req.agent_role!r}"
            )
        obs = req.observation_array()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_seq, self._hidden = self.q_net(obs_t, hidden=self._hidden)
            q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
        # Restrict to legal actions
        q_masked = q.copy()
        q_masked[self.n_legal_actions:] = -np.inf
        action = int(np.argmax(q_masked))
        return SelectActionResponse(
            action=action,
            q_value_for_action=float(q[action]),
            server_role=self.role,
        )


class CopMCPServer(BaseMCPServer):
    """Cop MCP server (6 actions including PLACE_BARRIER)."""

    def __init__(self, q_net: QPerAgent, token_registry: TokenRegistry) -> None:
        super().__init__(role="cop", q_net=q_net, token_registry=token_registry,
                          n_legal_actions=6)


class ThiefMCPServer(BaseMCPServer):
    """Thief MCP server (5 actions; no barrier placement)."""

    def __init__(self, q_net: QPerAgent, token_registry: TokenRegistry) -> None:
        super().__init__(role="thief", q_net=q_net, token_registry=token_registry,
                          n_legal_actions=5)
