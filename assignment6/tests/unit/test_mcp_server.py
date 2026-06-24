"""Layer 15 — MCP server logic + auth + protocol tests."""

from __future__ import annotations

import pytest

from marl_lab.auth.token_registry import TokenRegistry
from marl_lab.mcp.protocol import SelectActionRequest, SelectActionResponse
from marl_lab.mcp.server_base import (
    CopMCPServer,
    ThiefMCPServer,
    UnauthorizedError,
    WrongRoleError,
)
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim

# ----- Token registry -----

def test_token_registry_default_empty() -> None:
    reg = TokenRegistry(tokens=[])
    assert len(reg) == 0
    assert not reg.is_authorised("anything")


def test_token_registry_authorises_known_token() -> None:
    reg = TokenRegistry(tokens=["abc", "xyz"])
    assert reg.is_authorised("abc")
    assert reg.is_authorised("xyz")


def test_token_registry_rejects_unknown_token() -> None:
    reg = TokenRegistry(tokens=["abc"])
    assert not reg.is_authorised("def")
    assert not reg.is_authorised("")
    assert not reg.is_authorised(None)


def test_token_registry_strips_whitespace() -> None:
    reg = TokenRegistry(tokens=["  abc  ", " "])
    assert len(reg) == 1
    assert reg.is_authorised("abc")


# ----- Protocol round-trip -----

def test_protocol_request_round_trip() -> None:
    req = SelectActionRequest(agent_role="cop", observation=[0.0, 1.0, 2.0],
                              episode_step=7, auth_token="abc")
    d = req.to_dict()
    req2 = SelectActionRequest.from_dict(d)
    assert req == req2


def test_protocol_response_round_trip() -> None:
    resp = SelectActionResponse(action=3, q_value_for_action=0.7, server_role="cop")
    d = resp.to_dict()
    assert d["action"] == 3
    assert d["server_role"] == "cop"


def test_protocol_observation_array_dtype() -> None:
    req = SelectActionRequest(agent_role="cop", observation=[1.0, 2.0, 3.0],
                              episode_step=0, auth_token=None)
    arr = req.observation_array()
    assert arr.dtype.name == "float32"
    assert arr.shape == (3,)


# ----- MCP server logic -----

def _make_cop_server(token: str = "tk") -> CopMCPServer:
    o = obs_dim(2)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    return CopMCPServer(q_net=q_net, token_registry=TokenRegistry(tokens=[token]))


def _make_thief_server(token: str = "tk") -> ThiefMCPServer:
    o = obs_dim(2)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    return ThiefMCPServer(q_net=q_net, token_registry=TokenRegistry(tokens=[token]))


def test_cop_server_returns_action_in_legal_range() -> None:
    server = _make_cop_server()
    req = SelectActionRequest(agent_role="cop", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="tk")
    resp = server.select_action(req)
    assert resp.server_role == "cop"
    assert 0 <= resp.action < 6


def test_thief_server_action_never_includes_barrier() -> None:
    server = _make_thief_server()
    req = SelectActionRequest(agent_role="thief", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="tk")
    resp = server.select_action(req)
    # PLACE_BARRIER is action 5; thief n_legal=5, so action must be in 0..4
    assert 0 <= resp.action < 5


def test_cop_server_rejects_bad_token() -> None:
    server = _make_cop_server()
    req = SelectActionRequest(agent_role="cop", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="wrong")
    with pytest.raises(UnauthorizedError):
        server.select_action(req)


def test_cop_server_rejects_missing_token() -> None:
    server = _make_cop_server()
    req = SelectActionRequest(agent_role="cop", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token=None)
    with pytest.raises(UnauthorizedError):
        server.select_action(req)


def test_cop_server_rejects_thief_request() -> None:
    server = _make_cop_server()
    req = SelectActionRequest(agent_role="thief", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="tk")
    with pytest.raises(WrongRoleError):
        server.select_action(req)


def test_thief_server_rejects_cop_request() -> None:
    server = _make_thief_server()
    req = SelectActionRequest(agent_role="cop", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="tk")
    with pytest.raises(WrongRoleError):
        server.select_action(req)


def test_server_invalid_role_raises() -> None:
    import torch
    o = obs_dim(2)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    with pytest.raises(ValueError):
        from marl_lab.mcp.server_base import BaseMCPServer
        BaseMCPServer(role="banana", q_net=q_net,
                       token_registry=TokenRegistry(tokens=["x"]),
                       n_legal_actions=5)
    _ = torch  # silence unused


def test_cop_server_reset_hidden_clears_state() -> None:
    server = _make_cop_server()
    req = SelectActionRequest(agent_role="cop", observation=[0.0] * obs_dim(2),
                              episode_step=0, auth_token="tk")
    server.select_action(req)
    assert server._hidden is not None        # noqa: SLF001
    server.reset_hidden()
    assert server._hidden is None             # noqa: SLF001
