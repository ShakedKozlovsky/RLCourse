"""Layer 16 — MCP client + adjudicator-over-MCP (in-process transport)."""

from __future__ import annotations

import json

import pytest

from marl_lab.auth.token_registry import TokenRegistry
from marl_lab.environment.dec_pomdp import EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.mcp.client import MCPClient, MCPClientConfig
from marl_lab.mcp.protocol import SelectActionRequest
from marl_lab.mcp.server_base import CopMCPServer, ThiefMCPServer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim


def _make_in_process_transport(server) -> callable:
    """Return a transport function that calls ``server.select_action`` in-process."""
    def transport(payload: str) -> str:
        req = SelectActionRequest.from_dict(json.loads(payload))
        resp = server.select_action(req)
        return json.dumps(resp.to_dict())
    return transport


@pytest.fixture
def client_pair() -> tuple[MCPClient, CopMCPServer, ThiefMCPServer]:
    o = obs_dim(2)
    cop_q = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    thief_q = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    cop_server = CopMCPServer(q_net=cop_q, token_registry=TokenRegistry(tokens=["cop-tk"]))
    thief_server = ThiefMCPServer(q_net=thief_q, token_registry=TokenRegistry(tokens=["thief-tk"]))
    cfg = MCPClientConfig(cop_token="cop-tk", thief_token="thief-tk")
    client = MCPClient(cop_transport=_make_in_process_transport(cop_server),
                        thief_transport=_make_in_process_transport(thief_server),
                        cfg=cfg)
    return client, cop_server, thief_server


def test_client_play_sub_game_returns_result(client_pair) -> None:
    client, _, _ = client_pair
    env_cfg = EnvConfig(grid_size=(5, 5), max_moves=10, max_barriers=2,
                        enable_barriers=False, observation_radius=2)
    res = client.play_sub_game(env_cfg=env_cfg, reward_cfg=RewardConfig(),
                                sub_game_id=0, seed=0)
    assert res.id == 0
    assert res.moves >= 1
    assert res.winner in ("cop", "thief", "draw")


def test_client_uses_correct_token_per_role(client_pair) -> None:
    """A request with the wrong token should be rejected by the server."""
    _, cop_server, _ = client_pair
    # Build a client that has the cop token but uses it for the thief — should fail
    bad_cfg = MCPClientConfig(cop_token="cop-tk", thief_token="wrong")
    bad_client = MCPClient(
        cop_transport=_make_in_process_transport(cop_server),
        thief_transport=_make_in_process_transport(cop_server),  # wrong wiring
        cfg=bad_cfg,
    )
    env_cfg = EnvConfig(grid_size=(5, 5), max_moves=10, max_barriers=2,
                        enable_barriers=False, observation_radius=2)
    with pytest.raises((RuntimeError, ValueError)):
        bad_client.play_sub_game(env_cfg=env_cfg, reward_cfg=RewardConfig(),
                                  sub_game_id=0, seed=0)


def test_client_server_role_validation(client_pair) -> None:
    """If the server returns the wrong server_role, the client must raise."""
    _, cop_server, thief_server = client_pair
    # Cross-wire transports so cop_transport actually returns thief responses
    cfg = MCPClientConfig(cop_token="thief-tk", thief_token="cop-tk")
    client = MCPClient(
        cop_transport=_make_in_process_transport(thief_server),
        thief_transport=_make_in_process_transport(cop_server),
        cfg=cfg,
    )
    env_cfg = EnvConfig(grid_size=(5, 5), max_moves=10, max_barriers=2,
                        enable_barriers=False, observation_radius=2)
    with pytest.raises((RuntimeError, ValueError)):
        client.play_sub_game(env_cfg=env_cfg, reward_cfg=RewardConfig(),
                              sub_game_id=0, seed=0)
