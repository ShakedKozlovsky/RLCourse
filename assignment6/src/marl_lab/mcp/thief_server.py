"""FastMCP thief server — thin wiring over BaseMCPServer + ThiefMCPServer."""

from __future__ import annotations

import argparse
import json

import torch

from marl_lab.auth.token_registry import TokenRegistry
from marl_lab.mcp.protocol import SelectActionRequest
from marl_lab.mcp.server_base import ThiefMCPServer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim
from marl_lab.shared.logger import get_logger

LOG = get_logger("mcp.thief")


def build_server(checkpoint_path: str, observation_radius: int = 2) -> ThiefMCPServer:
    """Build a ThiefMCPServer from a checkpoint + the env's observation radius."""
    o = obs_dim(observation_radius)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128), gru_hidden_size=64)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    q_net.load_state_dict(ckpt["q_nets"]["thief"])
    q_net.eval()
    tokens = TokenRegistry()
    if len(tokens) == 0:
        LOG.warning("MARL_MCP_ALLOWED_TOKENS is empty — all requests will be denied")
    return ThiefMCPServer(q_net=q_net, token_registry=tokens)


def main() -> int:
    """CLI entry point. Wraps server in FastMCP if installed, else exits."""
    parser = argparse.ArgumentParser(description="MARL Thief MCP server")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--port", type=int, default=7302)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--obs-radius", type=int, default=2)
    args = parser.parse_args()
    server = build_server(args.checkpoint, observation_radius=args.obs_radius)

    try:
        from fastmcp import FastMCP
    except ImportError:
        LOG.error("fastmcp not installed; install with `uv add fastmcp`")
        return 1

    mcp = FastMCP("marl-thief")

    @mcp.tool()
    def select_action(payload: str) -> str:
        """select_action(payload) — JSON request → JSON response."""
        req = SelectActionRequest.from_dict(json.loads(payload))
        resp = server.select_action(req)
        return json.dumps(resp.to_dict())

    LOG.info("Starting thief MCP server on %s:%d", args.host, args.port)
    mcp.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
