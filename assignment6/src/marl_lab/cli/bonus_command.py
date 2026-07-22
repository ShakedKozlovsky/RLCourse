"""cmd_play_bonus — extracted from cli/commands.py to keep it ≤ 250 LOC.

Full spec § 9 inter-group bonus match orchestration:
  - Local policy from checkpoint OR from the SDK's default trainer
  - Peer policy from checkpoint (dry-run) OR from a live MCP HTTP server
  - After match: verify peer agreement + set mutual_agreement flag
  - Write § 9.4 JSON to --output (or stdout)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from marl_lab.environment.reward import RewardConfig
from marl_lab.gmail.bonus_formatter import (
    bonus_report_to_json,
    verify_peer_agreement,
)
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.sensor.partial_observation import obs_dim
from marl_lab.services.bonus_game_runner import (
    BonusGameRunner,
    BonusRunnerConfig,
    make_local_policy_from_qnet,
)
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import StudentEntry

LOG = get_logger("cli.bonus")


def _peer_policy_from_checkpoint(peer_ckpt: str, obs_r: int):
    """Load peer's Q-net from local .pt file → greedy policy fn."""
    import torch
    o = obs_dim(obs_r)
    qnet = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128),
                      gru_hidden_size=64)
    ckpt = torch.load(peer_ckpt, map_location="cpu", weights_only=True)
    qnet.load_state_dict(ckpt["q_nets"]["cop"])
    qnet.eval()
    return make_local_policy_from_qnet(qnet)


def _peer_policy_from_mcp(url: str, token: str | None):
    """Live cross-network peer via MCP HTTP transport."""
    from marl_lab.mcp.http_transport import build_http_transport
    from marl_lab.mcp.protocol import SelectActionRequest, SelectActionResponse
    if not token:
        raise SystemExit(
            "--peer-mcp-url requires --peer-mcp-token (peer's auth token)"
        )
    transport = build_http_transport(url, token)

    def peer_policy(role: str, obs: np.ndarray) -> int:
        req = SelectActionRequest(
            agent_role=role, observation=obs.astype(float).tolist(),
            episode_step=0, auth_token=token,
        )
        resp_json = transport(json.dumps(req.to_dict()))
        d = json.loads(resp_json)
        resp = SelectActionResponse(
            action=int(d["action"]),
            q_value_for_action=float(d["q_value_for_action"]),
            server_role=d["server_role"],
        )
        if resp.server_role != role:
            raise RuntimeError(
                f"peer server role mismatch: asked={role}, got={resp.server_role}"
            )
        return resp.action
    LOG.info("bonus: wired to peer MCP at %s", url)
    return peer_policy


def _students_from_config(sdk: MarlSDK) -> list[StudentEntry]:
    raw = sdk.config.get("submission.students", []) or []
    out = []
    for s in raw:
        out.append(StudentEntry(role=s.get("role", "A"),
                                  full_name=s.get("full_name", "?"),
                                  id=str(s.get("id", "?"))))
    return out or [StudentEntry(role="A", full_name="?", id="?")]


def cmd_play_bonus(args: argparse.Namespace) -> int:
    """Play a spec § 9 inter-group bonus match. See module docstring."""
    sdk = MarlSDK(cfg_path=args.config)
    if args.local_checkpoint:
        sdk.load_checkpoint(args.local_checkpoint)
    local_policy = make_local_policy_from_qnet(sdk.trainer.q_nets["cop"])
    obs_r = int(sdk.config.get("game.observation_radius", 2))
    if args.peer_checkpoint:
        peer_policy = _peer_policy_from_checkpoint(args.peer_checkpoint, obs_r)
    elif args.peer_mcp_url:
        peer_policy = _peer_policy_from_mcp(args.peer_mcp_url,
                                              args.peer_mcp_token)
    else:
        raise SystemExit(
            "must supply either --peer-checkpoint (dry-run) or "
            "--peer-mcp-url + --peer-mcp-token (live)"
        )

    runner = BonusGameRunner(
        cfg=BonusRunnerConfig(
            grid_size=tuple(sdk.config.get("game.grid_size", (5, 5))),
            max_moves=int(sdk.config.get("game.max_moves", 25)),
            max_barriers=int(sdk.config.get("game.max_barriers", 5)),
            enable_barriers=bool(sdk.config.get("game.enable_barriers", True)),
            observation_radius=obs_r,
            timezone_name=sdk.config.get("submission.timezone", "Asia/Jerusalem"),
        ),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(args.seed),
    )
    peer_students = [StudentEntry(role="A", full_name=n, id=i)
                       for n, i in zip(
                           (args.peer_students_names or "?").split(","),
                           (args.peer_students_ids or "?").split(","),
                           strict=False,
                       )]
    report = runner.play_bonus_match(
        local_group_name=sdk.config.get("submission.group_name", "TBD"),
        peer_group_name=args.peer_group_name,
        local_students=_students_from_config(sdk),
        peer_students=peer_students,
        local_github_repo=sdk.config.get("submission.github_repo", "?"),
        peer_github_repo=args.peer_github_repo,
        local_policy=local_policy, peer_policy=peer_policy,
        seed=args.seed,
    )
    if args.peer_report_json:
        peer_txt = Path(args.peer_report_json).read_text()
        agreed, reason = verify_peer_agreement(report, peer_txt)
        report.mutual_agreement = agreed
        LOG.info("mutual_agreement=%s (%s)", agreed, reason)
    js = bonus_report_to_json(report)
    if args.output:
        Path(args.output).write_text(js)
        LOG.info("bonus report written to %s", args.output)
    else:
        sys.stdout.write(js + "\n")
    return 0
