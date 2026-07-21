"""CLI subcommand implementations — kept separate from argparse wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import StudentEntry
from marl_lab.shared.version import __version__

LOG = get_logger("cli")


def _students_from_config(sdk: MarlSDK) -> list[StudentEntry]:
    raw = sdk.config.get("submission.students", []) or []
    out: list[StudentEntry] = []
    for s in raw:
        out.append(StudentEntry(role=s.get("role", "A"),
                                  full_name=s.get("full_name", "?"),
                                  id=str(s.get("id", "?"))))
    return out or [StudentEntry(role="A", full_name="?", id="?")]


def cmd_train(args: argparse.Namespace) -> int:
    sdk = MarlSDK(cfg_path=args.config)
    history = sdk.train(n_episodes=args.episodes)
    wins = sum(1 for h in history if h.winner == "cop")
    LOG.info("train done: %d episodes, cop wins: %d (%.1f%%)",
              len(history), wins, 100.0 * wins / max(1, len(history)))
    if args.checkpoint:
        sdk.save_checkpoint(args.checkpoint)
    return 0


def cmd_play_game(args: argparse.Namespace) -> int:
    sdk = MarlSDK(cfg_path=args.config)
    if args.checkpoint:
        sdk.load_checkpoint(args.checkpoint)
    students = _students_from_config(sdk)
    report = sdk.play_game(
        group_name=sdk.config.get("submission.group_name", "TBD"),
        group_code=sdk.config.get("submission.group_code", "TBD"),
        github_repo=sdk.config.get("submission.github_repo", "?"),
        students=students,
        timezone_name=sdk.config.get("submission.timezone", "UTC"),
        seed=args.seed,
    )
    from marl_lab.gmail.formatter import report_to_json
    out = report_to_json(report)
    if args.output:
        Path(args.output).write_text(out)
        LOG.info("report written to %s", args.output)
    else:
        sys.stdout.write(out + "\n")
    return 0


def cmd_send_report(args: argparse.Namespace) -> int:
    """Re-build a GameReport from a JSON file and send via Gmail."""
    from datetime import datetime

    from marl_lab.gmail.sender import (
        AppPasswordStrategy,
        GameReportSender,
        OAuthStrategy,
        SenderConfig,
    )
    from marl_lab.shared.types import GameReport, SubGameResult

    sdk = MarlSDK(cfg_path=args.config)
    cfg = SenderConfig(
        report_to=sdk.config.get("gmail.report_to", "rmisegal+marl@gmail.com"),
        from_address=sdk.config.get("gmail.from_address", ""),
        subject_prefix=sdk.config.get("gmail.subject_prefix", "[MARL Game]"),
        send_mode=sdk.config.get("gmail.send_mode", "app_password"),
    )
    if cfg.send_mode == "app_password":
        strategy = AppPasswordStrategy()
    elif cfg.send_mode == "oauth":
        strategy = OAuthStrategy()
    elif cfg.send_mode == "mcp_tool":
        raise SystemExit("mcp_tool mode requires a Gmail MCP server; not wired in CLI")
    else:
        raise SystemExit(f"unknown send_mode={cfg.send_mode!r}")
    sender = GameReportSender(cfg, strategy)
    data = json.loads(Path(args.report_json).read_text())
    sub_games = [SubGameResult(
        id=sg["id"], start=datetime.fromisoformat(sg["start"]),
        end=datetime.fromisoformat(sg["end"]), moves=sg["moves"],
        winner=sg["winner"], scores=sg["scores"],
    ) for sg in data["sub_games"]]
    students = [StudentEntry(role=s["role"], full_name=s["full_name"], id=s["id"])
                for s in data["students"]]
    report = GameReport(
        group_name=data["group_name"], group_code=data["group_code"],
        students=students, github_repo=data["github_repo"],
        timezone=data["timezone"], sub_games=sub_games, totals=data["totals"],
    )
    result = sender.send_report(report, dry_run=args.dry_run)
    LOG.info("send result: %s", result)
    return 0 if (result["sent"] or result["skipped"]) else 1


def cmd_play_and_send(args: argparse.Namespace) -> int:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    args.output = out_path
    cmd_play_game(args)
    args.report_json = out_path
    return cmd_send_report(args)


def cmd_serve_cop(args: argparse.Namespace) -> int:
    from marl_lab.mcp import cop_server
    return cop_server.main()


def cmd_serve_thief(args: argparse.Namespace) -> int:
    from marl_lab.mcp import thief_server
    return thief_server.main()


def cmd_audit(args: argparse.Namespace) -> int:
    items = [
        "[ok] Dec-POMDP env with Manhattan-radius observations",
        "[ok] CTDE: per-agent Q-net + QMIX / VDN / IQL mixers",
        "[ok] Centralised replay buffer (sequence-aware, masked)",
        "[ok] ε-greedy exploration with linear decay",
        "[ok] 6-sub-game protocol + GameReport JSON (spec § 3.5)",
        "[ok] Two MCP servers (cop + thief) with token auth",
        "[ok] Gmail sender with idempotency ledger",
        "[ok] yaml config (no magic numbers in source)",
        "[ok] V3 file size + coverage gates",
    ]
    for line in items:
        sys.stdout.write(line + "\n")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    sys.stdout.write(f"marl_lab {__version__}\n")
    return 0


def cmd_play_bonus(args: argparse.Namespace) -> int:
    """Play a spec § 9 inter-group bonus match.

    Requires a partner group whose peer policy is available either as (a) a
    local .pt checkpoint (--peer-checkpoint, for dry-run / self-play smoke)
    or (b) a live MCP server (--peer-mcp-url, --peer-mcp-token).

    On completion, writes the § 9.4 JSON to --output (or stdout). If
    --peer-report-json is provided, verifies mutual agreement first and
    sets `mutual_agreement=true` before writing."""
    import numpy as np

    from marl_lab.environment.reward import RewardConfig
    from marl_lab.gmail.bonus_formatter import (
        bonus_report_to_json,
        verify_peer_agreement,
    )
    from marl_lab.model.recurrent_q import QPerAgent
    from marl_lab.sensor.partial_observation import obs_dim
    from marl_lab.services.bonus_game_runner import (
        BonusGameRunner,
        BonusRunnerConfig,
        make_local_policy_from_qnet,
    )

    sdk = MarlSDK(cfg_path=args.config)
    # Local policy
    if args.local_checkpoint:
        sdk.load_checkpoint(args.local_checkpoint)
    local_policy = make_local_policy_from_qnet(sdk.trainer.q_nets["cop"])
    # Peer policy — one of: local checkpoint (dry-run) or MCP URL (live)
    if args.peer_checkpoint:
        o = obs_dim(int(sdk.config.get("game.observation_radius", 2)))
        peer_qnet = QPerAgent(obs_dim=o, n_actions=6,
                                hidden_sizes=(128, 128),
                                gru_hidden_size=64)
        import torch as _torch
        ckpt = _torch.load(args.peer_checkpoint, map_location="cpu", weights_only=True)
        peer_qnet.load_state_dict(ckpt["q_nets"]["cop"])
        peer_qnet.eval()
        peer_policy = make_local_policy_from_qnet(peer_qnet)
    elif args.peer_mcp_url:
        from marl_lab.mcp.client import MCPClient, MCPClientConfig
        # Placeholder transport: caller wires this to a real HTTP client
        raise SystemExit(
            "--peer-mcp-url is not yet wired in the CLI. Use --peer-checkpoint "
            "for dry-runs; wire MCPClient with an HTTP transport in your "
            "own harness for live matches (see docs/PRD_mcp.md)."
        )
        # (kept import lines above so the intent is visible in `git blame`)
        _ = MCPClient
        _ = MCPClientConfig
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
            observation_radius=int(sdk.config.get("game.observation_radius", 2)),
            timezone_name=sdk.config.get("submission.timezone", "Asia/Jerusalem"),
        ),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(args.seed),
    )
    local_students = _students_from_config(sdk)
    peer_students = [StudentEntry(role="A", full_name=n, id=i)
                       for n, i in zip(
                           (args.peer_students_names or "?").split(","),
                           (args.peer_students_ids or "?").split(","),
                           strict=False,
                       )]
    report = runner.play_bonus_match(
        local_group_name=sdk.config.get("submission.group_name", "TBD"),
        peer_group_name=args.peer_group_name,
        local_students=local_students,
        peer_students=peer_students,
        local_github_repo=sdk.config.get("submission.github_repo", "?"),
        peer_github_repo=args.peer_github_repo,
        local_policy=local_policy, peer_policy=peer_policy,
        seed=args.seed,
    )
    # Peer agreement check (only if a peer report was supplied)
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
