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
    """Train MARL agents.

    Optional --seed overrides the yaml seed for this run (useful for
    reproducing / A-B testing / multi-seed sweeps). Optional --curriculum
    ramps grid size 2×2 → 3×3 → 4×4 → 5×5 as cop win-rate crosses each
    stage's threshold (Lin 2025, `services/curriculum.py`)."""
    sdk = MarlSDK(cfg_path=args.config)
    if args.seed is not None:
        # Re-seed the underlying trainer + torch/numpy/python for this run
        import numpy as np

        from marl_lab.shared.seed import set_global_seed
        set_global_seed(args.seed)
        sdk._rng = np.random.default_rng(args.seed)  # noqa: SLF001
        sdk.trainer._rng = np.random.default_rng(args.seed)  # noqa: SLF001
        LOG.info("train seed override: %d", args.seed)
    if args.curriculum:
        from marl_lab.services.curriculum import CurriculumSchedule
        curriculum = CurriculumSchedule()  # default: 2×2 → 3×3 → 4×4 → 5×5
        LOG.info("curriculum enabled: %s", [s.grid_size for s in curriculum.stages])
        history = sdk.trainer.train(n_episodes=args.episodes, curriculum=curriculum)
    else:
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


def cmd_gui(args: argparse.Namespace) -> int:
    """Open the live Tkinter window playing a real sub-game (spec § 5.4).

    On headless environments (no DISPLAY) this exits with a helpful message
    pointing at the pre-rendered ``assets/figures/sub_game.gif`` alternative."""
    from marl_lab.interface.tk_gui import launch_live_gui
    sdk = MarlSDK(cfg_path=args.config)
    grid = tuple(sdk.config.get("game.grid_size", (5, 5)))
    launch_live_gui(
        checkpoint_path=args.checkpoint,
        grid_size=grid,
        max_moves=int(sdk.config.get("game.max_moves", 25)),
        observation_radius=int(sdk.config.get("game.observation_radius", 2)),
        delay_ms=args.delay_ms,
    )
    return 0


# cmd_play_bonus lives in cli/bonus_command.py to keep this file ≤ 250 LOC
from marl_lab.cli.bonus_command import cmd_play_bonus  # noqa: E402, F401

# cmd_play_bonus lives in cli/bonus_command.py to keep this file lean
