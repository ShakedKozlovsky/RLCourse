"""CLI entry point — ``marl`` command with subcommands.

Subcommands:
  train          — train MARL agents from yaml; save checkpoint
  play-game      — load checkpoint + play 6 sub-games; emit JSON
  send-report    — send a saved JSON report via Gmail (idempotent)
  play-and-send  — play + send in one go
  serve-cop      — start the cop MCP server
  serve-thief    — start the thief MCP server
  audit          — print spec-compliance audit
  version        — print the current marl_lab version

The cmd_* implementations live in cli/commands.py to keep this file lean."""

from __future__ import annotations

import argparse

from marl_lab.cli.commands import (
    cmd_audit,
    cmd_play_and_send,
    cmd_play_game,
    cmd_send_report,
    cmd_serve_cop,
    cmd_serve_thief,
    cmd_train,
    cmd_version,
)

DEFAULT_CONFIG = "configs/setup.yaml"


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Path to yaml config (default: configs/setup.yaml)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marl", description="MARL lab CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train MARL agents")
    _add_common_args(p_train)
    p_train.add_argument("--episodes", type=int, default=None)
    p_train.add_argument("--checkpoint", help="Where to save the .pt checkpoint")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play-game", help="Play 6 sub-games and emit JSON report")
    _add_common_args(p_play)
    p_play.add_argument("--checkpoint", help="Load trainer weights from this .pt file")
    p_play.add_argument("--output", help="Write JSON to this path (default: stdout)")
    p_play.add_argument("--seed", type=int, default=0)
    p_play.set_defaults(func=cmd_play_game)

    p_send = sub.add_parser("send-report", help="Send a report JSON via Gmail")
    _add_common_args(p_send)
    p_send.add_argument("--report-json", required=True)
    p_send.add_argument("--dry-run", action="store_true")
    p_send.set_defaults(func=cmd_send_report)

    p_ps = sub.add_parser("play-and-send", help="Play 6 sub-games then send report")
    _add_common_args(p_ps)
    p_ps.add_argument("--checkpoint")
    p_ps.add_argument("--seed", type=int, default=0)
    p_ps.add_argument("--dry-run", action="store_true")
    p_ps.set_defaults(func=cmd_play_and_send)

    p_cop = sub.add_parser("serve-cop", help="Start the cop MCP server")
    p_cop.set_defaults(func=cmd_serve_cop)

    p_thief = sub.add_parser("serve-thief", help="Start the thief MCP server")
    p_thief.set_defaults(func=cmd_serve_thief)

    p_audit = sub.add_parser("audit", help="Print spec-compliance audit")
    p_audit.set_defaults(func=cmd_audit)

    p_version = sub.add_parser("version", help="Print marl_lab version")
    p_version.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
