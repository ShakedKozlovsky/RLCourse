"""Layer 18 — CLI subcommand smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from marl_lab.cli.main import build_parser, main


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00", "seed": 0, "device": "cpu",
        "game": {"grid_size": [4, 4], "max_moves": 8, "num_games": 2,
                  "max_barriers": 2, "enable_barriers": False, "observation_radius": 1},
        "scoring": {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5},
        "marl": {
            "algorithm": "qmix", "gamma": 0.99, "tau": 0.005,
            "critic_lr": 1e-3, "batch_size": 4, "replay_capacity": 32,
            "warmup_steps": 25, "max_grad_norm": 1.0,
            "hidden_sizes": [16], "rnn_hidden_size": 8,
            "embed_dim": 8, "hyper_hidden": 16,
        },
        "exploration": {"epsilon_initial": 1.0, "epsilon_final": 0.05, "decay_steps": 100},
        "training": {"total_episodes": 3},
        "experiments": {},
        "mcp": {}, "gmail": {"report_to": "x@y.com", "from_address": "me@x.com",
                                "send_mode": "app_password"},
        "submission": {
            "group_code": "TEST1234", "group_name": "T",
            "students": [{"role": "A", "full_name": "Shaked", "id": "1"}],
            "github_repo": "r", "timezone": "UTC",
        },
        "paths": {}, "graphify": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_version_subcommand(capsys) -> None:
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "marl_lab" in out


def test_audit_subcommand_prints_checklist(capsys) -> None:
    rc = main(["audit"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dec-POMDP" in out
    assert "QMIX" in out
    assert "MCP" in out


def test_parser_supports_all_10_subcommands() -> None:
    """V3 § 3 requires CLI parity with spec § 5.4 (GUI) + § 5.6 + § 9 (bonus).

    10 subcommands: 8 core + `play-bonus` (§ 9) + `gui` (live § 5.4 widget)."""
    parser = build_parser()
    sub_action = next(a for a in parser._actions if a.choices)   # noqa: SLF001
    choices = set(sub_action.choices.keys())
    expected = {"train", "play-game", "send-report", "play-and-send",
                "serve-cop", "serve-thief", "audit", "version",
                "play-bonus", "gui"}
    assert expected == choices


def test_train_subcommand_runs(tiny_cfg: Path, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    rc = main(["train", "--config", str(tiny_cfg), "--episodes", "3",
                "--checkpoint", str(ckpt)])
    assert rc == 0
    assert ckpt.exists()


def test_train_with_seed_flag_reproduces(tiny_cfg: Path, tmp_path: Path) -> None:
    """--seed override must produce identical Q-net weights on two runs."""
    import torch
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    main(["train", "--config", str(tiny_cfg), "--episodes", "3",
           "--seed", "12345", "--checkpoint", str(ckpt_a)])
    main(["train", "--config", str(tiny_cfg), "--episodes", "3",
           "--seed", "12345", "--checkpoint", str(ckpt_b)])
    a = torch.load(str(ckpt_a), map_location="cpu", weights_only=True)
    b = torch.load(str(ckpt_b), map_location="cpu", weights_only=True)
    for agent in ("cop", "thief"):
        for k in a["q_nets"][agent]:
            torch.testing.assert_close(a["q_nets"][agent][k], b["q_nets"][agent][k])


def test_train_with_curriculum_flag_runs(tiny_cfg: Path, tmp_path: Path) -> None:
    """--curriculum flag routes to curriculum-aware train path."""
    ckpt = tmp_path / "c.pt"
    rc = main(["train", "--config", str(tiny_cfg), "--episodes", "3",
                "--curriculum", "--checkpoint", str(ckpt)])
    assert rc == 0
    assert ckpt.exists()


def test_env_var_overrides_student_id(tiny_cfg: Path, tmp_path: Path,
                                          monkeypatch) -> None:
    """MARL_STUDENT_A_ID env var must override yaml (keeps ID out of committed yaml)."""
    monkeypatch.setenv("MARL_STUDENT_A_ID", "999999999")
    out = tmp_path / "report.json"
    rc = main(["play-game", "--config", str(tiny_cfg), "--output", str(out),
                "--seed", "0"])
    assert rc == 0
    import json
    data = json.loads(out.read_text())
    assert data["students"][0]["id"] == "999999999"


def test_env_var_overrides_group_code_and_name(tiny_cfg: Path, tmp_path: Path,
                                                    monkeypatch) -> None:
    """MARL_GROUP_CODE / MARL_GROUP_NAME env vars must override yaml."""
    monkeypatch.setenv("MARL_GROUP_CODE", "abc12345")
    monkeypatch.setenv("MARL_GROUP_NAME", "test-team")
    out = tmp_path / "report.json"
    rc = main(["play-game", "--config", str(tiny_cfg), "--output", str(out),
                "--seed", "0"])
    assert rc == 0
    import json
    data = json.loads(out.read_text())
    assert data["group_code"] == "abc12345"
    assert data["group_name"] == "test-team"


def test_play_game_subcommand_emits_json(tiny_cfg: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main(["play-game", "--config", str(tiny_cfg), "--output", str(out),
                "--seed", "0"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "sub_games" in data
    assert len(data["sub_games"]) == 2     # num_games from tiny_cfg
    assert "totals" in data
    assert data["group_code"] == "TEST1234"


def test_send_report_with_dry_run(tiny_cfg: Path, tmp_path: Path) -> None:
    # First play-game to get a real report
    out = tmp_path / "report.json"
    main(["play-game", "--config", str(tiny_cfg), "--output", str(out), "--seed", "0"])
    rc = main(["send-report", "--config", str(tiny_cfg),
                "--report-json", str(out), "--dry-run"])
    assert rc == 0
