"""Layer 17 — Gmail formatter + ledger + sender (with FakeStrategy)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from marl_lab.gmail.formatter import build_idempotency_key, email_subject, report_to_json
from marl_lab.gmail.ledger import IdempotencyLedger
from marl_lab.gmail.sender import GameReportSender, GmailStrategy, SenderConfig
from marl_lab.shared.types import GameReport, StudentEntry, SubGameResult


def _dummy_report(seed: int = 0) -> GameReport:
    now = datetime(2026, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
    sg = [SubGameResult(id=i, start=now, end=now, moves=10 + i,
                          winner="cop" if i % 2 == 0 else "thief",
                          scores={"cop": 20 if i % 2 == 0 else 5,
                                  "thief": 5 if i % 2 == 0 else 10})
          for i in range(seed % 4 + 1)]
    return GameReport(
        group_name="g", group_code="C0DE1234",
        students=[StudentEntry(role="A", full_name="Shaked", id="1")],
        github_repo="https://github.com/x/y", timezone="UTC",
        sub_games=sg,
        totals={"cop": sum(s.scores["cop"] for s in sg),
                "thief": sum(s.scores["thief"] for s in sg)},
    )


# ----- Formatter -----

def test_report_to_json_is_valid_json() -> None:
    r = _dummy_report()
    js = report_to_json(r)
    data = json.loads(js)
    assert data["group_name"] == "g"
    assert "sub_games" in data
    assert "totals" in data


def test_idempotency_key_deterministic() -> None:
    r = _dummy_report()
    k1 = build_idempotency_key(r)
    k2 = build_idempotency_key(_dummy_report())   # same content → same hash
    assert k1 == k2
    assert len(k1) == 16


def test_idempotency_key_changes_with_content() -> None:
    k1 = build_idempotency_key(_dummy_report(seed=0))
    k2 = build_idempotency_key(_dummy_report(seed=1))
    assert k1 != k2


def test_email_subject_format() -> None:
    r = _dummy_report()
    subj = email_subject(r, prefix="[MARL Game]")
    assert subj.startswith("[MARL Game] g C0DE1234")
    assert "totals:" in subj


# ----- Ledger -----

def test_ledger_empty_at_start(tmp_path: Path) -> None:
    led = IdempotencyLedger(tmp_path / "ledger.json")
    assert len(led) == 0
    assert not led.has_been_sent("abc")


def test_ledger_records_and_persists(tmp_path: Path) -> None:
    led = IdempotencyLedger(tmp_path / "ledger.json")
    led.record_sent("abc", "subject")
    assert led.has_been_sent("abc")
    assert len(led) == 1
    # Reload from disk
    led2 = IdempotencyLedger(tmp_path / "ledger.json")
    assert led2.has_been_sent("abc")


def test_ledger_corrupt_file_recovers(tmp_path: Path) -> None:
    p = tmp_path / "ledger.json"
    p.write_text("not valid json {{{")
    led = IdempotencyLedger(p)
    assert len(led) == 0


# ----- Sender with FakeStrategy -----

class FakeStrategy(GmailStrategy):
    """In-memory strategy that records every send call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send(self, subject: str, body: str, to_address: str, from_address: str) -> None:
        self.calls.append({"subject": subject, "body": body,
                           "to": to_address, "from": from_address})


def test_sender_first_send_records_in_ledger(tmp_path: Path) -> None:
    cfg = SenderConfig(report_to="rmisegal+marl@gmail.com",
                       from_address="me@x.com",
                       ledger_path=str(tmp_path / "ledger.json"))
    strategy = FakeStrategy()
    sender = GameReportSender(cfg, strategy)
    result = sender.send_report(_dummy_report())
    assert result["sent"] is True
    assert not result["skipped"]
    assert len(strategy.calls) == 1
    assert strategy.calls[0]["to"] == "rmisegal+marl@gmail.com"


def test_sender_idempotency_skips_duplicate(tmp_path: Path) -> None:
    cfg = SenderConfig(report_to="x@y.com", from_address="me@x.com",
                       ledger_path=str(tmp_path / "ledger.json"))
    strategy = FakeStrategy()
    sender = GameReportSender(cfg, strategy)
    sender.send_report(_dummy_report())
    result = sender.send_report(_dummy_report())   # same content
    assert result["skipped"] is True
    assert result["reason"] == "already_sent"
    assert len(strategy.calls) == 1                  # not called twice


def test_sender_different_content_sends_again(tmp_path: Path) -> None:
    cfg = SenderConfig(report_to="x@y.com", from_address="me@x.com",
                       ledger_path=str(tmp_path / "ledger.json"))
    strategy = FakeStrategy()
    sender = GameReportSender(cfg, strategy)
    sender.send_report(_dummy_report(seed=0))
    sender.send_report(_dummy_report(seed=1))
    assert len(strategy.calls) == 2


def test_sender_dry_run_does_not_call_strategy(tmp_path: Path) -> None:
    cfg = SenderConfig(report_to="x@y.com", from_address="me@x.com",
                       ledger_path=str(tmp_path / "ledger.json"))
    strategy = FakeStrategy()
    sender = GameReportSender(cfg, strategy)
    result = sender.send_report(_dummy_report(), dry_run=True)
    assert result["skipped"] is True
    assert result["reason"] == "dry_run"
    assert len(strategy.calls) == 0


def test_app_password_strategy_raises_without_env(monkeypatch) -> None:
    from marl_lab.gmail.sender import AppPasswordStrategy
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    s = AppPasswordStrategy(sender_email="me@x.com")
    with pytest.raises(RuntimeError):
        s.send(subject="s", body="b", to_address="t@x.com", from_address="me@x.com")
