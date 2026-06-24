"""Three Gmail send strategies + a unified ``Sender`` facade.

PRD_gmail § 4 — pick a strategy via ``send_mode`` config:
  - app_password  : smtplib + Google App Password (simplest, no OAuth)
  - oauth         : google-auth OAuth2 flow + Gmail API
  - mcp_tool      : delegate to a Gmail MCP server (per spec § 5.4 hint)

All three implement the same `send(subject, body, to_address)` interface.
Idempotency is enforced by the unified Sender — it checks the ledger
BEFORE invoking any strategy, and records on success."""

from __future__ import annotations

import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path

from marl_lab.gmail.formatter import build_idempotency_key, email_subject, report_to_json
from marl_lab.gmail.ledger import IdempotencyLedger
from marl_lab.shared.logger import get_logger
from marl_lab.shared.types import GameReport

LOG = get_logger("gmail.sender")


class GmailStrategy(ABC):
    """Abstract sender — one method, three implementations."""

    @abstractmethod
    def send(self, subject: str, body: str, to_address: str, from_address: str) -> None:
        """Send a plain-text email. Raises on failure."""


class AppPasswordStrategy(GmailStrategy):
    """smtplib over SMTPS:587 + Google App Password (env: GMAIL_APP_PASSWORD)."""

    def __init__(self, sender_email: str | None = None) -> None:
        self.sender_email = sender_email or os.environ.get("GMAIL_USER", "")

    def send(self, subject: str, body: str, to_address: str, from_address: str) -> None:
        app_password = os.environ.get("GMAIL_APP_PASSWORD")
        if not app_password:
            raise RuntimeError("GMAIL_APP_PASSWORD env var is not set")
        from_email = from_address or self.sender_email
        if not from_email:
            raise RuntimeError("from_address is empty and GMAIL_USER is not set")
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_address
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)


class OAuthStrategy(GmailStrategy):
    """Gmail API + OAuth2 — requires `google-api-python-client`.

    The credentials file path is read from ``GMAIL_OAUTH_CREDENTIALS`` and the
    refresh token from ``GMAIL_OAUTH_TOKEN``. The full OAuth dance happens on
    first run; we don't reproduce it here — production users should run a
    one-shot bootstrap script (out of scope for this layer)."""

    def send(self, subject: str, body: str, to_address: str, from_address: str) -> None:
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError as e:
            raise RuntimeError("google-api-python-client not installed") from e
        token_path = os.environ.get("GMAIL_OAUTH_TOKEN")
        if not token_path or not Path(token_path).exists():
            raise RuntimeError("GMAIL_OAUTH_TOKEN env var not set or file missing")
        creds = Credentials.from_authorized_user_file(token_path)
        service = build("gmail", "v1", credentials=creds)
        import base64
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = to_address
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        service.users().messages().send(userId="me", body={"raw": raw}).execute()


class MCPToolStrategy(GmailStrategy):
    """Delegate to a Gmail MCP server — caller passes an injected ``send_fn``."""

    def __init__(self, send_fn):
        self.send_fn = send_fn

    def send(self, subject: str, body: str, to_address: str, from_address: str) -> None:
        self.send_fn({"subject": subject, "body": body,
                      "to": to_address, "from": from_address})


@dataclass(frozen=True)
class SenderConfig:
    """Top-level Gmail config (mirrors yaml `gmail` block)."""
    report_to: str
    from_address: str
    subject_prefix: str = "[MARL Game]"
    send_mode: str = "app_password"        # 'app_password' | 'oauth' | 'mcp_tool'
    ledger_path: str = "assets/logs/sent_games.json"


class GameReportSender:
    """Unified facade — pick a strategy, enforce idempotency via the ledger."""

    def __init__(self, cfg: SenderConfig, strategy: GmailStrategy,
                 ledger: IdempotencyLedger | None = None) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.ledger = ledger or IdempotencyLedger(cfg.ledger_path)

    def send_report(self, report: GameReport, *, dry_run: bool = False) -> dict:
        """Send the GameReport. Returns {'sent', 'game_id', 'subject', 'skipped'}."""
        game_id = build_idempotency_key(report)
        subject = email_subject(report, prefix=self.cfg.subject_prefix)
        body = report_to_json(report)
        if self.ledger.has_been_sent(game_id):
            LOG.info("idempotency: game_id=%s already sent — skipping", game_id)
            return {"sent": False, "game_id": game_id, "subject": subject,
                    "skipped": True, "reason": "already_sent"}
        if dry_run:
            return {"sent": False, "game_id": game_id, "subject": subject,
                    "skipped": True, "reason": "dry_run"}
        self.strategy.send(subject=subject, body=body, to_address=self.cfg.report_to,
                           from_address=self.cfg.from_address)
        self.ledger.record_sent(game_id, subject)
        return {"sent": True, "game_id": game_id, "subject": subject, "skipped": False}
