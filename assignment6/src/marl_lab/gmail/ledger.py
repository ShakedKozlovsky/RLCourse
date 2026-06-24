"""Idempotency ledger — local JSON file recording which game_ids have been sent.

ADR-010: A re-run of the same game (same seed + same model) generates the
same content hash, which becomes the game_id; we refuse to re-send. This
protects against accidentally double-emailing the grader.

The ledger is a flat JSON list at ``assets/logs/sent_games.json``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class IdempotencyLedger:
    """Track which game_ids have been emailed before."""

    def __init__(self, ledger_path: Path | str) -> None:
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                self._entries = []

    def has_been_sent(self, game_id: str) -> bool:
        """True if game_id is already in the ledger."""
        return any(e["game_id"] == game_id for e in self._entries)

    def record_sent(self, game_id: str, subject: str) -> None:
        """Add a successful send to the ledger; persist to disk."""
        self._entries.append({
            "game_id": game_id,
            "subject": subject,
            "sent_at": datetime.now(tz=timezone.utc).isoformat(),
        })
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._entries)
