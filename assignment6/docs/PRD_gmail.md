# PRD — Gmail API + JSON formatter + idempotency

> Per-mechanism PRD. Spec § 3.5 + § 5.5 are the contracts.

## 1. The hard requirement (spec § 3.5)

> *"The email is sent **exactly once**, by the **cop**, at the END of the **entire game** (after all 6 sub-games are complete)."*

| Item | Value |
|---|---|
| Recipient | `rmisegal+marl@gmail.com` |
| Sender | The cop agent (i.e., the local game adjudicator running on the user's machine) |
| Frequency | Once per game (game = 6 sub-games) |
| Subject | Ordered (spec example: `[MARL Game] Team-Alpha Game-1`) |
| Body | JSON per § 3.5 schema |

## 2. JSON body schema (spec § 3.5)

```json
{
  "group_name": "Team-Alpha",
  "students": [
    {"role": "A", "full_name": "...", "id": "123456789"},
    {"role": "B", "full_name": "...", "id": "234567890"}
  ],
  "github_repo": "https://github.com/...",
  "timezone": "Asia/Jerusalem",
  "sub_games": [
    {"id": 1, "start": "2026-06-17T18:00:05+03:00", "end": "...", "moves": 17, "winner": "cop", "scores": {"cop": 20, "thief": 5}},
    ...  // 6 entries total
  ],
  "totals": {"cop": 90, "thief": 40}
}
```

All 6 sub-games. ISO-8601 timestamps with timezone offset. The `totals` row equals the per-agent sum of `scores`.

## 3. Three implementation strategies (spec § 5.5 invites the student's choice)

| Strategy | Library | Setup difficulty | Reliability |
|---|---|---|---|
| **A. App Password + smtplib** | stdlib | Easy (2FA + generate App Password) | High |
| **B. Gmail API + OAuth** | `google-api-python-client` | Medium (Google Cloud Console + credentials.json) | High |
| **C. Custom MCP tool** | `fastmcp` + smtplib | Medium (wraps A behind MCP) | Highest (cloud-runnable) |

We implement **all three** behind a common `Sender` interface, with the default being `app_password` (simplest path to first success).

## 4. The Sender protocol

```python
class Sender(Protocol):
    def send(self, report: GameReport) -> str:
        """Send `report` as a Gmail message. Return the message_id.

        MUST be idempotent: calling twice with the same report.game_id is a
        no-op (the second call returns the original message_id from the local
        ledger).
        """
```

Implementations:

| Class | Path | Notes |
|---|---|---|
| `AppPasswordSender` | `gmail/senders/app_password.py` | smtplib + STARTTLS to `smtp.gmail.com:587` |
| `OAuthSender` | `gmail/senders/oauth.py` | Google API client; runs OAuth flow on first call |
| `McpToolSender` | `gmail/senders/mcp_tool.py` | Wraps either of the above as an MCP tool |
| `DryRunSender` | `gmail/senders/dry_run.py` | For testing — prints the email instead of sending |

## 5. Idempotency guard (ADR-010)

Spec mandates "exactly once". Implementation:

- `results/sent_games.json` is a ledger: `{"<game_id>": {"sent_at": "...", "message_id": "..."}}`
- Every `send()` call:
  1. Computes `game_id` deterministically from `report` content (SHA-256 of the JSON).
  2. Checks the ledger; if `game_id` already present, returns the stored `message_id` with a warning.
  3. Else: sends, then writes to the ledger atomically.

## 6. The formatter

```python
# gmail/formatter.py
def build_email(report: GameReport, cfg: GmailConfig) -> tuple[str, str]:
    """Returns (subject, body_json) per spec § 3.5."""
    subject = f"{cfg.subject_prefix} {report.group_name} Game-{report.game_id}"
    body = json.dumps(report.to_dict(), indent=2, default=_iso8601_default)
    return subject, body
```

`to_dict()` produces the canonical schema; `_iso8601_default` serialises `datetime` with the user's timezone (from `configs/setup.yaml::submission.timezone`).

## 7. Bonus emails (spec § 9.4)

The 10-point bonus game produces a DIFFERENT report shape:

```json
{
  "report_type": "bonus_game",
  "groups": {"group_1": "...", "group_2": "..."},
  "github_repo_group_1": "...",
  "github_repo_group_2": "...",
  "sub_games": [...],  // with cop_group + thief_group fields
  "totals_by_group": {...},
  "bonus_claim": {"Team-Alpha": 7, "Team-Beta": 10},
  "mutual_agreement": true
}
```

Both groups must send (idempotency guard still applies per-group). Disagreement on `totals_by_group` → 0 points.

## 8. Test plan

| Test | Pass criterion |
|---|---|
| `build_email(report)` shape | JSON schema-validates against spec § 3.5 example |
| `AppPasswordSender` dry-run | Prints subject + body; never calls SMTP |
| `OAuthSender` dry-run | Same |
| Idempotency: send + send same report | Second call returns cached message_id |
| Different `game_id` → real send | (Mocked SMTP) — actually attempts to send |
| ISO-8601 timestamp + TZ | parseable + matches `submission.timezone` |
| Bonus report shape | Matches spec § 9.4 example |

## 9. Acceptance criteria

1. All four sender implementations behind one Protocol.
2. Default is `app_password`; CLI flag swaps the implementation.
3. Idempotency ledger committed: `results/sent_games.json` is .gitignored (PII) BUT a sample is in `assets/diagrams/sent_games.example.json`.
4. The full email body passes a strict JSON-schema match against spec § 3.5.
5. `marl-lab report --dry-run` prints the exact email body that would be sent.

## 10. Risks + mitigations

| Risk | Mitigation |
|---|---|
| User doesn't have an App Password set up | README has step-by-step guide; CLI has `--dry-run` mode |
| Gmail throttles | Idempotency guard means we never retry-spam |
| Wrong recipient (typo) | Recipient is config-driven; default in `setup.yaml` matches the spec |
| Lecturer can't tell us apart | Subject + JSON `group_name` are both ordered + unique |

## 11. Non-goals

- HTML email (plain JSON is required + sufficient)
- Attachments (the JSON body is the whole report)
- Multi-recipient (only `rmisegal+marl@gmail.com` per spec)

## 12. Citation

- Google App Passwords: https://myaccount.google.com/apppasswords
- Gmail API: https://developers.google.com/gmail/api/quickstart/python
