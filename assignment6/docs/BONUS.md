# Spec § 9 — Inter-group bonus match (10 pts)

> **60-second summary.** This assignment's bonus (spec § 9) is a full
> inter-group match: two groups' trained agents play a 6-sub-game series,
> alternate roles at the halfway point, and BOTH groups email a JSON
> report that has to *mutually agree* on the match content before either
> group can claim the bonus points. Winner gets 10, loser 7, tie 5.
>
> The infrastructure is complete and demoable end-to-end without a
> partner group via `scripts/bonus_demo.py`. A real live match against a
> partner needs their MCP HTTP server URL.

## Files that make up the bonus

| Concern | File |
|---|---|
| 6-sub-game runner + role-swap orchestration | [`src/marl_lab/services/bonus_game_runner.py`](../src/marl_lab/services/bonus_game_runner.py) |
| Bonus scoring rule (10 / 7 / 5 per § 9.2) | [`src/marl_lab/services/bonus_scoring.py`](../src/marl_lab/services/bonus_scoring.py) |
| § 9.4 JSON shape + subject line + idempotency key + peer-agreement | [`src/marl_lab/gmail/bonus_formatter.py`](../src/marl_lab/gmail/bonus_formatter.py) |
| Bonus-report Gmail sender | [`src/marl_lab/gmail/sender.py::GameReportSender.send_bonus_report`](../src/marl_lab/gmail/sender.py) |
| CLI `play-bonus` + `play-bonus-and-send` | [`src/marl_lab/cli/bonus_command.py`](../src/marl_lab/cli/bonus_command.py) |
| Cross-machine peer transport | [`src/marl_lab/mcp/http_transport.py`](../src/marl_lab/mcp/http_transport.py) |
| Self-contained demo (MADDPG vs IQL, no partner needed) | [`scripts/bonus_demo.py`](../scripts/bonus_demo.py) |
| Tests (22 total) | [`tests/unit/test_bonus_game.py`](../tests/unit/test_bonus_game.py) |

## Reproduction paths

### Path A — solo demo (no partner group)

Runs a full bonus match between two DIFFERENT shipped checkpoints
(MADDPG as "Team-MADDPG", IQL as "Team-IQL") and produces a valid
§ 9.4 JSON. The peer-agreement handshake is emulated on the same
machine — see `_simulate_peer_report` in the demo script for the
faithful "peer observes the same moves and flips only the group_1/
group_2 labels" model.

```bash
cd assignment6
uv run python scripts/bonus_demo.py --seed 0 \
    --out assets/logs/bonus_demo.json
```

Expected stderr (deterministic given `--seed 0`):

```
bonus report written to assets/logs/bonus_demo.json
subject: [MARL Bonus Game] Team-MADDPG vs Team-IQL – Final Report
idempotency key: 6c25eebb1655e15f
totals: {'Team-MADDPG': 85, 'Team-IQL': 45}
bonus_claim: {'Team-MADDPG': 10, 'Team-IQL': 7}
mutual_agreement: True (match)
```

The JSON on disk has the exact § 9.4 shape (see § "JSON schema" below).

### Path B — dry-run against a peer's checkpoint file

For coordination *before* both machines are up and playing live.
Both groups exchange checkpoints; each runs:

```bash
uv run marl play-bonus \
    --local-checkpoint saved_models/maddpg_shaped.pt \
    --peer-checkpoint  path/to/peer_group.pt \
    --peer-group-name "Team-Beta" \
    --peer-github-repo "https://github.com/other/repo" \
    --peer-students-names "Alice,Bob" \
    --peer-students-ids "111,222" \
    --seed 0 \
    --output assets/logs/bonus_report.json
```

Same seed on both machines → identical rollouts → both reports agree
on match content. Then feed each other's JSON with `--peer-report-json`
to lock in `mutual_agreement=True`.

### Path C — live cross-machine match

Peer group runs their MCP HTTP server (`marl serve-cop` /
`serve-thief` via `mcp/http_transport.py`). We call it:

```bash
uv run marl play-bonus-and-send \
    --local-checkpoint saved_models/maddpg_shaped.pt \
    --peer-mcp-url "http://peer.example.com:7301" \
    --peer-mcp-token "$PEER_TOKEN" \
    --peer-group-name "Team-Beta" \
    --peer-github-repo "https://github.com/other/repo" \
    --peer-students-names "Alice,Bob" \
    --peer-students-ids "111,222" \
    --peer-report-json /path/to/peer_final_report.json \
    --to rmisegal+marl@gmail.com \
    --output assets/logs/bonus_report.json
```

`play-bonus-and-send` plays the match, verifies mutual agreement, and
emails the § 9.4 report via the same idempotent sender as
`play-and-send` (uses `[MARL Bonus Game]` subject prefix). Add
`--dry-run` for a build-and-log-without-send.

## Scoring rule (spec § 9.2)

Applied by [`compute_bonus_claim`](../src/marl_lab/services/bonus_scoring.py):

| Outcome | Winner | Loser |
|---|---|---|
| Different totals | **10 pts** | 7 pts |
| Equal totals | 5 pts | 5 pts |

Totals are the sum of Table-1 per-sub-game scores each group
accumulated across all 6 games (both as cop and as thief).

## Mutual agreement (spec § 9)

The bonus is only claimable if BOTH groups' reports agree on the match
content. The canonicaliser
([`_canonical_match_content`](../src/marl_lab/gmail/bonus_formatter.py))
compares:

- **`groups`** — normalised as a *sorted list of team names*
  (v1.17 fix: previously compared the raw `{group_1, group_2}` dict,
  which spuriously failed because each team assigns those positional
  labels arbitrarily from their own perspective).
- **`sub_games`** — per-sub-game `{id, cop_group, thief_group, winner,
  scores}`, sorted by id.
- **`totals_by_group`** — the per-team totals dict.
- **`bonus_claim`** — the per-team claimed bonus.

Explicitly excluded from the canonical view: `provenance` (env-
dependent), `mutual_agreement` (that's the output flag), `students`
(each group only definitively knows their own IDs before match time).

`verify_peer_agreement` returns `(True, "match")` on agreement or
`(False, "disagreement on 'FIELD'")` naming the first mismatched
field — helpful when e.g. one team has a stale checkpoint or a
different `EnvConfig`.

## JSON schema (spec § 9.4)

Deterministic (sort_keys=True). Truncated example — see
`assets/logs/bonus_demo.json` for the full file after running Path A.

```json
{
  "bonus_claim": { "Team-IQL": 7, "Team-MADDPG": 10 },
  "groups": { "group_1": "Team-MADDPG", "group_2": "Team-IQL" },
  "github_repo_group_1": "https://github.com/ShakedKozlovsky/RLCourse",
  "github_repo_group_2": "https://github.com/peer/repo",
  "mutual_agreement": true,
  "provenance": { "git_sha": "...", "marl_lab_version": "1.17", ... },
  "report_type": "bonus_game",
  "students_group_1": [ { "role": "A", "full_name": "...", "id": "..." } ],
  "students_group_2": [ ... ],
  "sub_games": [
    { "id": 1, "cop_group": "Team-MADDPG", "thief_group": "Team-IQL",
      "winner": "cop", "scores": { "cop": 20, "thief": 5 } },
    ...
  ],
  "timezone": "Asia/Jerusalem",
  "totals_by_group": { "Team-IQL": 45, "Team-MADDPG": 85 }
}
```

Idempotency key = first 16 hex chars of the SHA-256 of the canonical
match content (excludes provenance + mutual_agreement so retries
after the peer-agreement handshake don't re-send).

## Tests

Run just the bonus suite:

```bash
uv run pytest tests/unit/test_bonus_game.py -v
```

22 tests cover: scoring rule (5), runner + role swap (4), JSON shape
+ subject + idempotency (4), peer-agreement including v1.17 regression
tests for the label-flip and invalid-winner bugs (7), CLI registration
(1), timezone (1).
