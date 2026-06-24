# marl-lab — Cooperative-adversarial MARL Cops-and-Robbers with cloud MCP

**Assignment 6 of the RL Course (L10 — Multi-Agent RL)** — the foundation of the course final project, graded as one unit with it.

A complete laboratory for **Multi-Agent Reinforcement Learning** on the *Cops-and-Robbers* pursuit-evasion grid, built under the **Dec-POMDP / CTDE / VDN-QMIX** paradigm. Both agents (Cop, Thief) train under centralised state access, then run behind their own **MCP server** with **automated Gmail-API reporting** at the end of every 6-sub-game game.

## Status

**v1.00 — feature-complete.** 202/202 tests green; ruff clean; LOC audit clean.

| Layer | Module | Status |
|---|---|---|
| 0  | scaffold + docs (PRD + PLAN + TODO + 8 per-mechanism PRDs) | ✅ |
| 1  | `shared/` — ConfigManager, Logger, seed, types | ✅ |
| 2  | `game/` — Board + Action + MoveDynamics + WinAdjudicator | ✅ |
| 3  | `sensor/` + `environment/` — Manhattan obs + Dec-POMDP env + reward | ✅ |
| 4  | `model/recurrent_q.py` + `soft_update.py` — GRU Q-net + Polyak | ✅ |
| 5  | `model/vdn_mixer.py` — additive sum identity | ✅ |
| 6  | `model/qmix_mixer.py` — monotonic hypernet (IGM) | ✅ |
| 7  | `model/olora.py` — orthonormal low-rank PEFT (Büyükakyüz 2024) | ✅ |
| 8  | `memory/centralised_buffer.py` — sequence-aware, masked | ✅ |
| 9  | `noise/{epsilon_greedy, schedule}.py` | ✅ |
| 10 | `services/qmix_update.py` — headline CTDE math | ✅ |
| 11 | `services/{vdn_update, iql_update}.py` — baseline alternatives | ✅ |
| 12 | `services/marl_trainer.py` — CTDE end-to-end | ✅ |
| 13 | `services/game_runner.py` — 6 sub-games + spec § 3.5 JSON | ✅ |
| 14 | `sdk/marl_sdk.py` — high-level facade | ✅ |
| 15 | `mcp/{server_base, cop_server, thief_server}.py` + `auth/` | ✅ |
| 16 | `mcp/client.py` — adjudicator-over-MCP | ✅ |
| 17 | `gmail/{formatter, ledger, sender}.py` — 3 senders + idempotency | ✅ |
| 18 | `cli/{main, commands}.py` — 8 subcommands | ✅ |
| 19 | `interface/{board_renderer, game_gui}.py` — headless-testable | ✅ |
| 20 | `graphify/graphify.py` — auto-architecture.md | ✅ |
| 21 | `services/sweeps.py` — algo × grid × radius × seed | ✅ |
| 22 | `cloud/{local, prefect}.py` — local always; cloud opt-in | ✅ |
| 23 | `tests/integration/test_reproducibility.py` | ✅ |
| 24 | `notebooks/marl_walkthrough.py` — jupytext-compatible | ✅ |
| 25 | `docs/FAILURE_MODES.md` + `scripts/audit.py` | ✅ |
| 26 | README + v1.00 tag | ✅ |

## Quickstart

```bash
# Install (uv-managed)
uv sync

# Run the full audit (lint + tests + LOC + graphify)
uv run python scripts/audit.py

# Train + play + send (one-shot, idempotent)
uv run marl train --episodes 500 --checkpoint saved_models/cop_qmix.pt
uv run marl play-game --checkpoint saved_models/cop_qmix.pt --output report.json
uv run marl send-report --report-json report.json --dry-run

# Run an MCP server
export MARL_MCP_ALLOWED_TOKENS="my-secret-token"
uv run marl serve-cop --checkpoint saved_models/cop_qmix.pt --port 7301
```

## Documentation entry points

- [`docs/PRD.md`](docs/PRD.md) — main Product Requirements Document
- [`docs/PLAN.md`](docs/PLAN.md) — layered architecture + 10 ADRs
- [`docs/TODO.md`](docs/TODO.md) — 27-layer build plan with DoD per layer
- [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md) — honest limitations + fix-it paths
- [`docs/wiki/architecture.md`](docs/wiki/architecture.md) — auto-generated module map
- Per-mechanism PRDs: [Dec-POMDP](docs/PRD_dec_pomdp.md) · [Game rules](docs/PRD_game.md) · [CTDE+VDN+QMIX](docs/PRD_ctde.md) · [OLoRA](docs/PRD_olora.md) · [MCP servers](docs/PRD_mcp.md) · [Gmail API](docs/PRD_gmail.md) · [Partial observation](docs/PRD_partial_observation.md) · [IQL baseline](docs/PRD_iql_baseline.md)

## Architecture (one-line view)

```
yaml → ConfigManager → SDK ┬→ MarlTrainer (env + Q-nets + mixer + buffer)
                            ├→ GameRunner (6 sub-games + GameReport JSON)
                            └→ Gmail/Sender (3 strategies + idempotency ledger)

DecPomdpEnv → joint_obs → ε-greedy + GRU(Q-net) → joint_action → MoveDynamics →
              (capture / collision / barrier-place) → (joint_reward, done) → buffer.push(EpisodeSequence)
                                                                       ↘ sample → QMIX/VDN/IQL update step
                                                                                  ↘ Polyak target update

MCP (cop) ◀━━ select_action ━━ Adjudicator-over-MCP ━━ select_action ━━▶ MCP (thief)
                              [token auth + role check]
```

## Submission checklist

- ✅ `README.md` at repo root (this file)
- ✅ `docs/` with PRD + PLAN + TODO + per-mechanism PRDs + FAILURE_MODES
- ✅ Working pipeline: train → play 6 sub-games → email JSON report
- ✅ Both MCP servers (cop + thief) with token auth
- ✅ Gmail sender with idempotency ledger
- ⏳ Repo must be shared with `rmisegal@gmail.com` before submission
- ⏳ Group code (8 chars, no spaces) — TBD, filled into `configs/setup.yaml::submission.group_code` before submission

## Group + author

| Role | Name | ID | Status |
|---|---|---|---|
| A | Shaked Kozlovsky | (TBD) | Solo for now |
| B | — | — | (may add partner before submission) |

Submission acknowledges solo work in the JSON report.
