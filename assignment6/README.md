# marl-lab — Cooperative-adversarial MARL Cops-and-Robbers with cloud MCP

> **Assignment 6 of the RL Course** (L10 — Multi-Agent RL). This is the **foundation of the course final project**, graded as one unit with it.
>
> ⚠️ Placeholder README — full version lands at Layer 26 (last layer of the build).

## What this is

A complete laboratory for **Multi-Agent Reinforcement Learning** on the *Cops-and-Robbers* pursuit-evasion grid, built under the **Dec-POMDP / CTDE / VDN-QMIX** paradigm. Both agents (Cop, Thief) train locally under centralised state access, then run independently behind their own **MCP server** — first on localhost, then in the cloud — with **automated Gmail-API reporting** at the end of every 6-sub-game game.

## Status

| Layer | Status |
|---|---|
| 0 — scaffold + docs | ⏳ in progress |
| 1–26 | pending |

## Documentation entry points

- [`docs/PRD.md`](docs/PRD.md) — main Product Requirements Document
- [`docs/PLAN.md`](docs/PLAN.md) — layered architecture, 10 ADRs, pseudocode
- [`docs/TODO.md`](docs/TODO.md) — ~27-layer build plan with explicit DoD per layer
- Per-mechanism PRDs: [Dec-POMDP](docs/PRD_dec_pomdp.md) · [Game rules](docs/PRD_game.md) · [CTDE+VDN+QMIX](docs/PRD_ctde.md) · [OLoRA](docs/PRD_olora.md) · [MCP servers](docs/PRD_mcp.md) · [Gmail API](docs/PRD_gmail.md) · [Partial observation](docs/PRD_partial_observation.md) · [IQL baseline](docs/PRD_iql_baseline.md)

## Submission requirements (V3 PDF + Assignment 6 spec)

- ✅ `README.md` at repo root (this file)
- ✅ `docs/` with PRD + PLAN + TODO (mandatory minimum)
- ⏳ Repo must be shared with `rmisegal@gmail.com` before submission
- ⏳ Group code (8 chars, no spaces) — TBD, filled into `configs/setup.yaml::submission.group_code` before submission
- ⏳ Working pipeline: train → play 6 sub-games → email JSON report
- ⏳ Both MCP servers running on localhost AND deployed to cloud with token auth

## Group + author

| Role | Name | ID | Status |
|---|---|---|---|
| A | Shaked Kozlovsky | (TBD) | Solo for now |
| B | — | — | (may add partner before submission) |

Submission acknowledges solo work in the JSON report.
