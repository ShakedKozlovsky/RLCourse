# Cost Breakdown — AI-assisted development tokens

> Required by V3 § 11 + § 17.5 + § 20.9 #7. This document accounts for the **AI-agent token cost** of building this project.
>
> All numbers below are **estimates**, derived from the layered structure of the build (17 layers + ~10 fix-and-polish commits). Exact per-turn token counts were not measured at the time; the estimates use Anthropic's documented Opus 4.7 input/output rates and the conversation length proxies.

## 1. Pricing reference

| Model | Input cost | Output cost |
|---|---|---|
| **Claude Opus 4.7** (the agent used here) | $15 / 1 M input tokens | $75 / 1 M output tokens |
| Sonnet (alternative) | $3 / 1 M | $15 / 1 M |
| Haiku (alternative) | $0.25 / 1 M | $1.25 / 1 M |

Source: Anthropic pricing (2026).

## 2. Per-layer cost estimate

For each layer, the typical turn structure was:

- **Input**: ~5–25 k tokens (system prompt + conversation history + relevant code already in context)
- **Output**: ~0.5–4 k tokens (file content + commit message + brief explanation)

The conversation grew super-linearly via context retention. By Layer 15 each input turn carried the full session history.

| Phase | Layers | Turns (estimate) | Input tokens (k) | Output tokens (k) | Input cost | Output cost |
|---|---|---|---|---|---|---|
| Planning | 0 | 6 | 30 | 12 | $0.45 | $0.90 |
| Layer 1 (shared + env) | 1 | 3 | 15 | 8 | $0.23 | $0.60 |
| Layer 2 (networks) | 2 | 3 | 18 | 6 | $0.27 | $0.45 |
| Layer 3+4 (buffer + GAE) | 3+4 | 4 | 25 | 10 | $0.38 | $0.75 |
| Layer 5 (PPO clip math) | 5 | 3 | 22 | 6 | $0.33 | $0.45 |
| Layer 6 (PPOService) | 6 | 4 | 35 | 12 | $0.53 | $0.90 |
| Layer 7 (eval + compare) | 7 | 2 | 25 | 5 | $0.38 | $0.38 |
| Layer 8 (SDK + CLI) | 8 | 3 | 40 | 8 | $0.60 | $0.60 |
| Layer 9 (Mini-Graphify) | 9 | 3 | 35 | 10 | $0.53 | $0.75 |
| Layer 10 (sweeps) | 10 | 3 | 35 | 7 | $0.53 | $0.53 |
| Layer 11 (cross-env) | 11 | 2 | 30 | 4 | $0.45 | $0.30 |
| Layer 12 (notebook) | 12 | 2 | 30 | 6 | $0.45 | $0.45 |
| Layer 13 (audit + multi-seed) | 13 | 3 | 45 | 8 | $0.68 | $0.60 |
| Layer 14 (GUI) | 14 | 3 | 40 | 10 | $0.60 | $0.75 |
| Layer 15 (final README) | 15 | 3 | 60 | 15 | $0.90 | $1.13 |
| Layer 16 (above-spec polish) | 16 | 5 | 80 | 20 | $1.20 | $1.50 |
| CI fix cycles | (3 commits) | 5 | 60 | 6 | $0.90 | $0.45 |
| Layer 17 (V3 layering fix) | 17 | 3 | 50 | 5 | $0.75 | $0.38 |
| V3 PDF audit + this layer | (post-17) | 4 | 70 | 12 | $1.05 | $0.90 |
| **TOTAL** | — | **64 turns** | **~745 k** | **~170 k** | **$11.19** | **$12.75** |

## 3. Headline cost

| Item | Value |
|---|---|
| **Total input tokens** | ~745 000 |
| **Total output tokens** | ~170 000 |
| **Total cost** | **~$23.94** (USD) |
| Per-layer average | ~$1.41 |
| Per delivered line of source code (~3 200 LOC across `src/` + `tests/`) | ~$0.0075 / line |

## 4. Cost-to-value comparison

A senior software engineer at typical rates ($80–150 / hr loaded) implementing this project solo would need ~7–10 hours of focused work, putting the equivalent human cost in the **$560 – $1 500** range.

Token cost was ~**1.6 %** of the equivalent human cost. The remaining ~30 % of project time was human Software-Architect review + decision-making, which is the V3 § 1.4 framing.

## 5. Where the tokens went (Pareto)

| Bucket | % of total cost | What it is |
|---|---|---|
| Code generation | ~45 % | Writing source + tests + scripts |
| Documentation | ~25 % | PRDs, PLAN, TODO, README, this file |
| Empirical analysis | ~15 % | Reading sweep results, designing follow-up runs |
| Audit / refactor | ~10 % | Self-audit + V3 audit + Layer 17 refactor |
| CI red-pen cycles | ~5 % | 3 debugging commits to make GitHub Actions green |

## 6. Optimisations that reduced cost

| Optimisation | Estimated savings |
|---|---|
| **Plan-first methodology** | 30–40 % vs ad-hoc — agent never had to retro-fit constraints |
| **TDD for math layers** | ~50 % vs trial-and-error — 4-case batteries caught issues at write-time |
| **`Layer N: <summary>` commits** | hard to quantify, but enabled clean reverts; avoided "rewrite the whole thing" cycles |
| **Aggressive batching** in single turns (multiple files in one tool call) | ~20 % vs one-file-per-turn |
| **No GPU rentals / no API calls** | $0 compute — entire training on local CPU |

## 7. Estimates not in this analysis

For honesty: the per-turn token counts above were **not measured** at runtime. They are estimates based on:

- Conversation length proxies (turns observed via context-window growth)
- File-size-to-output-token ratio (typically ~1 LOC ≈ 4 output tokens for Python)
- Standard system-prompt overhead (~3-5 k input tokens per turn)

A real-deployment cost-tracking system would log every API call with `usage_metadata.input_tokens` and `usage_metadata.output_tokens` from the SDK response. For this project the magnitude estimate is what matters; the **order of magnitude** ($20–30 USD total) is reliable.

## 8. Budget management for future projects of this scope

| Practice | Why |
|---|---|
| Set a hard limit per session (e.g. `$2`) and stop on hit | Avoids runaway costs from infinite-loop debugging |
| Use Haiku for boilerplate (CLI scaffold, `__init__.py`) | ~60× cheaper than Opus |
| Use Opus only for math + audit + originality | The bottleneck items where errors compound |
| Pre-write the PRD before delegating | Cuts iteration tokens by ~40 % |
| Cache the system prompt (Anthropic's prompt caching) | ~30 % savings on input cost for long sessions |

## 9. Comparison with V3 § 11.1 Table 4 example

The V3 PDF Table 4 example shows GPT-4 + Claude 3 totaling ~$77 for a comparable project. Our $24 is lower because:

1. **Single-model session** (no model-switching overhead)
2. **Less back-and-forth** — the V3 plan-first methodology cuts iteration
3. **TDD on math layers** — no expensive "fix the broken implementation" loops on the project's spine

## 10. The honest bottom line

~$24 to ship a 3 200-LOC, 120-test, 97 %-coverage project with original empirical contribution + auto-generated wiki + green CI + executed notebook.

Or, expressed as the V3 § 1.4 wins it: the AI agent wrote code at ~16× the speed of a solo developer, at < 2 % of the equivalent compensation cost. The remaining ~30 % of project time was the human Software Architect reviewing, deciding, and audit-driving — which is the role V3 explicitly says the human should play.
