# Token Usage and Cost Estimate

> Per coding guidelines V3 §11: cost breakdown of AI token usage during this project's development.

## Model used

| Model | Provider | Context window | Pricing (per 1M tokens) |
|---|---|---|---|
| Claude Opus 4.7 | Anthropic | 1M tokens | Input: $15.00 / Output: $75.00 |

## Token usage estimate

### Session 1: full build (Layers 0–10 + experiments + reports)

| Phase | Turns | Est. input tokens | Est. output tokens | Notes |
|---|---|---|---|---|
| Initial context (3 PDFs) | 1 | ~80,000 | 0 | Assignment (12pp) + slides (31pp) + coding guidelines (39pp) |
| Layer 0: planning docs | ~5 | ~120,000 | ~15,000 | PRD, PLAN, TODO, 8 per-mechanism PRDs |
| Layer 1: data pipeline | ~15 | ~200,000 | ~12,000 | 6 source files + 8 test files + debug cycle |
| Layer 2: environment | ~10 | ~180,000 | ~8,000 | 3 source files + 3 test files + fixture fix |
| Layer 3: model + memory | ~10 | ~200,000 | ~10,000 | 4 source files + 4 test files |
| Layer 4: training service | ~10 | ~220,000 | ~10,000 | 4 source files + 3 test files + integration smoke |
| Layer 5: backtest + inference | ~8 | ~200,000 | ~7,000 | 3 source files + 3 test files |
| Layer 6: SDK + CLI | ~8 | ~220,000 | ~8,000 | 2 source files + 2 test files |
| Layer 7: GUI | ~12 | ~250,000 | ~12,000 | 10 source files + 1 test file + split refactor |
| Layer 8: experiments | ~6 | ~200,000 | ~6,000 | 1 source file + 1 test file |
| Layer 9: README + diagrams | ~5 | ~180,000 | ~10,000 | README.md, 2 Mermaid files, notebook |
| Layer 10: real experiments | ~15 | ~250,000 | ~8,000 | Scripts, plots, screenshots, patches |
| Quality pass (docstrings, reports) | ~10 | ~200,000 | ~15,000 | This document, research report, etc. |
| **Total** | **~115** | **~2,500,000** | **~121,000** | |

### Sub-agents (docstring additions)

| Agent | Model | Est. tokens |
|---|---|---|
| Docstrings agent 1 (data/env/model/memory) | Claude Sonnet 4.6 | ~30,000 total |
| Docstrings agent 2 (services/sdk/interface) | Claude Sonnet 4.6 | ~30,000 total |

## Cost calculation

| Model | Input tokens | Output tokens | Input cost | Output cost | Total |
|---|---|---|---|---|---|
| Claude Opus 4.7 | 2,500,000 | 121,000 | $37.50 | $9.08 | **$46.58** |
| Claude Sonnet 4.6 (sub-agents) | 50,000 | 10,000 | $0.15 | $0.30 | **$0.45** |
| **Grand total** | **2,560,000** | **131,000** | **$37.65** | **$9.38** | **$47.03** |

## Notes

- These are *estimates* based on the conversation length, the number of tool calls, and typical Claude Code token consumption patterns. Exact figures would require access to the Anthropic API usage dashboard.
- The cached-context ratio is likely high (~70%) because each successive layer re-reads the growing codebase — Anthropic's prompt caching reduces the effective input cost significantly below the headline $37.50 figure.
- **Effective cost per line of production code:** ~4,900 lines of source + tests → **$0.0096 per line** (~1 cent per line).
- **Effective cost per test:** 135 tests → **$0.35 per test**.
- No external API costs (yfinance is free; no paid data providers used).

## Optimization strategies employed

1. **Prompt caching:** Claude Code automatically caches prior conversation context. Keeping sessions under the 5-minute TTL reduced re-reads.
2. **Layered commits:** each layer's prompt inherits only the relevant context, not the full history of unrelated layers.
3. **Sub-agents for mechanical tasks:** docstring additions were delegated to cheaper Sonnet sub-agents instead of running on Opus.
4. **Synthetic test data:** all 135 tests run offline on a 400-day fixture, avoiding repeated yfinance API calls during development.
