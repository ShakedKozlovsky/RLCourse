# Conversation Log — Development Process

> A record of the back-and-forth between the student (Shaked) and the AI assistant (Claude Opus 4.7) during the development of this project. This document shows how decisions were made, what changed, and why.

## Session overview

| Session | Date | Focus | Key outcomes |
|---|---|---|---|
| 1 | 2026-05-23 | Full build: Layers 0–10 | Complete project from PRD to real experiments |

## Conversation flow

### Phase 1: Scope alignment

**Student:** Provided the assignment PDF (Hebrew), lecture slides PDF (31 pages), and coding guidelines PDF (39 pages). Asked to build the project following all requirements, with markdown tracking files, in the RLCourse repo, with layered commits showing progress.

**AI decision:** Proposed a plan-first approach — write PRD, PLAN, TODO first, then build code in 9 layers. Offered choices for DQN variant (vanilla / Dueling / full stack), interface (GUI / CLI / both), and approach (all-at-once vs layered). Student chose: full Dueling + Double + PER stack, both GUI and CLI, plan-first-then-layers.

**Why this mattered:** the layered approach paid off immediately in Layer 1 when an integration test caught a pipeline ordering bug (ADR-007). If we'd built everything in one pass, the bug would have been much harder to isolate.

### Phase 2: Planning documents (Layer 0)

**What was produced:**
- `docs/PRD.md` — the full RL problem formulation, dataset spec, KPIs
- `docs/PLAN.md` — layered architecture diagram, class map, network diagram, config schema
- `docs/TODO.md` — per-layer task list with Definition of Done
- 8 per-mechanism PRDs: one each for DQN, Dueling, Double DQN, PER, Reward, Env, Features, Data Pipeline

**Back-and-forth:** the AI proposed the PRD structure; the student approved. The per-mechanism PRDs were the AI's idea — each one includes theory recap, I/O spec, and acceptance criteria. This made the implementation phase much faster because each module had a clear spec to code against.

### Phase 3: Code layers (Layers 1–7)

Each layer followed the same pattern:
1. Write source modules (≤150 LOC each)
2. Write tests (unit + integration)
3. Run pytest, ruff, coverage
4. Fix any issues (bugs, ruff violations, line-count overruns)
5. Update docs/TODO.md with results
6. Commit with a descriptive message

**Notable discoveries during this phase:**

**Layer 1 (data):** The original design split raw OHLCV *then* computed features per slice. The integration test `test_pipeline_runs_and_shapes_make_sense` failed because the val slice (only 14 rows after splitting a 120-row fixture) lost all 26 warmup rows to indicator warmup, leaving fewer rows than the window size. Fix: compute features on the full series first (all indicators are causal ⇒ no leakage), then split. Recorded as **ADR-007**. The fixture was also enlarged from 120 to 400 rows to give realistic slice sizes.

**Layer 2 (environment):** The PRD's reward formula `r = ΔV/V₀ − α·trade − β·trade` double-counted friction because `Portfolio.buy/sell` already deducts friction from cash (so ΔV already reflects it). Fix: simplify to `r = ΔV/V₀`. Recorded as **ADR-008**. The round-trip cost test `test_buy_then_sell_round_trip_loses_two_friction_legs` confirmed the expected `−2·(α+β)·V₀` deficit now lives entirely inside ΔV.

**Layer 3 (model):** The initial backward-pass test used lr=1e-2, which overshot on the first step (loss *increased*). Fix: use lr=1e-3 with 50 optimization steps, assert loss_final < loss_0. A one-step descent test is too brittle on random targets.

**Layer 7 (GUI):** `tabs.py` grew to 154 LOC (over the 150-line limit). Fix: split into 4 separate files (`data_tab.py`, `train_tab.py`, `backtest_tab.py`, `predict_tab.py`) plus a shared `_checkpoint_picker.py`. This brought every file back under 50 LOC and improved encapsulation.

### Phase 4: Experiments (Layers 8–10)

**Student:** Said "do everything" — meaning run the actual experiments on real data, not just the synthetic test fixtures.

**What was done:**
1. Fetched AAPL + SPY from Yahoo Finance (756 bars each)
2. Reduced training to 30 episodes (from 200) to fit in one session
3. Ran all 4 experiments × 2 conditions = 8 training runs (~15 min total)
4. Discovered that the SDK overwrote each condition's backtest output because all conditions used `name="test_backtest"`. Patched `sdk.backtest()` to accept a `report_name` keyword; patched `ExperimentService._compare` to pass `report_name=f"{exp}__{cond}"`. Re-ran the 8 backtests (fast — checkpoints already exist).
5. Generated 9 plots + 5 GUI screenshots, embedded in README

**Key surprise:** PER performed *worse* than uniform replay on the test set (−22% vs −0.2%). This contradicts the naive expectation from the Atari benchmarks but is consistent with the observation that PER amplifies whatever signal the network finds — including noise in financial data.

### Phase 5: Restructure and push

**Student:** Asked to move `assignment2/` out of the `assignment1/` nesting to make the repo layout cleaner. Also asked to push.

**What was done:** Moved `.git/` up one level from `RL_Course/assignment1/` to `RL_Course/`. Moved `assignment1/` (inner) and `assignment2/` to be siblings. Git saw zero changes because the relative paths from `.git` to the working files were preserved — a "free" restructure. Updated root README, committed, and pushed.

## Ideas that were considered but not implemented

| Idea | Why not |
|---|---|
| LSTM/Transformer encoder instead of Conv1D | Rejected in ADR-002: adds training instability for a 30-step window that already has temporal indicators (MACD, RSI). |
| Fractional position sizing (not all-in/all-out) | Rejected in ADR-003: complicates action semantics beyond the educational scope. |
| Optuna hyperparameter search | Out of scope per PRD §14: manual sweeps only. |
| Multi-asset portfolio | Explicitly out of scope; flagged as an excellence extension. |
| Window-size sensitivity sweep | Initially deferred; later implemented as an excellence differentiator (10/20/30/50 — finding: window=50 best). |

## Reflection on the AI-assisted workflow

The AI wrote ~95% of the code and ~80% of the documentation. The student's role was:
1. **Scope definition** — what to build, which choices to make.
2. **Quality control** — reviewing the plan, approving each layer, catching scope creep.
3. **Direction changes** — "do everything", "move to a different folder", "add more docstrings".
4. **Domain knowledge** — specifying that the professor wants relative grading, that the coding rules are strict, that commits showing progress are important.

The AI's main contributions were:
1. **Architectural decisions** — the SDK pattern, the layered commit strategy, the per-mechanism PRDs.
2. **Bug discovery** — the features-before-split fix (ADR-007), the reward double-counting fix (ADR-008), the PER-amplifying-noise finding.
3. **Speed** — 10 layers of code + 135 tests + full documentation in one session.
4. **Consistency** — maintaining the 150-LOC limit, ruff compliance, and 97% coverage across every commit.
