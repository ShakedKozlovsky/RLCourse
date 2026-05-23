# TODO — Layered Implementation Plan

> Each layer corresponds to one or more commits. **Definition of Done (DoD)** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md).

---

## Layer 0 — Project scaffold + planning docs

Commit: `Layer 0: project scaffold + planning docs`

- [x] Create directory tree under `assignment2/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [~] `docs/TODO.md` (this file)
- [ ] Per-mechanism PRDs: `PRD_dqn.md`, `PRD_dueling.md`, `PRD_double_dqn.md`, `PRD_per.md`, `PRD_reward.md`, `PRD_env.md`, `PRD_features.md`, `PRD_data_pipeline.md`
- [ ] `README.md` (placeholder pointing to docs)
- [ ] `pyproject.toml` + `uv.lock`
- [ ] `.gitignore`, `.env-example`
- [ ] `configs/setup.json`, `configs/rate_limits.json`
- [ ] All `__init__.py` files in `src/dqn_trader/` packages
- [ ] `src/dqn_trader/shared/version.py` with `__version__ = "1.00"`

**DoD:** repo can be cloned, `uv sync` installs deps, `python -c "import dqn_trader"` works, no code yet beyond shared/version. Docs explain the full plan.

---

## Layer 1 — Data layer ✅

Commit: `Layer 1: data pipeline (yfinance, features, splits, windows) + tests`

- [x] `shared/config.py` — JSON loader, dotted access, version check
- [x] `shared/logger.py` — stdlib logger factory
- [x] `shared/seed.py` — `set_global_seed` (NumPy + Python + Torch)
- [x] `shared/gatekeeper.py` — token-bucket rate-limiter + retry/backoff
- [x] `shared/types.py` — `Action`, `Transition`, `StepInfo`, `SliceData`
- [x] `data/yfinance_client.py` — fetch + parquet cache + CSV fallback + MultiIndex collapse
- [x] `data/feature_engineer.py` — the 8 market channels (log_return, RSI, MACD×3, BB%, VWAP-dist, volume-z)
- [x] `data/splitter.py` — chronological 70/15/15
- [x] `data/scaler.py` — fit-once contract + save/load
- [x] `data/window_builder.py` — `(N, 30, 8)` stride-1 windows
- [x] `services/data_service.py` — features-first-then-split pipeline (causal indicators ⇒ no leakage)
- [x] Unit tests for every module (32 unit tests, 2 integration tests)
- [x] **39/39 tests pass**, **94% statement+branch coverage**, **ruff clean**, **largest file 80 LOC**

**Note** — pipeline ordering corrected during integration testing: features are computed on the full raw series first (all indicators are causal, so no leakage) *then* split, scaled on train only, and windowed. Documented as ADR-007 in PLAN.md.

**DoD met:** `uv run pytest tests/ -q` → 39 passed. Coverage of `data/` ≥ 90% (all modules ≥ 90%, three at 100%). Pipeline runs end-to-end on a synthetic 400-day OHLCV through `DataService.run()`.

---

## Layer 2 — Environment ✅

Commit: `Layer 2: TradingEnv + RewardFunction + Portfolio + tests`

- [x] `environment/portfolio.py` — cash, shares, MTM, all-in/all-out trade execution with symmetric friction (`α + β` per leg, applied multiplicatively to cash)
- [x] `environment/reward.py` — `BaselineReward`, `RiskAdjustedReward` (rolling annualised Sharpe), `build_reward` factory, abstract `RewardFunction` base
- [x] `environment/trading_env.py` — `reset`, `step`, observation assembly (8 market + 2 portfolio channels broadcast over time), terminal logic, invalid-action no-ops with optional penalty
- [x] `shared/types.py` — already in place from Layer 1
- [x] `tests/unit/test_portfolio.py` — buy/sell/no-ops, round-trip cost, MTM, PnL, invalid construction (8 tests)
- [x] `tests/unit/test_reward.py` — baseline ΔV/V₀, first-step bonus disabled, zero-variance handling, positive-trend Sharpe bonus, factory (8 tests)
- [x] `tests/unit/test_trading_env.py` — reset shape, step advance, position channel reflects holdings, termination, invalid action no-op, round-trip, post-termination raise, channel-count guard (9 tests)
- [x] **64/64 tests pass**, **95% coverage**, **ruff clean**, largest file 99 LOC

**ADR-008 added (PLAN.md)** — friction is folded into cash on each trade (`Portfolio.buy/sell`), not applied as a separate term in the reward. This eliminates double-counting between the bookkeeping and the reward, and makes the reward formula `r = ΔV/V₀` exactly — the cost still appears in the round-trip test as the expected `−2·(α+β)·V₀` deficit, since it lives inside ΔV.

**DoD met:** environment tests green, coverage of `environment/` ≥ 96% on all modules (`portfolio.py` 100%, `reward.py` 98%, `trading_env.py` 96%). Manual flat → long → flat round-trip with no price change produces a small negative reward equal to the round-trip friction, as expected.

---

## Layer 3 — Model + memory ✅

Commit: `Layer 3: Dueling DQN + Uniform/Prioritized replay + target sync + tests`

- [x] `model/dueling_dqn.py` — Conv1D trunk (2 layers), shared dense, V/A heads, mean-centred aggregation. `dueling=False` collapses heads into a single Q head — the *vanilla DQN* baseline.
- [x] `model/dueling_dqn.py` includes `hard_update` and Polyak `soft_update(τ)` for target sync (target network is just another instance of the same nn.Module — no separate file needed).
- [x] `memory/uniform_replay.py` — circular `deque`-backed buffer; `Batch` namedtuple shared with PER for trainer-side interchangeability.
- [x] `memory/sum_tree.py` — array-backed sum-tree of length `2N − 1`; `add`/`update`/`get` all O(log N); `sample_indices` does stratified per-segment sampling.
- [x] `memory/prioritized_replay.py` — proportional PER backed by the sum tree; tracks max-priority for new-transition default; emits `IS_weights = (N·P)^(-β)` normalised to ≤ 1.
- [x] `tests/unit/test_dueling_dqn.py` — forward shapes, Dueling identity (`Q − Q.mean ≡ A − A.mean`), vanilla fallback, hard/soft sync, backward-pass convergence over 50 Adam steps (9 tests)
- [x] `tests/unit/test_uniform_replay.py` — empty/length/capacity wrap, sample shapes, IS-weights=1.0, no-op `update_priorities` (7 tests)
- [x] `tests/unit/test_sum_tree.py` — sum invariants, propagation, prefix-sum sampling, frequency proportionality, error paths (7 tests)
- [x] `tests/unit/test_prioritized_replay.py` — IS-weight normalisation, priority boost shifts sampling, invalid β/lengths (8 tests)
- [x] **95/95 tests pass**, **96% coverage**, **ruff clean**, largest file 99 LOC

**Note** — the `Batch` namedtuple is intentionally shared between `UniformReplay` and `PrioritizedReplay` so `TrainingService` is buffer-agnostic. Uniform replay returns `is_weights == 1.0` and is a no-op on `update_priorities`, by design.

**DoD met:** all model + memory modules ≥ 96% coverage (most at 100%). The Dueling identity is verified structurally (not just numerically). Backward-pass convergence test exercises the full forward + loss + backprop path.

---

## Layer 4 — Training service ✅

Commit: `Layer 4: TrainingService (Double DQN, ε-greedy, target sync, checkpointing) + tests`

- [x] `services/epsilon_schedule.py` — `LinearSchedule` used as both `EpsilonSchedule` (decreasing) and `BetaSchedule` (increasing); aliases keep call-sites self-documenting.
- [x] `services/run_directory.py` — per-run timestamped folder with `config_snapshot.json`, `git_hash.txt`, `checkpoints/`, `plots/`, `metrics.csv`.
- [x] `services/dqn_agent.py` — owns online + target Dueling nets, Adam, Huber loss, PER-aware loss-with-IS-weights. Double-DQN target: action selected by *online*, evaluated by *target*. Periodic hard target sync. `save/load` with `weights_only=True`.
- [x] `services/training_service.py` — orchestrates: builds env from `PipelineOutput`, picks replay buffer per config (PER or Uniform), runs N episodes, evaluates greedily on val slice, saves *best* (by `val_return`) and *last* checkpoints, writes `metrics.csv` per episode.
- [x] `tests/unit/test_epsilon_schedule.py` — clamping, interpolation, both directions (4 tests)
- [x] `tests/unit/test_dqn_agent.py` — action sampling, ε=1 uniformity, optimize returns None when empty, target sync at interval, Double-DQN path, save/load round-trip, dueling parameter count (9 tests)
- [x] `tests/integration/test_training_smoke.py` — full PER + Uniform smoke runs on synthetic OHLCV → checkpoints + metrics.csv present (2 tests)
- [x] **110/110 tests pass**, **97% coverage**, **ruff clean**, largest file 144 LOC.

**Implementation notes**
- `Batch` namedtuple is shared between Uniform and Prioritized replay so the agent is buffer-agnostic. `update_priorities` is a no-op for uniform — the agent calls it unconditionally.
- The val evaluation uses `epsilon=0.0` (greedy) and a fresh RNG to keep the metric deterministic.

**DoD met:** smoke runs complete in seconds. Both PER and Uniform paths exercised by tests. Checkpoints, metrics CSV, and config snapshot land in `results/run_<ts>/` as designed.

---

## Layer 5 — Backtest + inference

Commit: `Layer 5: BacktestService + InferenceService + risk metrics + tests`

- [ ] `services/backtest_service.py` — runs greedy policy on test slice, computes equity curve, total return, Sharpe, Max DD, Win Rate, trade count
- [ ] `services/inference_service.py` — single state → action + Q-values + confidence
- [ ] Plot helpers: equity vs Buy-and-Hold, drawdown bands
- [ ] `tests/unit/test_backtest_metrics.py` — Sharpe on synthetic constant-return series; Max DD on a known dip
- [ ] `tests/unit/test_inference_service.py` — output schema (action ∈ {0,1,2}, q_values shape (3,), confidence ∈ [0,1])

**DoD:** running backtest on a trained checkpoint produces JSON + PNG in `results/backtest/`. README has "How to evaluate".

---

## Layer 6 — SDK + CLI

Commit: `Layer 6: TradingSDK facade + CLI (click) + tests`

- [ ] `sdk/sdk.py` — `TradingSDK` with `prepare_data`, `train`, `backtest`, `predict`, `run_experiment`
- [ ] `interface/cli/main.py` — Click-based CLI: `dqn-trader data prepare`, `train`, `backtest`, `predict`, `experiment <name>`
- [ ] `interface/__main__.py` so `python -m dqn_trader.interface.cli` works
- [ ] `tests/integration/test_sdk.py` — end-to-end smoke: prepare → train (1 ep) → backtest → predict, all in tmp dirs

**DoD:** `uv run python -m dqn_trader.interface.cli --help` prints the command tree. Smoke test green.

---

## Layer 7 — GUI

Commit: `Layer 7: PyQt6 GUI on top of SDK`

- [ ] `interface/gui/main_window.py` — tabs for Data, Train, Backtest, Predict
- [ ] `interface/gui/plot_widgets.py` — embedded matplotlib for equity, loss, reward, epsilon
- [ ] `interface/gui/components.py` — config form fields, ticker selector, date pickers
- [ ] Threaded training so the UI stays responsive (long-running task in QThread)
- [ ] Screenshots saved into `assets/gui/` for each tab
- [ ] `tests/unit/test_gui_smoke.py` — instantiation under `QApplication([])` headless

**DoD:** GUI launches via `uv run python -m dqn_trader.interface.gui`. All four tabs render. Screenshots in `assets/gui/`. (Skipped if no display: documented in README.)

---

## Layer 8 — Experiments

Commit: `Layer 8: comparative experiments + results notebook`

- [ ] `services/experiment_service.py` — `run_dqn_vs_dueling`, `run_uniform_vs_per`, `run_reward_variants`, `run_cross_ticker`
- [ ] Configurations baked into `configs/experiments/*.json`
- [ ] `notebooks/01_results_analysis.ipynb` — loads `results/`, produces comparison plots, summary table
- [ ] `results/experiments_summary.md` — written by `experiment_service`, machine-friendly + human-friendly

**DoD:** all four experiments run on AAPL + one cross-ticker (SPY or NVDA). Markdown summary table committed. Notebook produces all plots used in README.

---

## Layer 9 — Final README and writeup

Commit: `Layer 9: full README with screenshots, answers to 12 reflection questions, plots`

- [ ] Replace placeholder `README.md` with full learning document
- [ ] Include: install, run, screenshots, architecture diagram, class diagram, Bellman + Dueling formulas, slide cross-references, reward table, experiment results, **answers to all 12 reflection questions** from the assignment §13
- [ ] `assets/` populated with architecture diagram source (Mermaid in `docs/`) and rendered PNGs
- [ ] Sources/citations section

**DoD:** every checklist item in the assignment spec maps to a section in README. Self-grading checklist signed off.

---

## Cross-cutting tasks (in parallel with all layers)

- [ ] `ruff check` returns 0 across the whole repo before each commit
- [ ] Each commit message follows: `Layer N: <short summary>` + bullet body
- [ ] No magic numbers introduced in code (everything in `configs/`)
- [ ] No file exceeds 150 LOC (excluding blanks/comments)
- [ ] `uv.lock` regenerated when deps change
- [ ] After each layer: update this `TODO.md` to mark items `[x]` and bump version where applicable

## Excellence extensions (only if base is fully green)

- [ ] Window-size sensitivity sweep (10, 20, 30, 50) and plot
- [ ] Reward-hacking sanity probe: log action distribution per episode, flag if Hold > 95%
- [ ] Add an attention-based explainability module that highlights which days in the 30-window mattered most for the chosen action
- [ ] Multi-asset portfolio extension (single-asset → 3-asset basket) — explicitly out of base scope
