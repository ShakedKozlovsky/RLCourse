# PLAN — Architecture & Implementation Plan

> Reference: [`docs/PRD.md`](PRD.md), course slides, code-guidelines V3.

## 1. Layered architecture

The architecture is strictly layered. Inner layers must never import from outer layers. The SDK is the single entry point for all consumers (CLI, GUI, future tests/notebooks).

```
┌──────────────────────────────────────────────────────────┐
│  Interface layer  (src/dqn_trader/interface/)            │
│    cli/             gui/                                 │
└────────────────────────┬─────────────────────────────────┘
                         │  (only allowed entry point)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  SDK  (src/dqn_trader/sdk/sdk.py)                        │
│    TradingSDK — facade over all services                 │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Services  (src/dqn_trader/services/)                    │
│    DataService · TrainingService · BacktestService       │
│    InferenceService · ExperimentService                  │
└──────┬────────────┬───────────┬────────────┬─────────────┘
       │            │           │            │
       ▼            ▼           ▼            ▼
┌─────────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐
│ environment/│ │ model/   │ │memory/ │ │ data/      │
│ TradingEnv  │ │ DuelingDQN│ │PER     │ │ YFClient   │
│ Reward      │ │ TargetNet │ │Uniform │ │ Features   │
│ Portfolio   │ │ Heads     │ │        │ │ Splitter   │
└─────────────┘ └──────────┘ └────────┘ └────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  shared/  (cross-cutting: logger, config, gatekeeper)    │
└──────────────────────────────────────────────────────────┘
```

### Dependency rule

```
interface → sdk → services → {environment, model, memory, data} → shared
```

Forward arrows only. `model/` does not import from `services/`. `environment/` does not import from `interface/`. Any violation is a CI-style failure.

## 2. Package map

```
src/dqn_trader/
├── __init__.py
├── shared/
│   ├── __init__.py
│   ├── version.py             # __version__ = "1.00"
│   ├── config.py              # ConfigManager — JSON loader + validation
│   ├── logger.py              # get_logger(name) wrapper
│   ├── seed.py                # set_global_seed(seed)
│   ├── gatekeeper.py          # API rate-limiter for yfinance
│   └── types.py               # TypedDicts / dataclasses shared across layers
├── data/
│   ├── __init__.py
│   ├── yfinance_client.py     # raw fetch + cache + CSV fallback
│   ├── feature_engineer.py    # the 10-channel pipeline
│   ├── window_builder.py      # rolling 30-bar windows
│   ├── splitter.py            # chronological 70/15/15
│   └── scaler.py              # fit on train, apply to all
├── environment/
│   ├── __init__.py
│   ├── portfolio.py           # cash, position, MTM, trade execution
│   ├── reward.py              # baseline + risk-adjusted variants
│   └── trading_env.py         # Gymnasium-style Env
├── model/
│   ├── __init__.py
│   ├── dueling_dqn.py         # Conv1D feature extractor + V/A heads
│   └── target_network.py      # soft / hard sync helper
├── memory/
│   ├── __init__.py
│   ├── uniform_replay.py      # baseline buffer
│   └── prioritized_replay.py  # proportional PER with IS weights
├── services/
│   ├── __init__.py
│   ├── data_service.py        # orchestrates data pipeline end-to-end
│   ├── training_service.py    # train loop, checkpointing, eval hooks
│   ├── backtest_service.py    # equity curve + risk metrics
│   ├── inference_service.py   # single-state Q-value + action
│   └── experiment_service.py  # runs the comparative experiments
├── sdk/
│   ├── __init__.py
│   └── sdk.py                 # TradingSDK facade
└── interface/
    ├── __init__.py
    ├── cli/
    │   ├── __init__.py
    │   └── main.py            # `python -m dqn_trader` → CLI
    └── gui/
        ├── __init__.py
        ├── main_window.py     # PyQt6 main window
        ├── components.py      # reusable widgets
        └── plot_widgets.py    # matplotlib embed for equity / loss
```

Every file ≤ 150 lines (excluding blank lines and comments). When a module grows past this, split by responsibility (see §7).

## 3. Class diagram (textual UML)

```
                          ┌──────────────┐
                          │ TradingSDK   │
                          │--------------│
                          │ prepare_data │
                          │ train        │
                          │ backtest     │
                          │ predict      │
                          │ run_experiment│
                          └──┬───────────┘
                             │ uses
   ┌─────────────────────────┼──────────────────────────────┐
   ▼                         ▼                              ▼
┌──────────────┐    ┌──────────────────┐         ┌────────────────┐
│ DataService  │    │ TrainingService  │         │ BacktestService│
│--------------│    │------------------│         │----------------│
│ load(ticker) │    │ fit(env, agent)  │         │ evaluate(agent)│
│ split()      │    │ _optimize_step() │         │ metrics()      │
│ features()   │    │ _epsilon_action()│         │ equity_curve() │
└──────┬───────┘    └─────────┬────────┘         └────────┬───────┘
       │                      │                           │
       ▼                      ▼                           ▼
┌──────────────┐    ┌──────────────────┐         ┌────────────────┐
│ YFClient     │    │ DQNAgent         │         │ TradingEnv     │
│ FeatureEng.. │    │ ┌────────────┐   │         │ Portfolio      │
│ Splitter     │    │ │ DuelingDQN │   │         │ RewardFunction │
│ Scaler       │    │ │ TargetNet  │   │         └────────────────┘
└──────────────┘    │ │ Replay/PER │   │
                    │ └────────────┘   │
                    └──────────────────┘
```

`DQNAgent` is a thin coordinator class living in `services/training_service.py` — it owns the online net, target net, optimizer, and replay buffer. Model code stays in `model/`.

## 4. Data flow

```
yfinance ──▶ YFClient(cache, fallback) ──▶ raw OHLCV
   ▼
FeatureEngineer ──▶ 10 channels per day
   ▼
Splitter (70/15/15 chronological) ──▶ train_df, val_df, test_df
   ▼
Scaler.fit(train_df).transform(all)
   ▼
WindowBuilder ──▶ (N, 30, 10) tensors
   ▼
TradingEnv.reset() ──▶ state s_t = window stacked with portfolio channels
   ▼
agent.act(s_t) ──▶ a_t ∈ {0,1,2}
   ▼
TradingEnv.step(a_t) ──▶ s_{t+1}, r_t, done
   ▼
ReplayBuffer.add(s_t, a_t, r_t, s_{t+1}, done)
   ▼
sample mini-batch ──▶ DuelingDQN forward ──▶ y = Bellman target (Double DQN)
   ▼
loss = Huber(Q(s,a), y) · IS_weights ──▶ backprop ──▶ θ update
   ▼
every N steps: θ_target ← θ_online
```

## 5. State tensor specification

Final state passed to the network: shape **(30, 10)** float32, channels in the fixed order documented in PRD §5.

- Channels 1–8 are market features computed by `FeatureEngineer` and z-scored by `Scaler` (mean and std learned from train slice).
- Channels 9–10 are portfolio channels written by `TradingEnv` at every `step`; they are *not* scaled by the train-fitted scaler because they live on a fixed bounded scale.

The network treats the 30 time steps as the sequence dimension and the 10 features as the channel dimension of a `Conv1d` layer.

## 6. Network architecture

```
Input:  (B, 30, 10)
  │  permute → (B, 10, 30)
  ▼
Conv1d(in=10, out=32, kernel=5, padding=2) → ReLU
  ▼
Conv1d(in=32, out=64, kernel=3, padding=1) → ReLU
  ▼
Flatten → (B, 64 * 30)
  ▼
Linear(64*30, 128) → ReLU                ┐
   ├── Value head:    Linear(128, 1)     │   Q(s,a) =
   └── Advantage head: Linear(128, 3)    ┘     V(s) + A(s,a) − mean_a A(s,a)
```

`Q(s, ·)` shape `(B, 3)`. Loss is `HuberLoss` on TD error, weighted by PER importance-sampling weights when PER is enabled.

## 7. Coding standards

- ≤ 150 lines per file (excluding blanks/comments). Split strategies: helper extraction, mixin extraction, constants extraction, model definitions to dedicated files.
- No magic numbers — every tunable lives in `configs/setup.json` or a typed `Enum` / dataclass in `shared/types.py`.
- `ruff check` with the V3 ruleset must be 0 errors. `ruff format` for formatting.
- Every public function has a one-line docstring (or short multi-line) explaining **why**, not what.
- All randomness threads through `shared/seed.py`. PyTorch, NumPy, Python's `random`, and the env all share the seed from config.
- All file paths in the code are *project-relative*, resolved via `shared/config.py`'s `PROJECT_ROOT`.

## 8. Testing strategy

- Unit tests mirror `src/` structure under `tests/unit/`. Integration tests under `tests/integration/`.
- Coverage gate: ≥85% statements, ≥80% branches.
- TDD pairs documented in README:
  1. `RewardFunction` — write the test first (the long+sell path must produce reward = ΔV − cost), then implement.
  2. `PrioritizedReplay.sample` — write the test that asserts sampling probability ∝ priority^α, then implement.
- Network forward test: a `(2, 30, 10)` tensor returns `(2, 3)` and is finite.
- Training step test: one `optimize()` call decreases (or at least does not increase) loss on a tiny synthetic replay.
- Backtest service test: deterministic policy (always-Hold) on synthetic flat data yields zero return and zero trades.

## 9. Configuration files

`configs/setup.json` (versioned, value `version: "1.00"`):

```jsonc
{
  "version": "1.00",
  "seed": 208904839,
  "data": {
    "ticker": "AAPL",
    "start": "2020-01-01",
    "end":   "2023-01-01",
    "interval": "1d",
    "train_pct": 0.70,
    "val_pct":   0.15,
    "window_size": 30,
    "features": 10,
    "comparative_ticker": "SPY"
  },
  "env": {
    "initial_capital": 10000,
    "transaction_cost_alpha": 0.001,
    "slippage_beta": 0.001,
    "invalid_action_penalty": 0.0,
    "reward_variant": "baseline"      // or "risk_adjusted"
  },
  "agent": {
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end":   0.05,
    "epsilon_decay_steps": 50000,
    "lr": 0.0005,
    "batch_size": 64,
    "replay_capacity": 100000,
    "min_replay_size": 1000,
    "target_sync_every": 1000,
    "huber_delta": 1.0,
    "grad_clip": 10.0,
    "dueling": true,
    "double_dqn": true
  },
  "per": {
    "enabled": true,
    "alpha": 0.6,
    "beta_start": 0.4,
    "beta_end": 1.0,
    "beta_anneal_steps": 100000,
    "epsilon": 1e-6
  },
  "training": {
    "episodes": 200,
    "max_steps_per_episode": null,    // null ⇒ use full slice
    "eval_every_episodes": 10,
    "checkpoint_dir": "results/checkpoints",
    "best_metric": "val_sharpe"
  },
  "backtest": {
    "deterministic_policy": "greedy",
    "report_dir": "results/backtest"
  }
}
```

`configs/rate_limits.json` (versioned):

```json
{
  "version": "1.00",
  "services": {
    "yfinance": {
      "requests_per_minute": 30,
      "requests_per_hour": 500,
      "concurrent_max": 1,
      "retry_after_seconds": 30,
      "max_retries": 3
    }
  }
}
```

## 10. ADRs — key architectural decisions

- **ADR-001: Dueling + Double + PER is the primary stack, with a vanilla DQN baseline.** Rationale: the assignment grades excellence, comparative experiments are mandatory, and the lecture emphasises Dueling. Trade-off: more code than vanilla — mitigated by clean abstraction (`ReplayBuffer` interface implemented by both `UniformReplay` and `PrioritizedReplay`).

- **ADR-002: Conv1D feature extractor, not LSTM/Transformer.** Rationale: the lecture shows Conv1D over the 30-day axis as the canonical Dueling-DQN encoder for this setup; LSTM adds training instability and is not required. Trade-off: less temporal expressivity — acceptable because RSI/MACD already aggregate temporal information.

- **ADR-003: All-in / all-out positioning.** Rationale: matches the reference project, keeps action semantics simple and observable, avoids the can-of-worms of fractional sizing for an educational project. Trade-off: less realistic — explicitly called out as out-of-scope for the baseline.

- **ADR-004: 70/15/15 chronological split, scaler fit only on train.** Rationale: time-series leakage is the most common rookie mistake in financial ML — non-negotiable. Trade-off: smaller train slice than 80/10/10 — acceptable because the dataset is only ~750 daily bars.

- **ADR-005: yfinance via a rate-limited gatekeeper + parquet cache + CSV fallback.** Rationale: yfinance returns 429s and occasionally changes column shape; the gatekeeper insulates the entire codebase from these. Trade-off: extra plumbing — pays for itself the first time tests run offline.

- **ADR-006: PyQt6 GUI + Click-based CLI, both as thin wrappers over the SDK.** Rationale: the GUI is required for top grade; CLI is essential for headless training and scripting. Trade-off: two interfaces to maintain — mitigated because they share a single SDK and ~10 lines of orchestration each.

- **ADR-007 (added during Layer 1): features-first-then-split, not split-first-then-features.** Rationale: every market indicator we use (log_return, RSI, MACD, Bollinger %B, VWAP distance, rolling z-score of volume) is *causal* — it depends only on past prices/volumes. Computing them on the full raw series before splitting therefore introduces no leakage, while computing them per-slice loses ~26 days of indicator warmup from val and test independently (catastrophic on short slices). Trade-off: requires careful auditing if a non-causal indicator is added later — documented as a constraint in `docs/PRD_features.md`.

- **ADR-008 (added during Layer 2): friction is deducted from cash inside `Portfolio.buy/sell`, not subtracted as a separate term in the reward.** Rationale: the original PRD formula `r = ΔV/V₀ − α·|trade|/V₀ − β·|trade|/V₀` double-counts because ΔV already reflects the cost that was paid out of cash. Folding the friction multiplicatively into the trade (`cash_after = cash_before · (1 − α − β)`) keeps a single source of truth (`V_t`) and lets the reward be the clean `ΔV/V₀`. The round-trip test still asserts the expected `−2·(α + β)·V₀` deficit — it now lives inside ΔV rather than in a separate reward term.

## 11. Reproducibility plan

- `seed` is loaded from `configs/setup.json` and applied via `shared/seed.set_global_seed` to NumPy, Python `random`, PyTorch (CPU + CUDA), and Gymnasium.
- `uv.lock` pinned via `uv lock`; CI installs via `uv sync`.
- A `results/run_<timestamp>/` directory is created per training run and contains:
  - `config_snapshot.json` (the exact config used)
  - `git_hash.txt` (the commit at run start)
  - `metrics.csv` (episode, reward, loss, epsilon, val_sharpe)
  - `checkpoints/best.pt` + `checkpoints/last.pt`
  - `plots/*.png` (equity curve, loss, reward, epsilon)

## 12. Open questions for future commits

These are intentionally left to resolve as the implementation progresses (committed to docs/TODO.md):

- Whether to add a small invalid-action penalty by default (tracked in TODO; default 0 for now).
- Whether to log the action distribution to spot reward-hacking early (likely yes — easy to add in training).
- Whether the comparative experiment should also vary `window_size` (probably yes, as an excellence extension).
