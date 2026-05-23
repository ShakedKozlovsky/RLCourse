# DQN Trading Agent — Assignment 2

> **Course:** Reinforcement Learning with Deep Learning
> **Author:** Shaked Kozlovsky (ID 208904839)
> **What this project is:** an educational, full-stack RL system that learns a *daily trading policy* (Sell / Hold / Buy) via Dueling + Double DQN with Prioritized Experience Replay on Yahoo Finance OHLCV data.
> **What this project is NOT:** a price predictor, a trading recommendation, or a production trading system. The agent learns a decision policy — not a forecast. Backtests are a teaching aid, not an investment claim.

---

## Table of contents

1. [The thinking process — how this repo was built](#1-the-thinking-process--how-this-repo-was-built)
2. [Project goal and RL framing](#2-project-goal-and-rl-framing)
3. [Mapping to the lecture slides](#3-mapping-to-the-lecture-slides)
4. [Dataset specification](#4-dataset-specification)
5. [Feature engineering — the 10 state channels](#5-feature-engineering--the-10-state-channels)
6. [Algorithm — Dueling + Double DQN + Prioritized Replay](#6-algorithm--dueling--double-dqn--prioritized-replay)
7. [Reward function](#7-reward-function)
8. [Architecture and class diagram](#8-architecture-and-class-diagram)
9. [How to install and run](#9-how-to-install-and-run)
10. [How to reproduce the experiments](#10-how-to-reproduce-the-experiments)
11. [Test suite and quality gates](#11-test-suite-and-quality-gates)
12. [Answers to the 12 reflection questions](#12-answers-to-the-12-reflection-questions)
13. [Sources](#13-sources)

---

## 1. The thinking process — how this repo was built

The professor asked to see the *thinking* before the code. The repo's git log is intentionally readable as a story:

| Commit | Layer | Why this layer is its own commit |
|---|---|---|
| `9dbb167` | **0 — plan** | PRD, PLAN, TODO, per-mechanism PRDs. Code-free. |
| `6767cbf` | **1 — data** | Only layer touching network or filesystem — isolating it makes everything else testable offline. |
| `1ee686e` | **2 — env** | The contract between trading and RL lives here. |
| `375459d` | **3 — model + memory** | Pure tensor math: Dueling DQN + Uniform/Prioritized Replay + SumTree. |
| `9b12867` | **4 — training service** | First layer where everything composes (env + agent + replay + scheduler). |
| `1fbda83` | **5 — backtest + inference** | Evaluation, separated from training so the same trained checkpoint can be backtested and queried independently. |
| `3100523` | **6 — SDK + CLI** | The single facade that GUI and CLI both consume. |
| `56c5a65` | **7 — GUI** | PyQt6 on top of the SDK, never touching services directly. |
| `1fd6679` | **8 — experiments** | Four comparative experiments via config overrides. |
| *this commit* | **9 — README & writeup** | The grader-facing document. |

Two design decisions discovered *during* the build (ADRs in [`docs/PLAN.md`](docs/PLAN.md)):

- **ADR-007** — features-first, then-split (instead of split-first, then-features). Every market indicator is causal, so computing on the full raw series introduces no leakage but recovers ~26 days of warmup per slice. The original plan had it backwards; the integration test caught it.
- **ADR-008** — friction (`α + β`) is applied multiplicatively inside `Portfolio.buy/sell` so reward = `ΔV/V₀` cleanly, instead of subtracting the friction term in the reward formula (which would double-count the cost since ΔV already reflects it).

These are documented as ADRs precisely because they were *changes of mind* during implementation, not part of the original plan.

## 2. Project goal and RL framing

### What the agent learns

A policy `π(a|s)` over discrete actions `a ∈ {Sell, Hold, Buy}`, parameterised by a Dueling Q-network `Q_θ(s, a)`. **Not** the price of the next bar — `Q_θ` is the *expected discounted return* of an action under the current policy.

### The 7-tuple of RL primitives

| Component | This project |
|---|---|
| **Agent** | `DQNAgent` (`services/dqn_agent.py`) — owns Dueling online + target nets, optimizer, replay buffer. |
| **Environment** | `TradingEnv` (`environment/trading_env.py`) — Gymnasium-style env over a SliceData. |
| **State `s_t`** | `(30, 10)` tensor: 8 pre-scaled market features × 30 days + position (broadcast) + unrealised PnL (broadcast). |
| **Action `a_t`** | `Action ∈ {SELL=0, HOLD=1, BUY=2}` (`shared/types.py`). |
| **Reward `r_t`** | `BaselineReward` (`environment/reward.py`): `(V_t − V_{t-1}) / V_0`. Optional `RiskAdjustedReward` adds γ · rolling annualised Sharpe. |
| **Episode** | One full pass through a slice (train, val, or test) in chronological order. |
| **Policy** | ε-greedy during training (`EpsilonSchedule`), greedy during evaluation/backtest. |
| **Return** | `G_t = Σ γ^k · r_{t+k}` with `γ = 0.99` by default. |

## 3. Mapping to the lecture slides

| Lecture concept | Slides | Where in this project |
|---|---|---|
| Q-Table → Function Approximation | 3–6, 14–15 | `docs/PRD_dqn.md` + this README §6; `model/dueling_dqn.py`. |
| RL framing (Agent/Env/State/Action/Reward/Episode/Policy/Return) | 7–10 | This README §2; `environment/trading_env.py::reset/step`. |
| Data → state tensor | 11–13 | `data/feature_engineer.py`, `data/window_builder.py`; pipeline shape `(N, 30, 10)`. |
| DQN and Dueling DQN | 16–21 | `model/dueling_dqn.py` with `V(s)`/`A(s,a)` heads; mean-centred aggregation. |
| Exploration & stabilization (ε-greedy, Replay, Target Net) | 22–24 | `services/epsilon_schedule.py`, `memory/{uniform_replay,prioritized_replay}.py`, `dueling_dqn.hard_update/soft_update`. |
| Full training cycle | 25 | `services/training_service.py` — reset → act → step → store → sample → optimize → sync. |
| Backtest and results analysis | 26–27 | `services/backtest_service.py` + `services/risk_metrics.py`. |
| OOP, tests, architecture | 28–29 | This README §8, [`docs/architecture.mmd`](docs/architecture.mmd), [`docs/class_diagram.mmd`](docs/class_diagram.mmd). |
| Theoretical recap | 30–31 | This section's first sentence and §12 below. |

## 4. Dataset specification

| Parameter | Mandatory value | Implementation |
|---|---|---|
| Source | Yahoo Finance via `yfinance` | `data/yfinance_client.py` |
| Primary ticker | AAPL | `configs/setup.json:data.ticker` |
| Primary date range | 2020-01-01 → 2023-01-01 | `data.start` / `data.end` |
| Interval | `1d` | `data.interval` |
| Raw columns | Open / High / Low / Close / Volume | enforced by `_missing()` in the client |
| Cache | `data/raw/{ticker}_{start}_{end}.parquet` (snappy) | `YFinanceClient.fetch` |
| CSV fallback | `data/raw/{ticker}.csv` indexed by Date | same |
| Split | 70 / 15 / 15 **chronological**, no shuffle | `data/splitter.py` |
| Window size | 30 | `data.window_size` |
| Feature count | 10 (8 market + 2 portfolio) | `data.features` |

**Comparative ticker (mandatory):** SPY by default (`data.comparative_ticker`). Same pipeline, no code changes.

**No-leakage rules (non-negotiable):**

1. All normalisation statistics are fit on **train slice only** (`data/scaler.py`).
2. Hyperparameter selection uses the **validation** slice — never the test slice.
3. Features are computed *before* splitting, only because every indicator is **causal** — see ADR-007.

## 5. Feature engineering — the 10 state channels

| # | Channel | What it tells the agent | Computed by |
|---|---|---|---|
| 1 | `log_return` | direction + magnitude of daily move | `data/feature_engineer.py` |
| 2 | `rsi_14` | overbought / oversold (Wilder's RSI) | same |
| 3 | `macd` | momentum (EMA12 − EMA26 of Close) | same |
| 4 | `macd_signal` | trend confirmation (EMA9 of MACD) | same |
| 5 | `macd_hist` | acceleration of momentum | same |
| 6 | `bb_pct` | position within Bollinger band, volatility-normalised | same |
| 7 | `vwap_dist` | mean-reversion distance from rolling VWAP-20 | same |
| 8 | `volume_norm` | participation z-scored over a rolling 60-day window | same |
| 9 | `position` | currently long? 0 or 1, broadcast across the time dim | `environment/trading_env.py` |
| 10 | `pnl_unrealised` | mark-to-market PnL on the open position, scaled by V₀ | same |

Channels 1–8 are pre-scaled by the train-fitted `ZScoreScaler`. Channels 9–10 are written by the env at step time on the fixed bounded scale `[0, 1]` / `~O(1e-2)`.

## 6. Algorithm — Dueling + Double DQN + Prioritized Replay

### Bellman target (Double DQN)

```
a*       = argmax_{a'} Q_online(s', a')      # selection comes from the *online* net
y        = r + γ · (1 − done) · Q_target(s', a*)   # evaluation comes from the *target* net
L(θ)     = E_{(s,a,r,s')~D} [ w · Huber( Q_θ(s,a) − y ) ]   # w = PER importance-sampling weight
```

Double DQN's only change vs vanilla DQN is decoupling selection from evaluation. Vanilla DQN uses `Q_target(s', argmax_{a'} Q_target(s', a'))` and consequently overestimates Q-values (Hasselt 2010).

### Dueling aggregation

```
V(s)         ← Linear(128, 1)
A(s, ·)      ← Linear(128, |A|)            with |A| = 3
Q(s, ·)      = V(s) + ( A(s, ·) − mean_a A(s, a) )
```

Mean-centring the advantages removes the additive identifiability ambiguity in the `V + A` decomposition. Why this helps in trading: large stretches of state are flat (range-bound markets, near-equilibrium); the network can spend a single scalar `V(s)` on those instead of learning three near-equal Q-values.

### Prioritized Experience Replay (proportional)

```
p_i       = (|δ_i| + ε)^α                  # per-transition priority, α ∈ [0, 1]
P(i)      = p_i / Σ_j p_j                  # sampling probability
w_i       = (1 / (N · P(i)))^β / max(w)    # importance-sampling weight, β: 0.4 → 1.0
```

PER is backed by an array-based **sum tree** of length `2N − 1` (`memory/sum_tree.py`); insert and sample are O(log N). New transitions enter with the current max priority, ensuring at least one visit.

### Network architecture (Conv1D trunk)

```
Input  (B, 30, 10)
   permute → (B, 10, 30)
Conv1d(10 → 32, k=5, padding=2) → ReLU
Conv1d(32 → 64, k=3, padding=1) → ReLU
Flatten → (B, 64·30 = 1920)
Linear(1920, 128) → ReLU
   ├── Value head:     Linear(128, 1)        →  V(s)
   └── Advantage head: Linear(128, 3)        →  A(s, ·)
Q(s, ·) = V(s) + (A(s, ·) − A(s, ·).mean(dim=1, keepdim=True))
```

`dueling=False` collapses the heads to a single `Linear(128, 3)` — the vanilla DQN baseline used in the comparative experiment.

## 7. Reward function

| Variant | Formula | Notes |
|---|---|---|
| **Baseline (default)** | `r_t = ΔV_t / V_0` | ΔV already reflects friction (cost + slippage) deducted in `Portfolio.buy/sell` — see ADR-008 in `docs/PLAN.md`. |
| **Risk-adjusted** | `r_t = ΔV_t / V_0 + γ · rolling_sharpe_t` | Rolling annualised Sharpe of recent *portfolio* returns. Window = 20 days, γ = 1.0 by default. |

Friction: `α = 0.001` (transaction cost), `β = 0.001` (slippage). Combined `α + β` is applied per leg as a multiplicative deduction from cash.

The two variants are compared in `ExperimentService.run_reward_variants` — see §10.

## 8. Architecture and class diagram

### Layered architecture

```
Interface (CLI / GUI)
        │
        ▼
   TradingSDK            ◀── single entry point (no consumer touches services directly)
        │
        ▼
   Services              ◀── data · training · backtest · inference · experiment · DQNAgent
        │
        ▼
Env · Model · Memory     ◀── domain code, pure tensor math
        │
        ▼
      Shared             ◀── config · logger · seed · gatekeeper · types
```

Forward arrows only. The architecture diagram source is [`docs/architecture.mmd`](docs/architecture.mmd) (Mermaid). Render with `mmdc -i docs/architecture.mmd -o assets/architecture.png` (or paste into [mermaid.live](https://mermaid.live)).

### Class diagram

See [`docs/class_diagram.mmd`](docs/class_diagram.mmd). The key relationships:

- `TradingSDK` calls into all five services.
- `TrainingService`, `BacktestService`, `InferenceService` each receive (or construct) a `DQNAgent` and a `TradingEnv`.
- `DQNAgent` owns a `DuelingDQN` (online + target) and a `ReplayBuffer` (Uniform or Prioritized — both expose `add`, `sample(beta)`, `update_priorities`).
- `TradingEnv` owns a `Portfolio` and a `RewardFunction` (Baseline or RiskAdjusted).

## 9. How to install and run

```bash
# Install (uv is mandatory — see code guidelines V3)
cd assignment2
uv sync --extra dev

# Verify the package imports
uv run python -c "import dqn_trader; print(dqn_trader.__version__)"
# 1.00

# CLI — top-level help
uv run dqn-trader --help

# Run the data pipeline (uses cached parquet if present, else fetches yfinance)
uv run dqn-trader data --ticker AAPL

# Train for `training.episodes` episodes (defaults to 200 in configs/setup.json)
uv run dqn-trader train

# Backtest a trained checkpoint
uv run dqn-trader backtest --checkpoint results/run_<ts>/checkpoints/best.pt

# Single-action prediction on the latest test window
uv run dqn-trader predict --checkpoint results/run_<ts>/checkpoints/best.pt

# Launch the GUI (PyQt6) — needs a display
uv run python -m dqn_trader.interface.gui
```

The GUI smoke test runs headlessly under `QT_QPA_PLATFORM=offscreen` so it works on machines without an X server.

## 10. How to reproduce the experiments

Once data has been prepared (any `dqn-trader data` call warms the parquet cache), run the four comparative experiments through a short driver:

```python
from dqn_trader.shared.config import ConfigManager
from dqn_trader.services.experiment_service import ExperimentService

cfg = ConfigManager()  # reads configs/setup.json + configs/rate_limits.json
exp = ExperimentService(cfg)
exp.run_dqn_vs_dueling()
exp.run_uniform_vs_per()
exp.run_reward_variants()
exp.run_cross_ticker()    # primary vs configs.data.comparative_ticker (default SPY)
```

Each call appends a Markdown table to `results/experiments_summary.md` and writes a JSON payload `results/<experiment_name>.json` with the full per-condition `BacktestMetrics` + run directories.

The analysis notebook [`notebooks/01_results_analysis.ipynb`](notebooks/01_results_analysis.ipynb) loads these outputs and produces the comparison plots (equity curves, metric bars, drawdown bands) that go into a future revision of this README.

## 11. Test suite and quality gates

```bash
uv run pytest tests/ -q                    # 135 tests, all passing
uv run pytest --cov=dqn_trader tests/      # 97% statement+branch coverage (gate: 85%)
uv run ruff check src/ tests/              # 0 errors
```

- Largest source file: 144 LOC (limit: 150).
- Zero hardcoded magic numbers — every tunable lives in `configs/setup.json` or `configs/rate_limits.json`.
- No `print` statements in library code — everything goes through `shared/logger.py`.
- Tests run **fully offline** (synthetic OHLCV fixture in `tests/conftest.py`).

TDD pairs called out explicitly in the codebase (Red → Green → Refactor):

1. **`RewardFunction.compute`** — the test `test_reward.py::test_baseline_is_normalised_delta_v` was written first; implementation followed.
2. **`PrioritizedReplay.sample`** — `test_prioritized_replay.py::test_priority_update_changes_distribution` written first; implementation refined to satisfy it.

## 12. Answers to the 12 reflection questions

> The assignment §13 asks 12 conceptual questions. Each answer below is tied to specific code in this repo.

**Q1. What does Q(s,a) represent in *your* project, and how is it different from predicting next-day price?**
`Q_θ(s, a)` is the expected discounted *return* from taking action `a` in state `s` and following the learned policy thereafter — measured in our reward units (normalised ΔV ± optional Sharpe bonus), not in dollars. A price predictor would output a single scalar (tomorrow's Close); we output three Q-values, one per action, and the agent decides by `argmax`. Two different states with the same predicted next price can produce different Q-values because the agent considers *what to do* (e.g., a long agent at unrealised gain ≈ 0 may rationally `Sell`; a flat agent in the same market may `Buy`). Code: `model/dueling_dqn.py::forward` returns shape `(B, 3)`, not `(B, 1)`.

**Q2. Why does a continuous, multi-dimensional state space require function approximation instead of a Q-table?**
Our state is a `(30, 10)` float32 tensor — roughly `300` real-valued dimensions. Even after coarse discretisation (say 10 bins per feature), the table would have `10^300` cells, vastly more than the ~750 daily bars of AAPL data we have. A neural network instead learns a *parametric* approximation `Q_θ(s, a)` that *generalises* across similar states. Code: `DuelingDQN.trunk` (Conv1D layers) shares a representation across all observed states; the heads emit Q-values without ever materialising a table.

**Q3. How does the choice of reward function shape the policy?**
- Reward = pure ΔV ⇒ optimum may hold one big position forever (no incentive to trade).
- Reward = pure trade count ⇒ over-trading regardless of profitability.
- Reward = ΔV − cost (our baseline) ⇒ trades only when expected ΔV > friction.
- Reward = ΔV + Sharpe (our risk_adjusted) ⇒ prefers smoother equity curves even at cost of total return.
Code: `environment/reward.py::BaselineReward.compute` vs `RiskAdjustedReward.compute`. The comparison is the `ExperimentService.run_reward_variants` experiment.

**Q4. What if the agent gets rewarded only for immediate profit, with no penalty for transaction cost?**
The agent learns to take any trade with non-zero expected price move — including ones that lose money to friction. Backtest looks reasonable on noise-free synthetic data but fails on real markets, because real friction (`α + β`) is comparable to the average daily mean return. Our solution: `Portfolio.buy/sell` deducts friction at trade time so the friction enters reward implicitly (ADR-008), and the comparative experiment `reward_variants` can show the effect of disabling/enabling the optional risk bonus.

**Q5. Why must the test slice not bleed into training, and what is data leakage in a financial time series?**
Two distinct leakage failure modes:
- **Slice leakage** — shuffling train/test breaks the chronology, so the agent "sees the future" during training and metrics become meaningless. Mitigation: `data/splitter.py::ChronologicalSplitter` slices in order, never shuffles.
- **Statistics leakage** — fitting normalisation, indicators, or hyperparameters using *future* data (including the test slice) lets information leak. Mitigation: `ZScoreScaler.fit` is called exactly once on the train slice (`test_scaler.py::test_no_leakage_means_match_train_only`).

**Q6. When is `Hold` optimal?**
Three cases:
- The market is mean-reverting near current levels and the agent isn't yet positioned for the move.
- The expected ΔV of a Buy/Sell is below the friction floor — taking the trade is negative-EV.
- The agent's risk-adjusted reward says volatility is high but expected return is unchanged — Hold prevents Sharpe degradation. Note in our env: Buy-when-long and Sell-when-flat are also no-ops with reward unchanged (a kind of forced `Hold`).

**Q7. Why does Dueling DQN help when most states are "do nothing"?**
The Dueling decomposition `Q(s,a) = V(s) + (A(s,a) − mean A)` lets a single scalar `V(s)` carry the "how good is this state in general" signal. In trading, market states are often near-equilibrium where all three actions are roughly equal; `V(s)` absorbs that, freeing the Advantage stream to focus on the *rare* moments where actions actually differ. Without Dueling, the network has to learn three near-equal Q-values everywhere, wasting capacity.

**Q8. What's the difference between exploration during training and evaluation during backtest?**
Exploration during training (ε-greedy, `services/epsilon_schedule.py`) deliberately randomises actions so the agent sees diverse `(s, a)` pairs and the Q-function generalises. Evaluation during backtest (`BacktestService.run`) uses ε = 0 (greedy) so the metrics reflect the *policy*, not the noise. Greedy backtest is also reproducible — same checkpoint always yields the same equity curve. Code: `services/backtest_service.py::BacktestService.run` passes `epsilon=0.0` to `agent.act`.

**Q9. Is Total Return enough to evaluate an agent? Why also Sharpe, Max Drawdown, and Win Rate?**
- **Total Return** says nothing about *path*: 30% return achieved by a single windfall trade is fundamentally different from 30% delivered smoothly.
- **Sharpe** says how *consistent* the return was — high Sharpe means little volatility per unit of profit.
- **Max Drawdown** says how *painful* the worst stretch was — crucial for any real position.
- **Win Rate** says how often closed trades were profitable — sanity-checks that returns aren't dominated by a handful of lucky bets.
All four come from `services/risk_metrics.py::summarise` and `BacktestService` reports them together — there is no single sufficient statistic.

**Q10. What bugs in the env or reward could produce a great-looking backtest that's not real?**
- **Look-ahead bias** — executing at price `t` using information from price `t+1`. We avoid this: actions execute at the *current* `Close`; the observation only contains data up to that step.
- **Survivorship reward** — paying the agent just for holding cash (`Hold` always reward > 0). Our baseline gives `Hold` reward = 0 on a flat price.
- **Cost mis-accounting** — applying friction once but rewarding the gross trade. We deduct from cash in the portfolio (ADR-008), so ΔV already reflects friction.
- **Train-statistics on test** — applying a scaler fit on test mean. Our test `test_scaler.py::test_no_leakage_means_match_train_only` asserts this never happens.

**Q11. How would you tell the agent learned a *general* policy rather than memorising a feature of AAPL during the train period?**
- Run the **same training config on a different ticker** (our `ExperimentService.run_cross_ticker` does this) — if performance degrades catastrophically, the agent over-fit AAPL.
- Compare the agent's **trade pattern across regimes** in val and test — if it only trades during the train regime's volatility profile, it's memorised the regime.
- **Reward decomposition** — does Sharpe stay positive on test? Total return alone can be lucky on a single market trend.
- **Ablations** — dropping one of the 8 market features should not destroy performance if the policy is genuinely general.

**Q12. How would you extend this system to a non-financial problem without changing the RL structure?**
The RL contract is decoupled from the trading domain by design: `TradingEnv` is the only place where "trading" exists; everything else (agent, network, replay, training service, SDK) talks in `(state, action, reward, next_state, done)`. To port to, say, energy load shedding:
1. Replace `data/feature_engineer.py` with a feature pipeline over sensor readings.
2. Replace `environment/portfolio.py` with `BatteryState` (charge / discharge instead of buy / sell).
3. Replace `environment/reward.py` with a domain-appropriate reward (penalty for blackouts, reward for cheap-hour charging).
4. Re-tune hyperparameters in `configs/setup.json`.
Nothing in `model/`, `memory/`, `services/training_service.py`, the SDK, the CLI, or the GUI changes. That's the architecture's payoff.

## 13. Sources

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed.
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- Wang, Z. et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- van Hasselt, H. et al. (2015). Deep reinforcement learning with Double Q-learning. *AAAI*.
- Schaul, T. et al. (2016). Prioritized experience replay. *ICLR*.
- [Hugging Face Deep RL Course — Unit 3: Deep Q-Learning](https://huggingface.co/learn/deep-rl-course).
- [Gymnasium API documentation](https://gymnasium.farama.org/).
- [rmisegal/DQN-stock](https://github.com/rmisegal/DQN-stock) — the reference project from the assignment; we replicate the architecture pattern (`SDK → services → env / model / memory`) but the implementation here is independent.
- yfinance — Yahoo Finance market data downloader.

---

## Appendix: per-mechanism PRDs

Drill-down on individual algorithmic pieces:

- [`docs/PRD_dqn.md`](docs/PRD_dqn.md) — vanilla DQN baseline.
- [`docs/PRD_dueling.md`](docs/PRD_dueling.md) — Value / Advantage decomposition.
- [`docs/PRD_double_dqn.md`](docs/PRD_double_dqn.md) — decoupling selection from evaluation.
- [`docs/PRD_per.md`](docs/PRD_per.md) — proportional Prioritized Experience Replay.
- [`docs/PRD_reward.md`](docs/PRD_reward.md) — baseline + risk-adjusted reward.
- [`docs/PRD_env.md`](docs/PRD_env.md) — `TradingEnv` API and observation assembly.
- [`docs/PRD_features.md`](docs/PRD_features.md) — the 10 state channels.
- [`docs/PRD_data_pipeline.md`](docs/PRD_data_pipeline.md) — yfinance → tensors.
- [`docs/PRD.md`](docs/PRD.md) — top-level PRD.
- [`docs/PLAN.md`](docs/PLAN.md) — architecture and ADRs.
- [`docs/TODO.md`](docs/TODO.md) — per-layer build log with Definition of Done.
