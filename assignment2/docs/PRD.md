# PRD — DQN Trading Agent (Assignment 2)

> **Course:** Reinforcement Learning with Deep Learning
> **Author:** Shaked Kozlovsky (ID 208904839)
> **Reference implementation we draw inspiration from:** `rmisegal/DQN-stock`
> **What this project IS:** an educational, full-stack RL system that learns a *trading policy* via DQN.
> **What this project is NOT:** a price predictor, a trading recommendation, or a production trading bot.

---

## 0. PRD evolution log

This PRD evolved across the project's life — it was not written once and frozen. Each layer's implementation fed corrections back into the requirements.

| Version | Layer | What changed and why |
|---|---|---|
| v1 (Layer 0) | Planning | Initial requirements written from the assignment PDF + slides + coding rules. |
| v2 (Layer 1) | Data | ADR-007: changed pipeline from split-then-features to features-then-split. The integration test caught a warmup bug. Updated §4 and §5 to reflect the corrected ordering. |
| v3 (Layer 2) | Environment | ADR-008: simplified reward formula from `ΔV − cost` to `ΔV` only. Friction is now deducted inside Portfolio.buy/sell, not in the reward function. Updated §7 and created PRD_reward.md v2. |
| v4 (Layer 5) | Backtest | Added `report_name` parameter to `TradingSDK.backtest()` to prevent experiment conditions from overwriting each other's equity curves. |
| v5 (Layer 10) | Experiments | Added window-size sweep (10/20/30/50) as a new experiment — not in original scope but motivated by Finding 4 (regime specificity). |
| v6 (post-Layer 10) | Improvement | Applied learnings: changed defaults to window=50, uniform replay, lr=2e-4, 100 episodes. Results improved from −22.3% to −10.8% test return. |
| v7 (audit) | Quality | Outsider audit found 10 gaps: added data preview tables, TDD walkthrough, price/feature charts, reproducibility section, GUI feature explanation. |
| v8 (final) | Deliverables | Executed the results notebook (LaTeX + inline plots), committed pre-trained checkpoint (1.6 MB), wrote executive summary for grader quick-read. |
| v9 (proof) | Validation | 6-panel model proof: reward ↑, loss ↓, Q-spread > 0, val return **+17.2%** (positive on unseen data). Test failure (−10.8%) diagnosed as regime shift, not algorithm bug. |

This evolution is the point — a PRD that never changes means the team didn't learn anything during implementation.

---

## 1. Project goal

Implement a complete, object-oriented DQN system around a continuous-time decision environment, using daily stock data as the case study. The agent must learn a *decision policy* (Sell / Hold / Buy) from a sequence of states, actions, and rewards — not predict the next price.

The grade rewards understanding of the RL formulation, environment design, reward function, training stability, results analysis, architecture quality, and explanations. A high backtest return alone does **not** guarantee a high grade.

## 2. Scope mapping to lecture slides

| Lecture concept (slides) | How it appears in this project |
|---|---|
| Q-Table → Function Approximation (3–6, 14–15) | Section 8 + `model/network.py`; README explanation of why a table fails for a 30×10 continuous state. |
| RL framing: Agent/Env/State/Action/Reward/Episode/Policy/Return (7–10) | Section 3 below + README mapping table; mirrored in `environment/trading_env.py` (`reset()`, `step()`). |
| Data → state tensor (11–13) | `data/` pipeline + 30-day × 10-feature tensor (this PRD §5). |
| DQN and Dueling DQN (16–21) | `model/dueling_dqn.py` with separate Value and Advantage heads. |
| Exploration & stabilization (22–24) | ε-greedy schedule, Replay Buffer, Target Network in `training/`. |
| Full training cycle (25) | `services/training.py` orchestrates reset→act→step→store→sample→optimize→target-sync. |
| Backtest & results analysis (26–27) | `services/backtest.py` produces Equity Curve, Buy-and-Hold benchmark, Sharpe, Max Drawdown, Win Rate. |
| OOP, tests, architecture (28–29) | SDK + services + clean OOP layering; pytest with ≥85% coverage. |
| Theoretical recap (30–31) | README final section stresses that the agent learns a *policy*, not a price forecast. |

## 3. RL problem formulation

The agent acts at discrete daily steps. At each step `t`:

- `s_t` — observation window: last 30 daily bars × 10 features (market + portfolio channels). Shape `(30, 10)`.
- `a_t ∈ {Sell=0, Hold=1, Buy=2}` — discrete action.
- `r_t` — scalar reward derived from change in portfolio value, transaction cost, slippage, and a risk-adjusted Sharpe term.
- `s_{t+1}` — next-day window after applying the action.
- `done` — terminal flag at the end of the episode (end of the train/test slice).

**Policy:** π_θ(a|s) = argmax_a Q_θ(s, a) during evaluation; ε-greedy during training.
**Return:** G_t = Σ γ^k · r_{t+k}, discount γ from config (default 0.99).
**Position discipline:** all-in / all-out. A Buy while already long, or a Sell while flat, is a no-op (reward unchanged, optional small penalty configurable to discourage churn — disabled by default to keep the basic reward clean).

## 4. Dataset — mandatory specification

| Parameter | Required value |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Primary ticker | AAPL |
| Primary range | 2020-01-01 to 2023-01-01 |
| Interval | `1d` |
| Raw columns | Open, High, Low, Close, Volume |
| Cache | `data/raw/{ticker}_{start}_{end}.parquet` (snappy) |
| CSV fallback | `data/raw/{ticker}.csv` with Date index |
| Split | 70% Train / 15% Validation / 15% Test, **chronological**, no shuffle |
| Window size | 30 |
| Feature count | 10 |

**Comparative experiment (mandatory):** one additional ticker (SPY or NVDA) using the same pipeline.

**Data leakage rules (non-negotiable):**

- All normalization / indicator statistics are computed from the *train* slice only and applied to validation/test.
- Hyperparameter selection uses validation, never test.
- Train/Val/Test are split chronologically before any feature scaling.

## 5. Feature engineering — the 10 channels

| # | Channel | Definition (educational) | Why the agent needs it |
|---|---|---|---|
| 1 | `log_return` | `log(Close_t / Close_{t-1})` | Direction + magnitude of daily move. |
| 2 | `rsi_14` | Wilder's RSI, period 14 | Overbought / oversold context. |
| 3 | `macd` | EMA12 − EMA26 of Close | Momentum signal. |
| 4 | `macd_signal` | EMA9 of MACD | Trend confirmation. |
| 5 | `macd_hist` | MACD − Signal | Acceleration of momentum. |
| 6 | `bb_pct` | Bollinger %B (20, 2σ) | Position within volatility band. |
| 7 | `vwap_dist` | (Close − rolling VWAP_20) / VWAP_20 | Mean-reversion distance. |
| 8 | `volume_norm` | z-score of Volume over a rolling window | Participation strength. |
| 9 | `position` | {0, 1} — currently holding the asset | Without this, the same market state is identical for a flat or long agent. |
| 10 | `pnl_unrealised` | Mark-to-market PnL on current position, scaled | Lets the agent "feel" drawdown on an open trade. |

Window assembly: at step `t`, stack the 10 channels for days `t-29 .. t` ⇒ tensor of shape `(30, 10)`.

## 6. Action semantics

| State | Action 0 (Sell) | Action 1 (Hold) | Action 2 (Buy) |
|---|---|---|---|
| Flat (position=0) | No-op | No-op | Enter long at next bar's Open; apply commission + slippage |
| Long (position=1) | Exit at next bar's Open; realize PnL; apply commission + slippage | Hold | No-op |

Configurable optional **invalid-action penalty** (default 0). Documented but disabled by default.

## 7. Reward function

Baseline (default):

```
r_t = ΔV_t − transaction_cost_alpha · |trade_value_t| − slippage_beta · |trade_value_t|
```

Where `ΔV_t = portfolio_value_t − portfolio_value_{t-1}`, normalized by initial equity so reward stays O(1).

Optional risk-aware variant (used in comparative experiment):

```
r_t = ΔV_t − cost − slippage + sharpe_bonus_gamma · rolling_sharpe_t
```

Mandatory comparative experiment: train two agents — one with baseline reward, one with risk-aware reward — and compare equity curves, Sharpe, drawdown, and trade count. Reported in README §"Experiments".

Defaults (overridable via config):

- `transaction_cost_alpha = 0.001`
- `slippage_beta = 0.001`
- `sharpe_bonus_gamma = 1.0` (only when enabled)
- `initial_capital = 10000`

## 8. Algorithm requirements

All three of the following are mandatory:

1. **Dueling DQN** with separate Value V(s) and Advantage A(s,a) streams:
   `Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)`.
2. **Double DQN** for the Bellman target to reduce overestimation:
   `y = r + γ · Q_target(s', argmax_a Q_online(s', a))`.
3. **Prioritized Experience Replay (PER)** with proportional priorities, importance-sampling weights, and a TD-error-based priority update.

In addition: ε-greedy with decay, a periodically synced Target Network, Huber loss, gradient clipping, and checkpoint-on-best-validation.

**Comparative experiment (also mandatory):** a vanilla DQN + uniform Replay baseline vs the full Dueling + Double + PER stack on the same data — to demonstrate the contribution of each component.

## 9. Architecture

A strict layered architecture, the GUI / CLI never touch model or environment directly — they all go through the SDK. Detailed component design lives in `docs/PLAN.md`.

```
GUI / CLI ──▶ TradingSDK ──▶ Services (training, backtest, inference, data)
                              │
                              ├─▶ Environment (TradingEnv, RewardFunction)
                              ├─▶ Model (DuelingDQN, ReplayBuffer/PER, TargetNet)
                              └─▶ Data (YFinanceClient, FeatureEngineer, Splitter)
```

## 10. KPIs and acceptance criteria

| KPI | Target | Where it appears |
|---|---|---|
| Test coverage | ≥ 85% statement, branch enabled | `pytest --cov`, CI-style report |
| `ruff check` | 0 errors | linter gate |
| File size | ≤ 150 lines per .py file | enforced manually + checklist |
| Train converges | Episode reward trend ↑ across last 30% of training | Loss/Reward plots in `results/` |
| Backtest beats Buy-and-Hold on at least one of {AAPL, SPY/NVDA} on Sharpe | Reported with full metric table | `results/backtest_*.json` + README |
| Reproducibility | Same seed → same metrics within tolerance | seed fixed in `configs/setup.json` |
| Architecture purity | No DQN logic in GUI/CLI; no GUI imports in environment/model | code review + dependency graph |

## 11. Non-functional requirements

- **OOP-first:** every responsibility is a class; no procedural top-level code in the library.
- **Configuration:** zero hardcoding. All hyperparameters in `configs/setup.json` and `configs/rate_limits.json` (for yfinance API throttling).
- **uv** is the only package manager.
- **TDD:** tests written alongside (Red → Green → Refactor) for at least the environment, replay buffer, and Dueling forward pass — documented in README.
- **Logging:** every layer logs through `shared/logger.py` (stdlib logging, no print statements in library code).
- **No internet at evaluation time:** all data must be cached locally; tests must run fully offline.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| yfinance occasional 429 / column-shape changes | API gatekeeper (rate-limited, retried), MultiIndex collapse, CSV fallback per spec. |
| Data leakage from full-series normalization | Fit scaler on train slice only, persist `Scaler` artifact, apply to val/test. |
| Reward hacking via no-op spamming | Action semantics make no-ops free but produce no reward; small optional churn penalty available. |
| Q-value overestimation | Double DQN target; Huber loss; target net sync every N steps. |
| Backtest looks great but is a fluke | Mandatory comparative ticker; report Sharpe and Max DD, not just total return. |

## 13. Deliverables

- Full source under `src/dqn_trader/`
- Tests under `tests/` with ≥85% coverage
- `docs/` with this PRD, `PLAN.md`, `TODO.md`, per-mechanism PRDs
- `configs/setup.json`, `configs/rate_limits.json`
- `assets/` with screenshots and architecture diagrams
- `results/` with experiment outputs and the experiments summary file
- A README that doubles as a learning document with the 12 reflection questions answered

## 14. Out of scope

- Real broker integration, live trading, intraday data, options/futures, portfolios of >1 asset (an optional excellence extension may add this later but is not in the baseline).
- Hyperparameter search frameworks (Optuna, Ray Tune) — manual sweeps only.
- Distributed training.
