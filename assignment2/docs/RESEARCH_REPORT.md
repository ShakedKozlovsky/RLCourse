# Research Report — DQN Trading Agent

> What we built, what we found, what worked, and what didn't.

## 1. Research question

**Can a Dueling Double DQN with Prioritized Experience Replay learn a profitable daily trading policy on AAPL / SPY from 2020–2023 OHLCV data, and if so, how do the algorithmic components (Dueling, PER, reward shaping) contribute individually?**

## 2. Methodology

- **Data:** Yahoo Finance daily OHLCV for AAPL and SPY (756 bars each, 2020-01-01 to 2023-01-01). 70/15/15 chronological split — no shuffling, no leakage.
- **Features:** 8 causal market channels (log_return, RSI-14, MACD/signal/histogram, Bollinger %B, VWAP distance, z-scored volume) + 2 portfolio channels (position, unrealised PnL), giving a state tensor of shape (30, 10) per step.
- **Algorithms tested:** vanilla DQN, Dueling DQN, with both uniform and prioritised replay, and two reward variants (baseline ΔV/V₀ vs risk-adjusted ΔV/V₀ + γ·Sharpe).
- **Training:** 30 episodes per condition, seed 208904839, Adam lr=5e-4, batch_size=64, γ=0.99, ε: 1.0 → 0.05 over 8000 steps.

## 3. Key findings

### Finding 1: the agent overfits the train slice

| Metric | Train (ep 29) | Val (ep 29) | Test (backtest) |
|---|---|---|---|
| Return | +2700% | oscillates −10% to +4% | −22.3% |
| Sharpe | (not meaningful on one episode) | ~0 | −3.93 |

The agent learns to exploit patterns in the train slice (reward rises from 0 to ~2.5 over 30 episodes), but these patterns don't transfer to the validation or test slices. This is **expected** given:
- Only 459 train windows (each 30 days × 10 features)
- 30 training episodes — far too few to generalise a Conv1D+FC network
- AAPL 2020–2022 contains two distinct regimes (COVID crash-and-recovery, then a bear market) that don't repeat in the test slice

**Conclusion:** for an educational project, demonstrating *that* the agent learns on train and *discussing why* it overfits is more valuable than faking good test results.

### Finding 2: PER amplifies overfitting (not reduces it)

| Buffer type | Test return | Trades |
|---|---|---|
| Uniform | −0.24% | 3 |
| Prioritised | −22.31% | 14 |

Uniform replay produced a near-passive agent (3 trades, near-flat equity). PER, by design, up-weights transitions with high TD error — but on noisy financial data, the highest-error transitions are often *outliers*, not *signal*. The result: PER pushes the agent to trade more aggressively, incurring more friction and more over-fitted decisions.

**Conclusion:** PER is not universally better than uniform replay. In low-signal domains (daily equities), it can amplify noise. This is a finding *because* it contradicts the naive reading of Schaul et al. (2016), which was benchmarked on Atari, not financial time series.

### Finding 3: risk-adjusted reward reduces trade count and improves Sharpe

| Reward | Test return | Sharpe | Trades |
|---|---|---|---|
| Baseline | −22.31% | −3.93 | 14 |
| Risk-adjusted | −16.35% | −1.75 | 6 |

The Sharpe bonus in the risk-adjusted reward explicitly penalises volatility. The agent responds by trading less (6 vs 14), which reduces friction and produces a smoother (less bad) equity curve. Total return is also better (−16% vs −22%).

**Conclusion:** reward shaping works as theory predicts — the agent optimises what you measure. Including a risk term produces a more conservative policy.

### Finding 4: the policy is ticker-regime-specific

| Ticker | Test return | Trades |
|---|---|---|
| AAPL | −22.31% | 14 |
| SPY | −8.82% | 3 |

Training on SPY (a broad, lower-volatility ETF) produces a much more passive agent. The same hyperparameters that caused over-trading on AAPL's volatile 2020–2022 period produce near-inaction on SPY's calmer trajectory. This confirms that the policy is *not* a general "how to trade" skill — it's a regime-specific behaviour.

**Conclusion:** generalisation across asset regimes is an open problem. Our cross-ticker experiment provides quantitative evidence of the gap.

## 4. Design decisions discovered during implementation

We logged eight Architectural Decision Records (ADRs) in [`docs/PLAN.md`](PLAN.md). The two most interesting:

- **ADR-007: features-first-then-split.** The original plan computed features per-slice (after splitting). Integration tests caught a bug: per-slice warmup drops ~26 days from val/test independently, leaving too few windows. All our indicators are causal, so computing on the full series before splitting introduces no leakage but recovers the warmup days. *Lesson: integration tests catch architectural bugs that unit tests miss.*

- **ADR-008: friction in V, not in reward.** The initial reward formula subtracted friction as a separate term: `r = ΔV/V₀ − α·trade − β·trade`. But `ΔV` already reflects the friction deducted by `Portfolio.buy/sell` — the formula double-counted. *Lesson: the simplest formula (`r = ΔV/V₀`) is often the correct one when the bookkeeping layer is honest.*

## 5. What we would do differently with more time

1. **Longer training** (200+ episodes) to see whether Dueling and PER catch up to their lighter baselines.
2. **Window-size sweep** (10, 20, 30, 50) — the current 30-day window is a fixed hyperparameter.
3. **Feature ablation** — systematically drop one channel at a time to measure information contribution.
4. **Attention-based explainability** — which of the 30 days in the window mattered most for the chosen action?
5. **Regularisation** (dropout, weight decay, early stopping by val metric) — the obvious first response to the overfitting finding.

## 6. What worked well in our development process

- **Layered commits** made it trivial to bisect a bug to its layer.
- **TDD for the reward function** caught the double-counting friction bug before it contaminated training runs.
- **Synthetic OHLCV fixture** (400-day random walk) let all 135 tests run offline in 17 seconds.
- **SDK + thin CLI + thin GUI** meant every feature we added was automatically available in all three interfaces with no code duplication.
- **The ADR habit** (documenting *why* a decision was changed) produced the best content for this report.
