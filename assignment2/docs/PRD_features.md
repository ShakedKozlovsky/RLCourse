# PRD — Feature Engineering (10 channels)

## Goal

Transform raw OHLCV into 10 dense, scale-controlled feature channels per day. The output is a `pandas.DataFrame` indexed by date with exactly 10 columns in a fixed order.

## Channels (ordered, this order is contractual)

| # | Column name | Formula | Range after scaling | Notes |
|---|---|---|---|---|
| 1 | `log_return` | `log(Close_t / Close_{t-1})` | ~Normal(0, σ²) | First value is NaN → dropped |
| 2 | `rsi_14` | Wilder's RSI, 14 periods | [0, 100] then z-scored | Uses EMA-of-gains / EMA-of-losses |
| 3 | `macd` | `EMA(12) − EMA(26)` of Close | unbounded → z-scored | EMA of Close, not of returns |
| 4 | `macd_signal` | `EMA(9)` of `macd` | unbounded → z-scored | |
| 5 | `macd_hist` | `macd − macd_signal` | unbounded → z-scored | |
| 6 | `bb_pct` | `(Close − BB_lower) / (BB_upper − BB_lower)` | typically ∈ [0, 1] | Bollinger %B, window 20, σ=2 |
| 7 | `vwap_dist` | `(Close − VWAP_20) / VWAP_20` | small float → z-scored | Rolling VWAP over 20 days |
| 8 | `volume_norm` | z-score of `log(Volume + 1)` over a rolling 60-day window | ~Normal(0, 1) | Robust to volume regime shifts |
| 9 | `position` | (filled by `TradingEnv` at runtime) | {0, 1} | Not in the static feature DF |
| 10 | `pnl_unrealised` | (filled by `TradingEnv` at runtime) | scaled by V_0 | Not in the static feature DF |

`FeatureEngineer.fit_transform(raw_ohlcv_train_df) → train_features_df` produces channels 1–8 only. The portfolio channels 9–10 are injected by the environment at each step.

## Why each feature is in (not just "more is better")

- **log_return** — direction + magnitude in one number; gradient-friendly (centred near 0).
- **rsi_14** — standard overbought/oversold marker; gives the agent a momentum-mean-reversion signal that's complementary to MACD.
- **macd / signal / hist** — momentum on three time scales; the histogram captures the *acceleration* (second derivative), which the network would otherwise have to learn implicitly.
- **bb_pct** — volatility-normalised position; the same absolute price move means different things in calm vs noisy regimes.
- **vwap_dist** — price relative to a recent fair-value proxy; complements bb_pct (volatility-free measure of overextension).
- **volume_norm** — participation. A breakout on low volume is statistically different from one on high volume.
- **position / pnl_unrealised** — the agent must know it owns the asset and what its current paper P&L is, otherwise the same market state is observationally identical for a flat vs long agent.

## Data leakage prevention

`FeatureEngineer` is `fit_transform` on **train slice only** for normalisation parameters (means/stds for z-scoring). `transform` on val/test uses the fitted parameters — the artifact is persisted in `Scaler.state_dict()` and saved alongside the model checkpoint.

## Acceptance criteria

- Computed RSI / MACD / Bollinger values match a reference implementation (e.g. `pandas-ta` or `ta`) on a known toy series within 1e-6.
- Output DataFrame has exactly 8 columns (channels 1–8) named exactly as in the table above (order matters).
- The first ~26 rows (longest indicator warmup) are dropped — confirmed by row count in `test_feature_engineer.py`.
- No NaNs in the output.
