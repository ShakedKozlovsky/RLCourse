
## dqn_vs_dueling

| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |
|---|---|---|---|---|---|---|
| vanilla_dqn | `agent.dueling=False` | -13.383% | -1.55 | -19.13% | 25.00% | 8 |
| dueling_dqn | `agent.dueling=True` | -22.305% | -3.93 | -24.09% | 35.71% | 14 |

## uniform_vs_per

| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |
|---|---|---|---|---|---|---|
| uniform_replay | `per.enabled=False` | -0.238% | -0.11 | -2.35% | 66.67% | 3 |
| prioritized_replay | `per.enabled=True` | -22.305% | -3.93 | -24.09% | 35.71% | 14 |

## reward_variants

| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |
|---|---|---|---|---|---|---|
| baseline | `env.reward_variant=baseline` | -22.305% | -3.93 | -24.09% | 35.71% | 14 |
| risk_adjusted | `env.reward_variant=risk_adjusted` | -16.354% | -1.75 | -27.33% | 33.33% | 6 |

## cross_ticker

| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |
|---|---|---|---|---|---|---|
| AAPL | `data.ticker=AAPL` | -22.305% | -3.93 | -24.09% | 35.71% | 14 |
| SPY | `data.ticker=SPY` | -8.815% | -1.44 | -10.70% | 33.33% | 3 |

## window_sensitivity

| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |
|---|---|---|---|---|---|---|
| window_10 | `data.window_size=10` | -23.564% | -2.46 | -26.80% | 20.00% | 10 |
| window_20 | `data.window_size=20` | -24.167% | -2.18 | -27.52% | 12.50% | 8 |
| window_30 | `data.window_size=30` | -22.305% | -3.93 | -24.09% | 35.71% | 14 |
| window_50 | `data.window_size=50` | -13.662% | -1.58 | -17.91% | 0.00% | 1 |
