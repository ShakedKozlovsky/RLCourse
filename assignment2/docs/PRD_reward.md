# PRD — Reward Function

## Why the reward function is the heart of the problem

The reward function is the formal contract between the trading objective and the RL learning process. A poorly-shaped reward — for example, one that only credits realised PnL on Sell — produces an agent that hoards forever; one that rewards trade count produces an agent that churns. The whole problem of trading is encoded in the reward.

## Variants implemented

### Baseline (default)

```
r_t = ΔV_t / V_0  −  α · |trade_value_t| / V_0  −  β · |trade_value_t| / V_0
```

- `ΔV_t = portfolio_value_t − portfolio_value_{t-1}` — the marked-to-market change in equity (works for Hold too, not just Sell).
- `α` = `transaction_cost_alpha`, default `0.001`.
- `β` = `slippage_beta`, default `0.001`.
- `trade_value_t` = 0 when action is Hold or invalid; equals `notional` when action is Buy/Sell that actually changes the position.
- Normalisation by `V_0` keeps reward on `~O(1e-3)` per step, suitable for Huber + Adam at `lr=5e-4`.

### Risk-adjusted (comparative experiment)

```
r_t = ΔV_t / V_0  −  α · trade_cost  −  β · slippage  +  γ · rolling_sharpe_t
```

- `rolling_sharpe_t` = annualised Sharpe of the last 20 daily returns of the *portfolio* (not the asset). Computed inside `RewardFunction` so the env stays stateless on this aspect.
- `γ` = `sharpe_bonus_gamma`, default `1.0` (only when this variant is selected).

## Inputs / outputs / setup

- **Inputs to `RewardFunction.compute(...)`:** `prev_value`, `new_value`, `trade_value`, `step_return_history` (for the risk-adjusted variant).
- **Output:** scalar float reward.
- **Setup:** `env.reward_variant ∈ {"baseline", "risk_adjusted"}`, plus the three coefficients.

## Acceptance criteria

- `test_reward.py::test_baseline_no_trade_equals_delta_v` — when no trade, reward equals normalised ΔV.
- `test_reward.py::test_buy_then_sell_round_trip` — net reward on a flat (no-price-change) round-trip equals `−2·(α+β)·notional/V_0` (the two-way cost).
- Comparative experiment (Layer 8) reports for each variant: total return, Sharpe, Max DD, **trade count**, **fraction-of-time-in-position**. We expect risk-adjusted to trade less and have higher Sharpe; the comparison is the educational outcome whether or not that expectation holds.

## Pitfalls explicitly avoided

- **Survivorship reward:** never give a positive reward purely for surviving — that would create a Hold-only optimum (which is actually fine but uninteresting). Our baseline reward gives 0 for a Hold on a flat price, which is correct.
- **Per-trade bonus:** never reward "made a trade" — that's the textbook recipe for over-trading.
- **End-of-episode lump sum:** never delay all reward to the final step — the temporal-credit-assignment burden becomes unrealistic for a Q-learner.
