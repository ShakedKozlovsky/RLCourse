# PRD — TradingEnv (Gymnasium-style)

## Responsibilities

- Maintain market position (cash, holdings, portfolio value) via `Portfolio`.
- Step through the dataset one daily bar at a time.
- Translate `(action, current_bar, next_bar)` into a trade, applying commission + slippage.
- Compute reward via `RewardFunction`.
- Build the observation tensor `(30, 10)` by combining the pre-scaled 8 market channels with the 2 portfolio channels.
- Emit termination at the end of the slice.

## API (Gymnasium-compatible)

```python
class TradingEnv:
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    @property
    def action_space(self) -> spaces.Discrete: ...  # = Discrete(3)
    @property
    def observation_space(self) -> spaces.Box: ...  # = Box(shape=(30, 10))
```

`info` dict carries: `portfolio_value`, `cash`, `position`, `step_idx`, `trade_executed`, `trade_value`, `realized_pnl_step`.

## State assembly

At step `t` the observation is built from:

- Columns 0..7: pre-scaled market features for days `t-29 .. t` (already prepared by `DataService`).
- Column 8 = current `position` ∈ {0, 1} broadcast across the time dimension.
- Column 9 = current `pnl_unrealised` (scaled by `V_0`) broadcast across the time dimension.

Broadcasting the portfolio scalars across all 30 time steps gives the network a constant signal of the agent's *current* state, in contrast to the time-varying market features.

## Termination

`done = True` when the cursor reaches the last bar in the current slice (train/val/test). `truncated` is `False` for now (we use full episodes); reserved for future per-step time-limits via `training.max_steps_per_episode`.

## Acceptance criteria

- `test_trading_env.py::test_reset_returns_correct_shape` — `(30, 10)` float32.
- `test_trading_env.py::test_step_advances_one_bar` — `info["step_idx"]` increments by exactly 1.
- `test_trading_env.py::test_buy_at_flat_then_sell_zero_drift` — flat-price round-trip ends with `cash ≈ V_0 − 2·(α+β)·V_0` (within float tolerance).
- `test_trading_env.py::test_invalid_action_is_no_op` — Buy while long doesn't change `position` or `cash`.
- `test_trading_env.py::test_done_at_end` — after exactly `N_bars` steps, `done == True`.

## Why this is not `gym.Env` exactly

We follow the Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)` but expose the `np.ndarray` directly rather than registering with `gym.envs`. The training service does not call `gym.make` — we instantiate `TradingEnv` directly. This keeps the environment trivially mockable and dependency-light.
