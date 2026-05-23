# PRD — Dueling DQN

## Theory recap

The Q-value decomposes into a state-value `V(s)` and an action-advantage `A(s,a)`:

```
Q(s,a) = V(s) + ( A(s,a) − mean_{a'} A(s,a') )
```

Subtracting the mean of the advantages identifies the decomposition (otherwise V and A are only identifiable up to an additive constant) and stabilises training.

## Why it helps in trading

In daily stock trading, *Hold* is the optimal action for a large fraction of states — the market is often range-bound, illiquid, or near-equilibrium. Vanilla DQN forces the network to learn three nearly-equal Q-values for those states. Dueling lets the network learn a single scalar `V(s)` for "how good is this market state in general" and only spend capacity in the Advantage head when the action *actually* matters.

## Architecture

After the shared Conv1D feature extractor and `Flatten`:

```
Linear(64*30, 128) → ReLU
   ├── Value head:      Linear(128, 1)        → V(s)
   └── Advantage head:  Linear(128, |A|=3)    → A(s,a)
Q(s, ·) = V(s) + ( A(s, ·) − A(s, ·).mean(dim=-1, keepdim=True) )
```

## Inputs / outputs / setup

- **Input:** `(B, 30, 10)` float32.
- **Output:** `(B, 3)` Q-values.
- **Setup:** `dueling: true` flag in `configs/setup.json` (when false, the heads collapse into a single `Linear(128, 3)`).

## Acceptance criteria

- Test `test_dueling_dqn.py::test_q_decomposition_identity` — given fixed weights, `Q(s, ·)` reconstructs from `V(s)` and `A(s, ·) − mean`.
- Test `test_dueling_dqn.py::test_forward_shape` — input `(2, 30, 10)` produces output `(2, 3)`, all finite.
- README plot in Layer 8: training reward curve for Dueling vs vanilla on the same seed.

## Alternatives considered (ADR-002)

- LSTM/GRU encoder → rejected: adds training instability and doesn't help on 30-day windows once MACD/RSI are present.
- Transformer encoder → rejected: overkill for `(30, 10)`; high training cost.
- Conv1D → chosen.
