# PRD — Vanilla DQN (baseline)

> Used as the baseline in the comparative experiment against the Dueling + Double + PER stack.

## Theory recap

The action-value function `Q*(s,a)` satisfies the Bellman optimality equation:

```
Q*(s,a) = E[ r + γ · max_{a'} Q*(s', a') | s, a ]
```

A neural network `Q_θ` approximates `Q*`. Training minimizes the temporal-difference error:

```
y = r + γ · max_{a'} Q_{θ_target}(s', a')        (target net, frozen between syncs)
L(θ) = E_{(s,a,r,s')~D} [ Huber( Q_θ(s,a) − y ) ]
```

## Inputs / outputs / setup

- **Input:** state tensor `(B, 30, 10)` float32.
- **Output:** Q-values vector `(B, 3)` for actions {Sell, Hold, Buy}.
- **Setup:** `gamma`, `lr`, `batch_size`, `target_sync_every`, `huber_delta` — all from `configs/setup.json`.

## Acceptance criteria

- Implements Bellman target as **single-network max** (this is the vanilla form, *not* Double DQN).
- Reuses `model/dueling_dqn.py` with a flag (or an alternative `model/vanilla_dqn.py` that shares the Conv1D extractor) — to be decided in Layer 3 commit. Trade-off documented in the commit message.
- Test: forward pass + one optimization step on synthetic batch → loss is finite, gradient non-zero.

## Why we include it

We need the vanilla baseline to demonstrate (in Layer 8) the *quantitative* contribution of Dueling and Double-DQN — otherwise the choice of the heavier architecture is unjustified.
