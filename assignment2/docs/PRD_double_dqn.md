# PRD — Double DQN

## Theory recap

Vanilla DQN uses the same network to *select* and *evaluate* the best next action in the Bellman target, which causes systematic overestimation of Q-values (Hasselt 2010; van Hasselt et al. 2015):

```
y_DQN     = r + γ · max_{a'} Q_target(s', a')
y_Double  = r + γ · Q_target( s', argmax_{a'} Q_online(s', a') )
```

Double-DQN decouples selection (from the *online* network) and evaluation (from the *target* network).

## Why it helps in trading

Financial reward signals are noisy. Overestimation in vanilla DQN tends to push the policy toward aggressive Buy/Sell actions early in training, inflating costs. Double-DQN's conservative target makes the policy more stable in the presence of transaction-cost terms.

## Inputs / outputs / setup

- **Setup flag:** `agent.double_dqn: true` in `configs/setup.json`.
- When `true`, the training service computes the target using two networks; when `false`, falls back to single-net max (vanilla DQN target — see `PRD_dqn.md`).

## Implementation note

The change is exactly two lines in `services/training_service.py`. The trade-off is zero — Double DQN is strictly an improvement on the target computation. We include the flag only to enable the comparative experiment.

## Acceptance criteria

- Comparative experiment in Layer 8: same seed, same data, only flag differs. Reported metrics: mean episode reward over last 30% of training, val_sharpe, test total return.
- Unit test on synthetic 2-state, 2-action problem with a known optimal Q* — Double-DQN target value matches the analytical formula within float tolerance.
