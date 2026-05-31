# PRD — LSTM World Model (Part C)

## What this is and isn't

The LSTM is **the environment**, not the recommender. It learns the *transition function* `P(s_{t+1} | s_t, a_t)` from the synthetic trajectory:

```
s_{t+1} ≈ f_φ(s_t, a_t, h_t)
```

Where `h_t` is the LSTM's hidden state, encoding the trainee's recent history. This makes the system POMDP-aware (slide 1.4): the true Markov state would need to capture all relevant history, and the LSTM approximates that hidden state via its recurrent representation.

## Why an LSTM, not a feed-forward network?

The next day's state depends on more than just today — accumulated fatigue, training pattern over the past week, recovery from earlier sessions. A feed-forward net taking `(s_t, a_t)` would lose this. The LSTM's hidden state `h_t` carries that history forward.

This is exactly the **World Model** concept (Ha & Schmidhuber, 2018): learn a recurrent dynamics model, then train a policy *inside* the model.

## Architecture

```
input (B, T, D_s + D_a)  ←  [state_t, action_one_hot_t] concatenated, T=window=7
   │
LSTM(D_s+D_a → 64, num_layers=1, batch_first=True)
   │
take last hidden state h_T  (B, 64)
   │
Linear(64, D_s)  →  predicted s_{t+1}
```

- `D_s = 16` (state dim from PLAN.md §5)
- `D_a = 5` (action one-hot)
- Window length `W = 7` days of context
- 1 layer keeps the parameter count low; can extend if validation justifies.

## Training procedure

1. Build all rolling windows of length `W` from the synthetic trajectory.
2. Split 80/20 train/val by window position (chronological — earlier windows train, later validate).
3. For each window: input = sequence of `(s_t, a_t)` pairs of length `W`, target = `s_{t+W}` (the next state after the window).
4. Loss = MSE between predicted and actual `s_{t+W}`.
5. Adam, lr = 1e-3, 100 epochs, early stopping on val loss with patience 10.

## Why MSE on continuous state, not classification?

Most state features are continuous (volume, distribution, duration). The `day_in_cycle` one-hot is the exception, but we treat the whole vector as continuous regression — clean, single-loss formulation. The downstream policy uses these continuous values regardless of how they were produced.

## Inputs / outputs / setup

- **Input to `WorldModelService.train(trajectory)`:** the synthetic trajectory as a numpy array of shape `(T, D_s)` plus per-day actions.
- **Output:** trained `LSTMWorldModel` saved to `saved_models/world_model.pt` + loss curve.
- **Setup:** `configs/setup.json:world_model.*`.

## Acceptance criteria

- `test_lstm_world_model.py::test_forward_shape` — `(B, W, D_s+D_a)` input produces `(B, D_s)` output.
- `test_lstm_world_model.py::test_finite_outputs` — no NaN/Inf for any randomly-initialised forward pass.
- `test_world_model_service.py::test_training_reduces_loss` — 50 epochs on synthetic data reduces val loss from initial value.
- `test_world_model_service.py::test_save_load_roundtrip` — saved model produces identical outputs when reloaded.

## Where the model is used

After training:
- `environment/world_env.py` instantiates a `WorldEnv` that wraps the LSTM.
- `WorldEnv.step(action)` runs one LSTM forward step using current state + action + maintained hidden state.
- Both REINFORCE and A2C use the same env, ensuring algorithmic comparison fairness.

## Honest acknowledgement (from assignment §F.1)

The dataset is workout *programs*, not physiological *outcomes*. The LSTM learns the temporal pattern of a chosen program — not muscle growth, fatigue, or injury risk. This is documented prominently in the README: the system is a "structurally realistic recommender", not a medical-grade simulator.
