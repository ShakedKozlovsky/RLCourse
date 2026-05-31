# PRD — A2C (Advantage Actor-Critic) (Part E)

## Theory recap

A2C addresses REINFORCE's high variance by adding a **learned state-value function** `V_ψ(s)` (the critic) that serves as a state-conditional baseline. The advantage becomes:

```
A_t = Q^π(s_t, a_t) − V_ψ(s_t)  ≈  r_t + γ·V_ψ(s_{t+1}) − V_ψ(s_t)  =  δ_t
```

The right-hand side is the **TD error**, used both as the advantage signal for the actor and as the training signal for the critic:

```
Actor:  θ ← θ + α · δ_t · ∇_θ log π_θ(a_t|s_t)
Critic: ψ ← ψ + β · δ_t · ∇_ψ V_ψ(s_t)               (equivalently: minimize ½ δ_t²)
```

## Architecture

```
Input s_t  (B, 16)
  │
  ├── shared trunk: Linear(16,128) → ReLU → Linear(128,128) → ReLU
  │
  ├── actor head:   Linear(128, 5)        → π_θ(a|s) via softmax
  └── critic head:  Linear(128, 1)        → V_ψ(s)
```

Shared trunk allows feature reuse and reduces parameter count vs separate networks. The actor and critic interact through the TD error.

## Implementation choices

| Decision | Value | Why |
|---|---|---|
| Actor lr | 5e-4 | Lower than REINFORCE — TD targets evolve, slower actor avoids chasing a moving baseline |
| Critic lr | 1e-3 | Critic should converge faster than the actor for stable advantage estimates |
| γ | 0.99 | Same as REINFORCE for fair comparison |
| Update rule | One-step TD (each (s,a,r,s') updates both nets) | Matches the slide's pseudo-code (slide 20) |
| Entropy bonus | 0.01 × H(π) | Prevents premature policy collapse (mentioned in slide 22 as a stability issue) |
| Episodes | 300, 28 steps each | Same budget as REINFORCE |

## Why A2C is "more stable" than REINFORCE

The slide §21 emphasises: **REINFORCE's signal arrives at end-of-episode, A2C's signal arrives each step.** Concretely:

- REINFORCE update at episode end uses `G_t` (sum of future rewards) — noisy, all steps share the same signal.
- A2C uses `δ_t = r_t + γV(s') − V(s)` — local, per-step, and the critic learns to reduce its own noise.

Empirically: A2C's reward curve should show **lower variance** than REINFORCE's even when both converge to similar mean rewards.

## Inputs / outputs / setup

- **Input to `A2CService.fit(env)`:** a `WorldEnv` (same as REINFORCE).
- **Output:** trained `ActorCriticNet` + per-episode metrics CSV.
- **Setup:** `configs/setup.json:a2c.*`.

## Acceptance criteria

- `test_actor_critic_network.py::test_forward_shapes` — actor returns `(B, 5)`, critic returns `(B, 1)`.
- `test_a2c_service.py::test_td_error_sign` — synthetic transition where `V(s') > V(s)` and `r > 0` ⇒ `δ > 0`.
- `test_a2c_service.py::test_single_update_reduces_critic_loss` — one step decreases `½ δ²` on synthetic data.
- A2C reward curve has **lower coefficient of variation** than REINFORCE over the last 30% of training.

## Caveats from the lecture (slide 22)

A2C is not strictly better — it requires:
1. An accurate critic (biased V hurts the actor).
2. Hyperparameter compatibility (actor and critic learning rates must be matched).
3. Balance between learners (one converges too fast → the other gets stuck).

We document these explicitly in the README's analysis section.
