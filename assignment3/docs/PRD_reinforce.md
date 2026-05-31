# PRD — REINFORCE (Part D)

## Theory recap

REINFORCE is the original episodic policy-gradient algorithm (Williams, 1992). It directly differentiates `J(θ) = E_τ[R(τ)]`:

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ) ]
```

Replacing the global return `R(τ)` with the per-step reward-to-go `G_t = Σ_{k=t}^T γ^{k-t} r_k` gives better credit assignment (Sutton & Barto §13.3). Subtracting a state-independent baseline `b` reduces variance without bias:

```
∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b)
```

## Implementation choices

| Decision | Value | Why |
|---|---|---|
| Policy network | MLP: 16 → 128 → 128 → 5 logits | Matches the actor head of A2C for fair comparison |
| Sampling | `Categorical(softmax(logits)).sample()` | Standard stochastic policy |
| Baseline | Mean of `G_t` across the episode | Kept simple; A2C's learned baseline is the natural next step |
| γ (discount) | 0.99 | Matches A2C |
| lr (Adam) | 1e-3 | Stable for episode counts of 300 |
| Episodes | 300, 28 steps each | Matches assignment §7.4.1 |

## Inputs / outputs / setup

- **Input to `ReinforceService.fit(env)`:** a `WorldEnv` with `reset()` and `step(action)`.
- **Output:** trained `PolicyNet` weights + per-episode metrics CSV.
- **Setup:** `configs/setup.json:reinforce.*`.

## Acceptance criteria

- `test_policy_network.py::test_forward_shape` — `(B, 16)` input produces `(B, 5)` logits.
- `test_policy_network.py::test_softmax_sums_to_one` — output probabilities sum to 1.
- `test_reinforce_service.py::test_single_update_reduces_loss` — one gradient step on a synthetic episode reduces a quadratic surrogate loss.
- `test_reinforce_service.py::test_reward_to_go_arithmetic` — `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...` matches hand-computed values.
- Reward curve over 300 episodes shows a positive trend (mean of last 30 > mean of first 30).

## Variance characteristics

REINFORCE accumulates rewards across the whole episode before updating. Variance is high because:
- Single trajectory samples can be very different from the expectation.
- All steps in an episode share the same return signal — they're rewarded or penalised together.

The mean-baseline correction reduces but doesn't eliminate this. A2C's learned state-conditional baseline `V_ψ(s)` is the next improvement, and is the entire point of Part E.

## Why include REINFORCE if A2C is "better"?

The course pedagogy (slides 1–25) is the *evolution* REINFORCE → +baseline → +Advantage → Actor-Critic → A2C. Each step removes one problem at a time. Skipping REINFORCE breaks the pedagogical chain — we wouldn't be able to show why A2C is needed.
