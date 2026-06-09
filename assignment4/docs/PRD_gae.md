# PRD — Generalized Advantage Estimation (GAE)

## Theory recap

GAE (Schulman et al. 2016) is an estimator for the advantage function `A^π(s, a) = Q^π(s, a) − V^π(s)`. The naive choices are:

- **1-step TD**: `Â_t = δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`. Low variance, high bias (depends on `V` accuracy).
- **Monte-Carlo return**: `Â_t = Σ γ^l · r_{t+l} − V(s_t)`. Unbiased, high variance.

GAE introduces a **bias-variance dial** via a second discount factor `λ ∈ [0, 1]`:

```
δ_t^V = r_t + γ·V(s_{t+1}) − V(s_t)                                                (2a)
Â_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}^V                                      (2b)
```

In reverse-recursion form (used in code):

```
Â_t = δ_t + γλ · (1 − done_t) · Â_{t+1}                                            (2c)
```

The `(γλ)^l` weight is *geometric in two factors*: γ controls how much future matters at all, and λ controls how much we trust *future TD errors* relative to the current one. The slide-16 limits:

| λ | What GAE collapses to | Bias | Variance |
|---|---|---|---|
| 0 | TD error `δ_t` | high (only V's bias) | low |
| 1 | Monte-Carlo `Σ γ^l r_{t+l} − V(s_t)` | low (unbiased) | high |
| 0.95 (default) | weighted blend | medium | medium |

## Why "geometric weighted sum of TD errors"?

The math derivation (Schulman 2016 § 3) shows that GAE is the unique advantage estimator that satisfies a particular "exponentially weighted average of n-step returns" property. Intuitively: a small λ trusts the immediate TD signal heavily (the critic is doing the work); a large λ trusts the trajectory's actual rewards more (the critic is just a baseline).

The connection to PPO: PPO clips on the *advantage sign* (slides 11–12), so noisy or biased advantages cause PPO to make wrong-direction updates that the clipping can't prevent. **Quality of GAE → stability of PPO** (slide 14). This is the core link between the two halves of the lecture.

## Implementation

```python
def compute_gae(
    rewards: np.ndarray,      # shape (T,)
    values: np.ndarray,        # shape (T,)
    last_value: float,         # bootstrap V(s_T)
    dones: np.ndarray,         # shape (T,) bool
    gamma: float,
    lam: float,
) -> np.ndarray:               # shape (T,) advantages
    T = rewards.shape[0]
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages
```

**Critical detail**: `next_non_terminal` zeroes out the bootstrap on terminal transitions. Without it the recursion bleeds across episode boundaries — a common bug in beginner GAE impls.

## Inputs / outputs / setup

- **Input to `compute_gae`**: per-step rewards, critic values, the bootstrap value for the rollout's terminal state, and per-step `done` flags.
- **Output**: per-step advantages (shape `(T,)`).
- **Setup**: `configs/setup.json:env.gamma` + `gae.lambda`.

## Acceptance criteria

- `test_gae.py::test_lambda_zero_reduces_to_td_error` — λ = 0 ⇒ output equals `r + γV(s') − V(s)` per step.
- `test_gae.py::test_lambda_one_reduces_to_mc_minus_v` — λ = 1, dones all False ⇒ output equals `Σ γ^l r_{t+l} + γ^T·last_value − V(s_t)`.
- `test_gae.py::test_closed_form_three_step` — known 3-step trajectory with hand-computed expected GAE values.
- `test_gae.py::test_terminal_truncates_recursion` — done at step `k` zeroes the bootstrap through that step.

## Why this lives in a standalone module

GAE is a **pure function** — no state, no logger, no side effects. Putting it inside `PPOService` would tangle two concerns (advantage estimation + training loop). Standalone makes it trivially testable and reusable; if a future student wants to plug GAE into a different algorithm (DDPG advantage variant, etc.) the import is `from proximal_lab.services.gae import compute_gae`.

## Where this lives

- `src/proximal_lab/services/gae.py` — `compute_gae`, ≤ 60 LOC.
- `tests/unit/test_gae.py` — 4-test math battery.

## Caveats

- GAE assumes the critic `V(s)` is unbiased (or close to it). If `V` systematically over- or under-estimates, all λ values inherit that bias. The slide-21 "Advantage quality" pillar is about *making `V` good*, which is the critic training loop's job.
- GAE is computed over **fixed-length rollouts**, not full episodes. The `last_value` bootstrap is what stitches across rollout boundaries.

## Sources

- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR, 2016.
- L08 lecture slides 15–16 (Dr. Yoram Segal, May 2026).
