# PRD — PPO (Layer 15 — beyond-spec)

## Theory recap

PPO (Schulman et al. 2017) replaces A2C's single-step policy gradient with a **clipped surrogate objective** computed over a buffer of recent transitions, and optimises it for several gradient steps per data batch:

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)         # importance ratio
L_clip(θ) = E_t [ min(r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t) ]
```

The clip prevents the actor from drifting too far from the policy that *collected* the data, which is the failure mode A2C exhibits at higher learning rates. The full objective adds a value-function loss and an entropy bonus:

```
L(θ) = −L_clip(θ) + c1 · L_VF(θ) − c2 · H[π_θ]
```

## Why PPO completes the chain

| Algorithm | Update | Variance | Stability | Where in this repo |
|---|---|---|---|---|
| REINFORCE | full episode `G_t` | very high | low | `services/reinforce_service.py` |
| REINFORCE + mean baseline | `G_t − b` | high | low | same, `use_baseline=True` |
| A2C | one-step TD `δ_t` | low | medium — sensitive to actor/critic balance | `services/a2c_service.py` |
| **PPO** | clipped surrogate over a buffer | low | **high** — clipping bounds the update | `services/ppo_service.py` |

The lecture mapped REINFORCE → +baseline → +advantage → Actor-Critic. PPO is the production-grade successor and is the natural fourth step.

## Implementation choices

| Decision | Value | Why |
|---|---|---|
| Network | re-use `ActorCriticNet(hidden=128)` | same architecture as A2C → comparison isolates the learning rule |
| `clip_eps` | 0.2 | Schulman et al. recommended default |
| `n_epochs_per_batch` | 4 | reuse data 4× per rollout; standard PPO recipe |
| `n_steps_per_update` | 28 | one full episode buffer; matches assignment episode length |
| `gamma` | 0.99 | same as REINFORCE / A2C |
| `lr` | 5e-4 | same as A2C actor lr |
| `value_coef` | 0.5 | standard |
| `entropy_coef` | 0.01 | matches A2C |

## Advantage estimation

We use the same one-step TD advantage A2C uses (`A_t = r_t + γ·V(s_{t+1}) − V(s_t)`) rather than full GAE. This keeps the PPO/A2C ablation clean — the *only* difference is the clipped objective + multi-epoch update. GAE is listed as future work in [`docs/TODO.md`](TODO.md).

## Acceptance criteria

- `test_ppo_service.py::test_invalid_init_args_raise` — bad clip_eps, n_epochs, n_steps rejected.
- `test_ppo_service.py::test_fit_returns_per_episode_metrics` — episode_count, action_counts, mean_entropy shapes.
- `test_ppo_service.py::test_single_update_changes_weights` — one update visibly moves the network.
- `test_ppo_service.py::test_clip_bounds_importance_ratio` — verify the clipped objective implementation against the mathematical definition.
- Multi-seed final-30 % reward ≥ A2C's on the same env (within CI).

## Honest acknowledgement

PPO was not in the original assignment spec — the assignment lists REINFORCE + A2C as the two algorithms. PPO is added as a *beyond-spec* differentiator to demonstrate the full policy-gradient evolution chain. It is not a substitute for any required deliverable.
