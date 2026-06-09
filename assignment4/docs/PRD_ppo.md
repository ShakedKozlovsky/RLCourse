# PRD — PPO (Proximal Policy Optimization)

## Theory recap

PPO (Schulman et al. 2017) emerged as the practical successor to TRPO. TRPO bounded the KL divergence between the new and old policies explicitly — solving a constrained optimisation with a Fisher information matrix — which delivered monotonic improvement guarantees but required second-order solvers. PPO replaced the explicit KL constraint with a **clipped surrogate objective**:

```
L^CLIP(θ) = Ê_t [ min( r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]                  (1)

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

The clip flattens the surrogate outside `[1−ε, 1+ε]` so the optimiser never gets rewarded for taking enormous policy steps. The `min` makes the choice **deliberately pessimistic**: PPO always picks the more conservative of the two surrogates (slide 13).

## Why "min" and not "clip alone"?

If the surrogate were just `clip(r_t)·Â`, then for `r > 1+ε ∧ Â < 0` the clipped value would be `(1+ε)·Â` (less negative than `r·Â`), so the gradient would *encourage* further movement in the wrong direction — precisely the catastrophic step PPO is trying to prevent. The `min` ensures: when the policy has *already* moved past the trust region in the wrong direction, the unclipped (more negative) surrogate wins, and the gradient pulls the policy back. The four cases (slides 11–12):

| Sign of `Â` | `r` region | Winner of `min` | Effect on gradient |
|---|---|---|---|
| `Â > 0` | `r ∈ [1−ε, 1+ε]` | doesn't matter (equal) | normal PG update |
| `Â > 0` | `r > 1+ε` | clipped branch (smaller) | gradient is flat → no further push |
| `Â > 0` | `r < 1−ε` | unclipped branch | normal PG, encourage growth |
| `Â < 0` | `r ∈ [1−ε, 1+ε]` | equal | normal PG update |
| `Â < 0` | `r > 1+ε` | **unclipped (more negative) — the safety case** | gradient pulls policy back into trust region |
| `Â < 0` | `r < 1−ε` | clipped branch (flat) | no further push |

This is the headline diagnostic the project tests against.

## Implementation choices

| Decision | Value | Why |
|---|---|---|
| Optimizer | Adam, lr=3e-4 | Schulman 2017 + most subsequent PPO implementations |
| Clip ε | 0.2 | Slide 10 default; sweep in Layer 10 |
| Rollout steps | 2048 | Standard PPO config |
| Minibatch size | 64 | Standard PPO config |
| Epochs per update | 10 | Standard PPO config; pivot of slide-4 "On-Policy sensitivity" |
| Value coefficient `c_1` | 0.5 | Standard |
| Entropy coefficient `c_2` | 0.0 | Continuous control rarely needs it; ablated in Layer 10 |
| Max gradient norm | 0.5 | Stability — clip global grad norm |
| Advantage normalisation | per-minibatch (`(adv − mean) / (std + 1e-8)`) | Standard PPO trick |
| Target-KL early stop | off by default; optional | Stops an epoch early if KL exceeds target |

## Key diagnostics to log per update

- **Mean KL** between old and new policy — should be small, monotone-increasing across epochs within an iteration.
- **Clip fraction** — what proportion of minibatch transitions had `r_t` outside `[1−ε, 1+ε]`. Slide 21 says PPO trusts the update only when this fraction is *small*; if it spikes, the policy is moving too aggressively.
- **Explained variance** of the critic — `1 − Var(returns − V) / Var(returns)`. Should approach 1.0.
- **Episode reward** mean and std across vectorised envs.

These four diagnostics together prove the slide-21 "Three stability layers" — Loop (rollout pipeline), Signal (advantage quality), Policy (clipping bound).

## Inputs / outputs / setup

- **Input to `PPOService.fit(net, env, total_timesteps)`**: a vectorised `MuJoCoEnv` and an `ActorCriticNet`.
- **Output**: trained network in-place + `TrainResult` with per-iteration metrics.
- **Setup**: `configs/setup.json:ppo.*`.

## Acceptance criteria

- `test_ppo_clip_math.py::test_clip_within_window` — `r ∈ [1−ε, 1+ε]` ⇒ unclipped branch loss = `−r·Â`.
- `test_ppo_clip_math.py::test_clip_above_window_positive_adv` — `r > 1+ε ∧ Â > 0` ⇒ clipped branch wins; loss flattens.
- `test_ppo_clip_math.py::test_clip_above_window_negative_adv` — `r > 1+ε ∧ Â < 0` ⇒ unclipped branch wins; pulls policy back.
- `test_ppo_clip_math.py::test_clip_below_window_negative_adv` — `r < 1−ε ∧ Â < 0` ⇒ clipped branch wins.
- `test_ppo_service.py::test_smoke_5000_steps_halfcheetah` — completes without divergence.
- `test_ppo_service.py::test_target_kl_early_stop` — when `target_kl` is set, an epoch breaks early.

## Where this lives

- `src/proximal_lab/services/ppo_service.py` — `PPOService` class + `ppo_clip_loss` helper.
- `tests/unit/test_ppo_clip_math.py` — the 4-test math battery.
- `tests/integration/test_ppo_service.py` — smoke + diagnostic tests.

## Caveats

- PPO is **on-policy**. The buffer is discarded after each update; re-using stale data would violate the surrogate-objective derivation. Slide 4 explicitly flags this as a sensitivity, not a feature to work around.
- The clipped surrogate is a *bound*, not the true objective. PPO can take a sub-optimal step if the advantage estimate is noisy (slide 14) — which is why GAE quality (next PRD) matters so much.

## Sources

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel, "Trust Region Policy Optimization," ICML, 2015.
- L08 lecture slides 3–14, 18, 21 (Dr. Yoram Segal, May 2026).
