# PRD — Soft (Polyak) target updates

> Mechanism PRD for slide 6 of L09 and Requirement 2 of EX05. The headline stability mechanism of DDPG.

## 1. Goal

Implement Polyak averaging to maintain "slow-moving" target networks for the actor and critic, **so that the critic's bootstrap target Q'(s', μ'(s')) does not chase a constantly-moving target during training**.

## 2. The equation (slide 6, EX05 § Item 2)

For each pair of parameter tensors `(θ_target, θ_source)`:

$$\theta_{\text{target}} \;\leftarrow\; \tau \,\theta_{\text{source}} + (1 - \tau)\,\theta_{\text{target}}$$

with `τ ∈ (0, 1]`. Default τ = 0.005 per Lillicrap 2016 + EX05 explicit example.

Three regimes:

| τ | Behaviour |
|---|---|
| 0.0 | Target frozen forever |
| 0.005 (default) | ~200-step exponential smoothing |
| 1.0 | Hard copy — equivalent to **no target network** |

## 3. Implementation

```python
# model/soft_update.py
def polyak_update(
    target_params: Iterable[torch.nn.Parameter],
    source_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """Polyak (soft) update: θ_target ← τ · θ_source + (1 − τ) · θ_target.

    Performed in-place under `torch.no_grad()` to keep the target network
    disconnected from autograd. Spec § Item 2: 'show exact code lines'.
    """
    assert 0.0 <= tau <= 1.0
    with torch.no_grad():
        for t, s in zip(target_params, source_params, strict=True):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)
```

## 4. Reflection answer for the spec (Q3)

> *"How do target networks + soft updates protect the critic from collapse?"*

Two distinct mechanisms:

1. **Stationary target**: The TD target `y = r + γ Q'(s', μ'(s'))` is computed from a network whose parameters change much more slowly than `Q`'s own. Without this, the critic would chase a target that moves at every gradient step → divergence (the classic deadly triad).
2. **Slow policy drift**: The actor target `μ'` similarly smooths the policy used inside the Q-target. If we used the **current** μ inside the bootstrap, a single overshooting actor update would propagate into a wildly different y, then into a wildly different Q, then into a wildly different actor — a tight feedback loop. Polyak breaks the loop by spreading any single update over ~`1/τ` ≈ 200 steps.

This PRD's empirical evidence is **Layer 11 § run_target_network_ablation.py**: a side-by-side training curve of `τ=0.005` (smooth Polyak) vs `hard_copy_every=1` (= no target net = equivalent to τ=1.0 every step). The hard-copy variant should show critic-loss divergence or oscillation that does not appear under soft updates.

## 5. Test plan (the 4-test math battery)

In `tests/unit/test_soft_update.py`:

| Test | Setup | Assert |
|---|---|---|
| τ = 0 → target unchanged | Random source params; clone target; call `polyak_update(τ=0)` | `target == target_before` |
| τ = 1 → hard copy | Random source/target params; call `polyak_update(τ=1)` | `target == source` |
| τ = 0.5 → midpoint | source = ones, target = zeros, τ=0.5 | `target == 0.5` |
| Repeated calls converge | source = ones, target = zeros, τ=0.05, 100 calls | `target → 1.0 within 1e-3` |

## 6. Acceptance criteria

1. `polyak_update` lives in `model/soft_update.py` with the exact docstring above.
2. All 4 math-battery tests pass.
3. `services/ddpg_service.py` calls `polyak_update` exactly once per update step, after both actor + critic optimisations.
4. The hard-copy ablation flag (`ddpg.hard_copy_every_n > 0`) replaces Polyak with `target.load_state_dict(source.state_dict())` every N steps.

## 7. Non-goals

- TD3-style double critics — out of scope; mentioned in PRD § 13 honest-acknowledgements only.
- Layer-norm or weight-norm on critic — out of scope.
