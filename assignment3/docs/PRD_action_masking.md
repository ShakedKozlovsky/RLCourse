# PRD — Action Masking (Excellence — Assignment §F.1)

## What and why

**Action Masking** = injecting expert domain knowledge into the policy by setting the logits of *unsafe* or *implausible* actions to `−∞` before softmax. The policy can still choose any masked action only if all alternatives are masked (it shouldn't happen with sensible rules).

Per the assignment §F.1 and Huang & Ontañón (2022): masking is a more principled way to combine expert priors with learning than "guard-rails" applied externally.

## Rules implemented

| Rule | Logic | Justification |
|---|---|---|
| **No 3 consecutive same-group days** | If the last 2 days were both PUSH (or PULL, or LEGS, ...) the corresponding action is masked on day t. | A real coach would not prescribe the same group 3 days in a row — recovery time required. |
| **No 3 consecutive REST days** | If the last 2 days were both REST, REST is masked on day t. | Avoid degenerate "always rest" policy. |
| **Day 0 fallback** | On the first step, no masking (no history). | We need history to apply the rules; first step is unrestricted. |

## Where this lives

`src/fitness_rl/environment/action_mask.py`:

```python
class ActionMask:
    """Pre-softmax mask producing a (-inf, 0) mask vector by examining recent history."""

    def __init__(self, max_same_group: int = 2, max_rest: int = 2): ...
    def mask(self, recent_actions: list[int]) -> np.ndarray:
        """Return a (5,) array of 0s and -inf values."""
```

Both `PolicyNet` (REINFORCE) and `ActorCriticNet` (A2C) accept an optional mask argument in `forward()`:

```python
logits = self.linear_out(features)
if mask is not None:
    logits = logits + mask        # adds -inf to masked actions
return logits
```

## Configuration

`configs/setup.json:env.action_masking_enabled = false` by default. Enabled in one experiment to demonstrate its effect (per ADR-006: don't bake the differentiator into the "vanilla" comparison).

## Acceptance criteria

- `test_action_mask.py::test_no_history_no_mask` — empty `recent_actions` ⇒ all-zero mask.
- `test_action_mask.py::test_two_consecutive_push_masks_push` — `[PUSH, PUSH]` ⇒ logits[PUSH] = -inf.
- `test_action_mask.py::test_two_consecutive_rest_masks_rest` — `[REST, REST]` ⇒ logits[REST] = -inf.
- `test_action_mask.py::test_masked_softmax_sums_to_one` — after softmax, total probability is still 1 (mass redistributed to unmasked actions).
- Experiment: REINFORCE + masking vs REINFORCE without — compare reward curves and action distributions.

## Why this is a differentiator

The base assignment requires implementing REINFORCE and A2C. Action Masking is explicitly called out as an **excellence** direction in §F.1. Most students will not implement it — making this a clear differentiator. We:

1. Implement masking as a reusable component.
2. Run one experiment showing the effect.
3. Discuss in the README's analysis section: when masking helps, when it hurts.

## Caveats

- Hard masking changes the policy's effective support set. The gradient is computed only over the unmasked actions, which can speed up convergence at the cost of removing the policy's ability to "learn" the bad behaviour was bad (it's just forbidden).
- The choice of mask rules is a domain-expert decision. We document our rules and acknowledge that they're heuristic, not derived from physiological data.
