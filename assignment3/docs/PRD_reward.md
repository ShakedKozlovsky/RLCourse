# PRD — Reward Function

## The contract

The reward function is the formal contract between fitness-domain objectives and policy gradient learning. It must:

1. **Encourage useful training** — high enough that the policy learns to prefer training over rest.
2. **Penalise overload** — high recent volume = injury risk.
3. **Penalise imbalance** — repeating the same muscle group = poor program design.

## Formula

```
r_t = gain_t  −  λ_1 · overload_penalty_t  −  λ_2 · imbalance_penalty_t
```

| Term | Computation | Range (typical) |
|---|---|---|
| `gain_t` | `total_volume_t / V_max`, clamped to `[0, 1]` | [0, 1] |
| `overload_penalty_t` | rolling 7-day volume average / V_max, clamped | [0, 1] |
| `imbalance_penalty_t` | KL-divergence of muscle distribution from uniform | ≥ 0 |

`V_max` is the trajectory's maximum total volume (computed once during data preparation).

## Defaults

| Parameter | Value | Source |
|---|---|---|
| λ_1 (overload) | 0.2 | `configs/setup.json:env.reward_overload_lambda` |
| λ_2 (imbalance) | 0.3 | `configs/setup.json:env.reward_imbalance_lambda` |

## Why these defaults?

- λ_1 = 0.2 < 1: gain dominates; the agent should still want to train, just not maximally.
- λ_2 = 0.3 > λ_1: imbalance is the more pedagogically important problem (collapse to one action = bad policy). Higher λ_2 makes the imbalance signal stronger.

These are documented as pedagogical defaults, not medical recommendations.

## Reward signal example

For a day where the agent chose Push (chest+shoulders+triceps), with `gain_t = 0.6`, rolling avg `= 0.5`, imbalance KL `= 0.4`:

```
r_t = 0.6 − 0.2·0.5 − 0.3·0.4 = 0.6 − 0.1 − 0.12 = 0.38
```

For a Rest Day (gain=0, no volume, no imbalance change): `r_t = 0`.

For a 3rd consecutive Rest Day: `gain=0` but the action mask (if enabled) would forbid this — so this scenario is policy-level rather than reward-level.

## Acceptance criteria

- `test_reward.py::test_gain_zero_on_rest` — Rest action with no volume produces `r_t = 0`.
- `test_reward.py::test_overload_penalty_increases_with_volume` — repeated high-volume days produce increasing penalty.
- `test_reward.py::test_imbalance_decreases_with_variety` — alternating muscle groups produces lower imbalance penalty than repeating one.
- `test_reward.py::test_lambdas_from_config` — values come from `configs/setup.json`, not hardcoded.

## Comparative experiment

`scripts/run_reward_sweep.py` varies `(λ_1, λ_2)` in `{(0.0, 0.0), (0.2, 0.0), (0.0, 0.3), (0.2, 0.3), (0.5, 0.5)}` and reports action distribution + reward curves. Expected: higher λ_2 → more balanced policies (more action diversity).

## Pitfalls explicitly avoided

- **No per-action bonus**: rewarding "made a choice" leads to compulsive trading; rewarding "took a non-rest action" leads to ignoring rest. We reward outcomes (volume gain), not actions.
- **No survival bonus**: paying for "didn't fail" is fitness-equivalent of paying for "didn't sell" — produces a degenerate stay-still policy.
- **No end-of-episode lump sum**: the assignment §7.4.2 emphasises continuous, dense reward — and the slide §6 shows why dense rewards (Reward-to-Go) improve credit assignment.
