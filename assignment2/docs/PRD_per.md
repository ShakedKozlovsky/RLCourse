# PRD — Prioritized Experience Replay (PER)

## Theory recap

Uniform replay treats all transitions equally. PER (Schaul et al., 2016) samples transitions in proportion to their TD-error magnitude — surprising transitions are sampled more often because the network can learn more from them.

Proportional variant:

```
p_i  = ( |δ_i| + ε )^α                      # priority (α ∈ [0,1], 0 = uniform, 1 = pure prioritized)
P(i) = p_i / Σ_j p_j                        # sampling probability
w_i  = ( 1 / (N · P(i)) )^β                 # importance-sampling weight (β annealed 0.4 → 1.0)
```

The loss is multiplied element-wise by `w_i / max(w)` to correct for the non-uniform sampling bias.

## Implementation choice

- **Backing data structure:** SumTree (binary tree with leaf priorities and internal sums). O(log N) insert + sample.
- **Capacity:** 100k (from config). Initial priority for new transitions = current max priority (so they're sampled at least once).
- **`update_priorities(indices, td_errors)`:** called after every optimization step.

## Inputs / outputs / setup

- **Inputs to `sample(batch_size)`:** the desired batch size + current `beta`.
- **Outputs:** `(states, actions, rewards, next_states, dones, indices, is_weights)`.
- **Setup keys in config:** `per.enabled`, `per.alpha`, `per.beta_start`, `per.beta_end`, `per.beta_anneal_steps`, `per.epsilon`.

## Acceptance criteria

- `test_prioritized_replay.py::test_sample_probability_proportional_to_priority` — empirical sample frequency after many samples matches `p_i^α` within statistical tolerance.
- `test_prioritized_replay.py::test_is_weights_in_unit_range` — `w_i ∈ (0, 1]` after normalisation.
- `test_prioritized_replay.py::test_update_priorities_changes_distribution` — after `update_priorities`, sampling shifts toward the updated transitions.
- Comparative experiment in Layer 8: Uniform vs PER with otherwise identical configs.

## Trade-offs

- More code than uniform — mitigated by encapsulating it behind a `ReplayBuffer` interface so the training service only depends on the interface.
- More memory (SumTree overhead) — acceptable: 100k transitions × (state + scalars) ≈ a few hundred MB, well within memory budget.
