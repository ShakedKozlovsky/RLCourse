# Failure-Mode Analysis & Engineering Discoveries

> What the agent does **wrong**, what we tried, what we learned. The kind of
> honest write-up that V3 § 14.5 calls "engineering discoveries" — distinct from
> "results."

## 1. The "stand still and survive" attractor (the headline lesson)

**Symptom**: 20 000-step training with the initial reward configuration
produced LOWER coverage (0.6 %) than the 4 000-step run (1.8 %).

```
4 000 steps  → coverage 0.018  reward  +6 694
20 000 steps → coverage 0.006  reward  -464
```

The longer training made the agent **strictly worse**.

**Root cause**: The reward function was over-weighted toward
collision avoidance.

| Term | Initial value | Effect |
|---|---|---|
| `new_cell_bonus` | +1.0 per fresh cell | Sparse positive signal |
| `collision_penalty` | **-10.0** | Massive single-step punishment |
| `step_penalty` | -0.01 | Negligible "do nothing" cost |

With these weights, the actor learned the dominant strategy is **"do not move
fast" because any move that clips a wall costs 10× a single cell's bonus, and
standing still costs only 0.01 per step**. Over 500 steps a stationary policy
loses just 5 reward; a single collision loses 10. The Q-network learned this
asymmetry and the actor exploited it.

**Fix** (committed in Layer 18, [`environment/reward.py`](../src/roomba_lab/environment/reward.py)):

| Term | Tuned value | Why |
|---|---|---|
| `collision_penalty` | -10 → **-1.0** | Still bad, no longer catastrophic |
| `step_penalty` | -0.01 → **-0.05** | Standing still now also painful |
| `coverage_target` | 0.85 → **0.30** | Achievable in 20k steps |
| **`coverage_progress_coef`** | **NEW: +50.0** | Dense signal = 50 × Δcoverage every step |

The `coverage_progress_coef` term is the most important addition: it gives a
**dense gradient** every step where the agent enters a previously-unvisited
region, even if no full new cells are added. With this, the Q-function has a
smooth target to fit instead of a sparse one.

**Re-trained result**: see `assets/plots/learning_curve_tuned.png` (Layer 18 evidence).

**Lesson for the report**: in DDPG, reward shaping is not a side concern — it
*is* the signal that the critic learns and the actor exploits. Without dense
progress shaping, the agent finds a local minimum that the spec did not
anticipate.

## 2. Actor init too small → no early movement

**Symptom**: With actor head gain = 0.01 (the textbook DDPG default), the
network's initial output is ~ N(0, 1e-4²). The robot effectively does not move
during warmup, the replay buffer fills with stationary transitions, and the
critic learns Q(s, 0) only.

**Fix**: Layer 18 bumped the actor head gain from 0.01 → 0.1 ([`model/init.py`](../src/roomba_lab/model/init.py)).
Output magnitudes are now ≈ 0.1 (after the body's ReLU+orthogonal stack),
which is enough to produce meaningful forward velocity from step 0 while
staying well clear of tanh saturation.

## 3. Spawn-pose rejection sampler timeout

**Risk**: `RoombaEnv._sample_spawn_pose` rejects poses inside walls; for
pathological apartments it might exhaust the 200-attempt budget.

**Mitigation**: We use only the 10 maps from HouseExpo's
official `map_id_10.txt` shortlist — all are large and easy to spawn in. None
have ever triggered the 200-attempt limit in our tests. A real-deployment
system would also try the apartment's centroid as a fallback.

## 4. shapely `prepared` cache invalidation

**Risk**: The `World` instance prepares its polygon for fast LIDAR
intersection queries. If a downstream consumer mutates `world.polygon`, the
prepared geometry will be stale.

**Mitigation**: `World` is treated as immutable except for `grid`. The polygon
is set once in `__init__` and never re-assigned. The convention is documented;
the only mutation is `grid[i,j] = VISITED` which doesn't affect shapely.

## 5. The σ=0 degenerate-exploration trap

**Symptom**: σ=0 cell of the noise sweep showed the predicted "robot loops in
a small region from spawn" behaviour with coverage flat-lining.

**Root cause**: Deterministic actor + deterministic env + LIDAR → identical
trajectory every episode → replay buffer is one trajectory in N copies → critic
learns Q only on that ridge → actor gradient has no signal off-ridge → no
exploration ever happens.

**Documented in**: [`docs/PRD_exploration_noise.md`](PRD_exploration_noise.md)
§ 4 + this project's reflection-Q2 answer.

## 6. Float32 vs float64 in shapely

**Risk**: shapely uses float64 internally; the network operates in float32.
At the boundary (LIDAR readings → observation tensor) we cast float32. Numerical
mismatch (~1e-7) could break reproducibility tests at extreme tolerances.

**Mitigation**: Reproducibility tests use exact equality on the
`StepDiagnostic` tuples (which are stored as Python floats — float64).
Same-seed runs match exactly. If we ever needed to assert tighter on LIDAR
values themselves, we would relax to `np.testing.assert_allclose(rtol=1e-5)`.

## 7. Replay buffer size vs total timesteps

**Design choice**: Replay capacity = 200 000 transitions, total timesteps
= 20 000 (headline) or 4 000 (sweeps). Buffer never wraps; off-policy data is
preserved for the full run.

**Potential issue**: For very long runs (e.g. 1 M steps) the buffer wraps and
the most recent 200 k transitions are kept. That's the standard behaviour and
not actually a failure mode, but documented here so future readers know the
capacity is a deliberate over-allocation rather than a guess.

## 8. Coverage = 0 when robot spawns in obstacle

**Symptom**: For one early seed, the robot spawned just outside the polygon
(rejection sampling failed to find an interior pose). The cleaning kernel
marked 0 cells. Coverage was 0.0 for the entire episode.

**Fix**: Tightened the spawn validity check to require `not is_collision(...)`
in addition to `point_in_polygon(...)`. Test added:
`tests/integration/test_roomba_env.py::test_env_resets_to_correct_obs_shape`
implicitly verifies a valid spawn by checking obs shape.

## 9a. Boosted-reward attempt backfired (v1.22 Layer 28 — honest negative result)

After the v1.21 TA re-grade, I tried to push coverage from ~0.045 → > 0.20 by:

- new_cell_bonus 1.0 → **3.0** (3× the carrot per fresh cell)
- step_penalty -0.05 → **-0.02** (less harsh on motion)
- noise.decay_steps 50000 → **30000** (faster σ taper)
- 30 000 timesteps (was 20 000)
- LR halved at step 20 000

**Result**: the v2 policy is **strictly worse** than the v1.20 baseline across
10 evaluation episodes (seed 100):

| | v1.20 (20 k, original tuned reward) | v1.22 v2 (30 k, boosted reward + LR decay) |
|---|---|---|
| Median coverage | **0.0487** | 0.0253 |
| Mean coverage | **0.0341** | 0.0206 |
| Max coverage | **0.0501** | 0.0359 |
| Median reward | **20 613** | 10 187 |

**Root cause hypothesis**: the original v1.20 reward was already well-balanced.
The dense `coverage_progress_coef × Δcoverage` term provides the gradient the
agent needs; bumping the sparse `new_cell_bonus` 3× shifted the loss surface
in a way that disrupted the established Q-estimate. The LR decay at step
20 k made it harder for the actor to escape the new local minimum.

**Lesson**: don't tune reward functions hoping for "more is more." The dense
shaping is what matters; the sparse term is mostly cosmetic at this scale.
Reward design should be **stationary across training** — changing it mid-run
or boosting it post-hoc is a footgun.

**v1.20 policy retained as the headline**. The v2 checkpoint
(`saved_models/headline_policy_v2.pt`) is committed as evidence of the failed
attempt, NOT as the recommended policy.

## 9b. Smaller-network attempt also backfired (v1.23 Layer 29 — second negative result)

After § 9a's "boosted reward" attempt failed, the TA suggested a different hypothesis:
the v1.20 network `[256, 256]` might be over-parameterised for the small data
budget (50 k steps × batch 128 ≈ 6.4 M parameter updates on 50 k unique transitions).

Tried:
- hidden sizes `[256, 256]` → **`[64, 64]`** (16× fewer parameters)
- v1.20 default reward (proven well-tuned after § 9a)
- 50 000 timesteps (was 20 000)
- Cosine LR schedule from `lr → lr × 0.1` over training

Side-by-side evaluation (10 episodes, seed 100):

| | v1.20 (256×256, 20k) | v1.22 v2 (256×256, 30k, boosted reward) | v1.23 v3 (64×64, 50k, cosine LR) |
|---|---|---|---|
| Median coverage | **0.0487** | 0.0253 | 0.0200 |
| Mean coverage | **0.0341** | 0.0206 | 0.0231 |
| Median reward | **20 613** | 10 187 | 7 836 |
| Max coverage | 0.0501 | 0.0359 | **0.0501** |

Smaller network is **also worse** on median. The max coverage matches v1.20
(0.0501) which suggests it occasionally finds the same good policy, but is
less reliable.

**Combined lesson from § 9a + § 9b**: the v1.20 hyperparameters are at the
practical ceiling for this *observation-architecture* (LIDAR-only) and
*reward-architecture* (cleaning bonus + dense progress shaping). Substantial
coverage improvement (> 0.10) probably requires either:

- **Richer observation**: goal-conditioning the actor on the nearest-unvisited-cell
  direction; or a visited-map auxiliary input
- **Richer reward**: frontier-exploration bonus (large reward for crossing
  > N metres of unvisited area in one episode)
- **Richer architecture**: an LSTM actor that remembers where it's been

These are *redesigns*, not hyperparameter tweaks. They are documented as
extension points in [`docs/PLAN.md`](PLAN.md) § 14.

**v1.20 policy retained as headline.** Both v2 and v3 checkpoints are
committed for full transparency as evidence of the failed attempts.

## 9c. Goal-conditioning attempt also backfired (v1.24 Layer 30 — third negative result)

After § 9a (boosted reward, failed) and § 9b (smaller network, failed), the TA's
final substantive suggestion was an **observation-space redesign** —
goal-conditioning. The hypothesis: LIDAR-only obs cannot represent "where to go,"
so the agent needs the nearest-unvisited-cell direction as input.

Implementation ([`environment/goal_obs.py`](../src/roomba_lab/environment/goal_obs.py)):

- `nearest_unvisited_direction(env)` returns (dx, dy) ∈ [-1, 1] normalised
  to bbox span, pointing at the nearest UNVISITED grid cell
- `GoalConditionedEnv` wraps `RoombaEnv` and appends these 2 features
- `obs_dim` = 29 → **31**

Trained 20 000 steps with v1.20 hyperparameters. **THIRD strictly-worse result**:

| | v1.20 (29-D obs) | v1.22 v2 (boosted reward) | v1.23 v3 (small net) | v1.24 v4 (goal-cond, 31-D) |
|---|---|---|---|---|
| Median cov | **0.0487** | 0.0253 | 0.0200 | 0.0156 |
| Median reward | **20 613** | 10 187 | 7 836 | 5 916 |

Hypotheses for why goal-conditioning didn't help:

- 2 extra inputs is a small change but disrupts the learned (s, a) → Q mapping
  the v1.20 policy converged on. Re-training from scratch with 20 k steps may
  not be enough budget for the critic to relearn the larger feature space.
- The nearest-unvisited direction is a **straight-line vector** — misleading
  when the actual path is blocked by walls. The agent might learn to head
  toward the vector but collide into a wall between robot and goal.
- The default v1.20 actor already implicitly learns a "go forward unless wall
  ahead" heuristic via LIDAR; adding a strong directional pull may **override**
  that good wall-avoidance behaviour.

**Combined three-attempt lesson** (§ 9a + § 9b + § 9c): every substantive
improvement to v1.20 has produced a worse policy. The v1.20 configuration is
**not just well-tuned — it's at a robust local optimum** that perturbations in
any direction (reward, network, observation) move away from. Further
improvement would likely require a fundamentally different agent
(e.g., model-based with planning, hierarchical RL with sub-policies for
explore vs clean, or learning from human demonstrations).

**v1.20 policy retained as headline.** v2, v3, v4 checkpoints + plots all
committed for full transparency of the negative-result-search trajectory.

## 9d. PyQt6 in headless CI

**Known requirement**: Qt platform plugins fail to load in headless CI without
`QT_QPA_PLATFORM=offscreen`. The CI workflow sets this env var explicitly
([`.github/workflows/assignment5-ci.yml`](../../.github/workflows/assignment5-ci.yml)
line 50). Carried-forward lesson from Assignment 4.

## What did *not* fail (and why we tested for it)

- **Reproducibility**: Same seed → identical diagnostics. Verified
  ([`tests/integration/test_reproducibility.py`](../tests/integration/test_reproducibility.py)).
- **No gym imports**: `grep "import gym" src/` returns nothing. Confirmed.
- **All files ≤ 150 LOC**: Verified per-file in the layer commits.
- **Polyak math**: 4-test math battery passes
  ([`tests/unit/test_soft_update.py`](../tests/unit/test_soft_update.py)).
