# PRD — Partial observation (Manhattan-radius local view)

> Per-mechanism PRD for `src/marl_lab/sensor/partial_observation.py`. L10 § 2 + the Dec-POMDP `Ω, O` elements.

## 1. The model

Each agent sees ONLY cells within **Manhattan distance r** of its position. `r` is configurable (`game.observation_radius`, default 2). L10 page 8 has the canonical diagram: cop at `(2, 2)` with `r = 1` sees a 3×3 patch; the thief — even at `(3, 4)` — is hidden if its Manhattan distance to the cop exceeds `r`.

## 2. The observation vector

For agent i at position `(x, y)`:

```
obs_i = concat[
    flatten( cells_in_radius(state.grid, x, y, r) ),  # walls + barriers + this agent + opponent (if visible)
    [x_norm, y_norm],                                  # this agent's normalised position
    [steps_remaining / max_moves],                     # global step counter (this IS available — it's clock, not state)
    [barriers_remaining / max_barriers],               # only cop knows? — see § 4
    one_hot(self_role),                                # 1 if cop, 0 if thief
]
```

Dimensionality: depends on `r` and grid size:

| r | Visible cells | obs_dim |
|---|---|---|
| 1 | 9 (3×3 patch) | 9 + 5 = 14 |
| 2 | 13 (Manhattan-2 diamond) | 13 + 5 = 18 |
| 3 | 25 (Manhattan-3 diamond) | 25 + 5 = 30 |

Each cell encodes 4 bits: `{empty, wall, barrier, opponent_visible}`. The opponent-visible bit ONLY fires if the opponent is within radius r.

## 3. Cell-encoding choice

| Encoding | Pros | Cons | Choice |
|---|---|---|---|
| One-hot per cell (4 channels × N cells) | clean | high-dim, sparse | **✓ default** |
| Single int per cell | low-dim | embedding required downstream | no |
| Image-style (C × H × W tensor) | CNN-friendly | overkill for ≤ 25 cells | no (we use MLP/GRU) |

## 4. Edge cases

| Case | Behaviour |
|---|---|
| Opponent outside radius | `opponent_visible` = 0 everywhere in patch |
| Opponent inside radius | bit set on the cell where opponent is |
| Cell outside grid (radius extends past boundary) | encoded as a special "wall" / "off-grid" value |
| Barrier on the agent's own cell | impossible (barrier placement excludes own cell, per § 3.3 of spec) |
| Barriers visible to thief? | Yes — barriers are static and physically present. Thief sees barriers within radius. |
| Barriers-remaining info | Only the cop's observation includes the `barriers_remaining` count. Thief sees 0. |

## 5. Test plan

| Test | Pass criterion |
|---|---|
| Manhattan radius r=1, agent at (2,2) on 5x5 | exactly 9 cells visible |
| Manhattan radius r=2, agent at (2,2) on 5x5 | exactly 13 cells visible (diamond) |
| Opponent at (0,0), agent at (4,4), r=2 | `opponent_visible` = 0 everywhere |
| Opponent at (2,3), agent at (2,2), r=1 | `opponent_visible` bit set at (2,3) |
| Off-grid cell encoding | special wall value |
| Barriers visible | both agents see barriers within radius |
| obs_dim matches table § 2 | exact integer match |

## 6. Acceptance criteria

1. `observe(state, agent_id, radius) → np.ndarray` is pure.
2. obs_dim is deterministic given (radius, grid_size) — config-driven, not hard-coded.
3. The L10 diagram (cop sees 3×3, thief hidden) is reproducible as a unit test.
4. Agent-role one-hot included so a single Q-net family can serve both agents (parameter sharing optional but available).

## 7. Non-goals

- LIDAR-style ray-cast (`marl_lab` is grid-based, not continuous)
- History of past observations as input (the GRU does that, slide 5)
- Multi-modal observations (images / audio)

## 8. Why this matters for the analysis

The partial-observation horizon `r` is **the** parameter that controls Dec-POMDP difficulty:

- r = ∞ → fully observable → Markov game (much easier)
- r → 0 → blind agent → near-impossible

The spec § 5.1 Table 2 staged validation explicitly studies the influence of `r` ("Affect of partial observation; LIDAR-radius influence"). The Layer 21 `observation_radius_sweep` (`r ∈ {1, 2, 3}` × 3 seeds) is the empirical contribution.
