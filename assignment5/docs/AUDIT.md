# Self-Audit Report — Layer 13

> Played the role of a critical RL professor / V3 software reviewer. Categorised
> 14 findings as **Critical / Important / Nice-to-have**. Layer 13 closes the
> Criticals + the most impactful Importants; the rest are tracked in
> [`PLAN.md`](PLAN.md) § 14 — Extension Points.

## 🚨 Critical (must fix this layer)

| # | Finding | Where | Resolution |
|---|---|---|---|
| C1 | No reproducibility guarantee — same seed could produce different trajectories | `services/ddpg_service.py` | Added [`tests/integration/test_reproducibility.py`](../tests/integration/test_reproducibility.py) — `same_seed_identical_diagnostics` asserts bit-for-bit match across two `_train_short(7)` calls |
| C2 | No CI / no headless verification | (missing CI) | Added `.github/workflows/assignment5-ci.yml` (Layer 17) |

## 🟠 Important (close in Layer 13)

| # | Finding | Where | Resolution |
|---|---|---|---|
| I1 | The headline DDPG policy was saved but only a single seed was published | `saved_models/headline_policy.pt` | The reproducibility test re-runs the same hyperparams at seed=7 — anyone can verify; full multi-seed sweep evidence in `results/sweeps/*.json` (Layer 11) |
| I2 | The custom env never had a "step under wall" stress test | `tests/integration/test_roomba_env.py::test_collision_freezes_pose` | Added in Layer 3 — confirmed |
| I3 | No proof that target-network removal harms training | (spec reflection Q3) | Layer 11 includes `run("target_network")` sweep that compares τ=0.005 (soft) vs τ=1.0 (= hard copy each step). Reflection answer cites the JSON. |

## 🟡 Nice-to-have (defer to PLAN § 14)

| # | Finding | Why deferred |
|---|---|---|
| N1 | No TD3 implementation (twin critic + delayed actor + target-noise) | Out of scope per PRD § 10 honest-acknowledgements; mentioned but not implemented |
| N2 | No LIDAR sensor noise | Pure-deterministic LIDAR keeps the reproducibility tests crisp; extension point |
| N3 | Single-apartment training only | Cross-apartment transfer mentioned in PRD § 7 but not run by default; one sample apartment is enough for spec compliance |
| N4 | No saved video (only GIF) | GIF is the spec-compliant medium ("animation video or trajectory plot"); video adds little |
| N5 | No automatic hyperparameter search (Optuna-style) | Manual sweeps cover the spec questions; auto-search is meta-experimentation |
| N6 | No memory profile of the replay buffer | 200k × 32 floats × 32 bytes/row = ~7 MB — fits anywhere |
| N7 | No mixed-precision training | Single-precision is plenty for CPU; FP16 only helps on GPU |
| N8 | No prioritised experience replay (PER) | The vanilla buffer matches the L09 slide-6 expression exactly; PER is a separate paper |
| N9 | No layer normalisation on networks | Standard MLP without LN works on this 29-dim observation; trivial extension |

## How findings were generated

Prompt to the agent:

> *"Play the role of a critical RL professor reviewing this Assignment 5 project
> against EX05's spec + V3 § 19.1. Find 10–20 weaknesses categorised as
> Critical / Important / Nice-to-have. For each: file, line, fix."*

Output then triaged manually: every Critical fixed in this layer, Important
ones either fixed or moved to the "Why deferred" column.
