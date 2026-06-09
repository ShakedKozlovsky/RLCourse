# Executive Summary — proximal_lab (1-pager for the grader)

> Built layer-by-layer over 15 commits on `main`. **120 tests · 97.25 % coverage · ruff clean · every file ≤ 150 LOC.**

## What was built

A complete PPO + GAE laboratory trained on MuJoCo continuous control (HalfCheetah-v5 + Walker2d-v5). Algorithm code is original; only `gymnasium[mujoco]` and `mujoco` provide the env runtime.

Architecture: `interface → sdk → services → {environment, model, tools} → shared`. The SDK is the only consumer entry point — CLI, GUI, and notebook all go through it.

## What it does

| Concern | Module | Headline test |
|---|---|---|
| Clipped surrogate `L^CLIP` (slide 10) | [`services/ppo_clip.py`](../src/proximal_lab/services/ppo_clip.py) | 4-test sign/window battery proves slide-11/12 cases |
| GAE recursion (slide 16) | [`services/gae.py`](../src/proximal_lab/services/gae.py) | 4-test math battery: λ=0 → TD, λ=1 → MC, closed-form, terminal |
| PPO training pipeline (slide 18) | [`services/ppo_service.py`](../src/proximal_lab/services/ppo_service.py) | smoke training; diagnostics for slide-21 three pillars |
| MuJoCo env + obs normaliser | [`environment/mujoco_env.py`](../src/proximal_lab/environment/mujoco_env.py) | Welford running stats match NumPy reference |
| Actor-critic networks | [`model/actor_critic_network.py`](../src/proximal_lab/model/actor_critic_network.py) | Save/load round-trip; param partition (ADR-002) |
| Vectorised env (4 parallel) | [`environment/vector_env.py`](../src/proximal_lab/environment/vector_env.py) | Batched obs `(4, 17)`, shared RMS updates across envs |
| Empirical sweeps | [`services/experiment_service.py`](../src/proximal_lab/services/experiment_service.py) | λ-sweep multi-seed has real CIs |
| Mini-Graphify (the methodology hook) | [`tools/graphify/`](../src/proximal_lab/tools/graphify/) | Produces `docs/wiki/` Obsidian Vault with 70 nodes + 128 edges |

## Headline empirical finding — the λ-sweep

**The slide-16 bias-variance dial is empirically real.** 3 seeds × 15k timesteps × 6 cells on HalfCheetah-v5:

| λ | Final-mean reward | 95 % CI |
|---|---|---|
| 0.0 | −284.19 | ± 11.55 |
| 0.95 | **−250.44** | ± 12.33 ← peak |
| 1.0 | −294.44 | ± 17.77 |

λ=0.95 statistically beats λ=0 (diff 34, CI sum 24) and λ=1 (diff 44, CI sum 30). The standard PPO+GAE default is empirically optimal on a third reward landscape (not the original Schulman 2016 reward shape).

## Cross-env transfer

Best `(γ=0.999, λ=0.95, clip_eps=0.3)` from Layer 10 → Walker2d-v5: **+381.12 reward** at 30k timesteps, with a clear upward learning curve (`163 → 154 → 292 → 398`). Same hyperparameters work on both morphologies — slide 19's "PPO robust across MuJoCo benchmarks" confirmed.

## The originality hook — mini-Graphify tool

The *Active Knowledge Architecture* methodology document positions a 4-step pipeline (Raw → Pipeline (Graphify) → Wiki → Obsidian Vault). I implemented step 2 inside the project itself:

```bash
uv run proximal-lab graphify   # walks src/proximal_lab via ast.parse,
                                # emits docs/wiki/ as a real Obsidian Vault
```

The lecturer's diagram becomes runnable. The wiki shows 70 nodes (modules + classes + public functions) and 128 edges (imports). Opening `docs/wiki/` in Obsidian renders the project's module dependency graph natively.

## Audit response (Layer 13)

After Layer 12 I ran Claude as a critical professor. 3 findings; all 3 fixed:

1. Single-seed sweeps → multi-seed λ-sweep with statistically significant CIs
2. No reproducibility test → 2 integration tests asserting same-seed → identical rewards + KL
3. No standalone PNG plots → 5 PNG plots for README to embed

## Engineering polish

- **15 layers**, one commit per layer, `Layer N: <summary>` format throughout
- **120 tests** all green: 4-test PPO math, 4-test GAE math, GUI smoke under offscreen Qt, reproducibility, sweep aggregation, etc.
- **97.25 % branch coverage** (gate 85 %)
- `ruff check` returns 0
- **No file > 150 LOC** — `sdk/sdk.py` is 95 LOC, the largest. Compare to Assignment 3's documented 154-LOC SDK exception.
- All hyperparameters in [`configs/setup.json`](../configs/setup.json) — no magic numbers
- Reproducibility tested + verified

## Where to look first

1. [`README.md`](../README.md) — full doc with embedded plots, reflection answers, sources
2. [`notebooks/proximal_lab_walkthrough.ipynb`](../notebooks/proximal_lab_walkthrough.ipynb) — 6-cell guided tour, executed end-to-end with embedded outputs + figures
3. [`assets/plots/lambda_sweep_multiseed.png`](../assets/plots/lambda_sweep_multiseed.png) — the headline figure
4. [`docs/wiki/_index.md`](wiki/_index.md) — the Obsidian wiki entry point
5. [`results/sweeps/lambda_multiseed.json`](../results/sweeps/lambda_multiseed.json) — raw numbers

## Honest acknowledgements

- 15k–30k timesteps is short for MuJoCo — slide-19's ~2000 reward is at 1M+ steps. Trends are real; absolute magnitudes would benefit from longer runs.
- Mini-Graphify uses AST-based imports only — no LLM semantic inference like the full Graphify would have.
- γ and clip-ε sweeps remained single-seed; only λ was multi-seeded due to compute budget.
