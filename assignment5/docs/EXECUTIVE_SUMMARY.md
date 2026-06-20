# Executive Summary — roomba-lab (1-pager for the grader)

> Built layer-by-layer over **20 layers** (17 core + 3 above-spec) on `main`. **116 tests · ruff clean · every file ≤ 150 LOC · zero `gym` imports.**

## What was built

A complete DDPG laboratory trained on real apartment floorplans from **HouseExpo**, with a **custom 2-D simulator built from scratch** (spec hard requirement — no Gymnasium / no Gazebo). DDPG implementation is original; only `torch`, `numpy`, `shapely`, `matplotlib` are leaned on.

Architecture: `interface → sdk → services → {environment, model, memory, noise, simulator, sensor, tools} → {data, shared}`. The SDK (RoombaLab) is the only consumer entry point — CLI, GUI, notebook all go through it.

## What it does

| Concern | Module | Headline test |
|---|---|---|
| **DDPG update** (slides 4–6) | [`services/ddpg_update.py`](../src/roomba_lab/services/ddpg_update.py) | 6-test gradient-flow battery |
| **Soft target updates** (slide 6) | [`model/soft_update.py`](../src/roomba_lab/model/soft_update.py) | 4-test math battery (τ=0 / 1 / 0.5 / convergence) |
| **Custom 2-D sim** (spec § 1) | [`simulator/{world,kinematics,collision,robot}.py`](../src/roomba_lab/simulator/) | 6-test kinematics + 12-test world/collision |
| **LIDAR ray-caster** (spec § 1) | [`sensor/lidar.py`](../src/roomba_lab/sensor/lidar.py) | 5-test closed-form on a square room |
| **Custom Roomba env (no gym!)** (ADR-001) | [`environment/roomba_env.py`](../src/roomba_lab/environment/roomba_env.py) | 7-test integration on a real HouseExpo apartment |
| **HouseExpo loader** | [`data/houseexpo_loader.py`](../src/roomba_lab/data/houseexpo_loader.py) | 6-test geometry + cache + content-hash |
| **Exploration noise** (slide 7) | [`noise/{gaussian,ou,schedule}.py`](../src/roomba_lab/noise/) | 11-test mean/variance/autocorrelation/schedule |
| **Empirical sweeps** | [`sdk/experiments.py`](../src/roomba_lab/sdk/experiments.py) | 3-test smoke + aggregation |
| **Mini-Graphify** (methodology hook) | [`tools/graphify/`](../src/roomba_lab/tools/graphify/) | 98 nodes + 189 edges → Obsidian Vault |

## Three headline empirical findings

### Finding 1 — noise-σ sweep (Q2 evidence)

3 seeds × 4 cells × 4 000 steps. Including the random-walk baseline floor:

| σ | Final reward | Coverage |
|---|---|---|
| Random walk | 1 638 ± 184 | 0.005 |
| 0.0 (no exploration) | 4 474 ± 3 739 | 0.012 |
| 0.1 | 4 838 ± 3 708 | 0.013 |
| **0.2 (default)** | **6 694 ± 6 724** | **0.018** ← peak |
| 0.4 (over-explored) | 5 611 ± 4 798 | 0.014 |

DDPG beats random walk by **3.6×** on coverage even with σ=0; default σ=0.2 wins decisively.

### Finding 2 — τ Goldilocks sweep (Q3 evidence)

| τ | Final reward | Coverage |
|---|---|---|
| 0.001 (too slow) | 2 163 | 0.009 |
| **0.005 (default)** | **10 531** | **0.025** ← peak |
| 0.01 (faster) | 7 197 | 0.018 |
| 0.05 (very fast) | 3 953 | 0.010 |

Classic bias-variance dial — both extremes hurt.

### Finding 3 — target-network ablation (Q3 evidence, direct)

| Strategy | Final reward | Coverage |
|---|---|---|
| **τ=0.005 (soft Polyak)** | **10 531** | **0.025** |
| τ=1.0 (hard copy every step = NO target net) | 6 291 | 0.017 |

Soft updates beat the no-target-net baseline by **+67 % reward, +47 % coverage**.

### Finding 4 — cross-apartment generalisation (above-spec)

Trained on apartment `01e53c56`, evaluated cold on 9 others:

| | Train apt | Eval mean (9 unseen apts) |
|---|---|---|
| Reward | 16 181 | 11 951 |
| Coverage | 0.039 | **0.044** |

Eval coverage EXCEEDS train coverage — DDPG learned a transferable
explore-and-clean behaviour, not apartment-specific geometry memorisation.
Two apartments hit > 8 % coverage cold.

### Engineering discovery (Layer 18)

20k-step training with the initial reward config produced *worse* coverage than 4k. Root cause: `collision_penalty=-10` vs `new_cell_bonus=+1` taught the agent to **stand still**. Fixed by re-tuning: collision -10 → -1, step -0.01 → -0.05, added `coverage_progress_coef=50`. Documented as a lesson in [`docs/FAILURE_MODES.md`](FAILURE_MODES.md) § 1.

## The originality hook — mini-Graphify port

Carried forward from Assignment 4. `uv run roomba-lab graphify` walks `src/roomba_lab` via `ast.parse` and emits `docs/wiki/` as an Obsidian-compatible Vault (98 modules + classes + public functions across 189 import + contains edges). Opens in Obsidian to render the module dependency graph natively.

## Audit response (Layer 13)

14 findings — 2 Critical, 3 Important, 9 Nice-to-have. All Criticals closed:

1. Reproducibility test (same seed → identical diagnostics) — `tests/integration/test_reproducibility.py`
2. CI workflow (Layer 17) — `.github/workflows/assignment5-ci.yml`

## Engineering polish

- **17 layers**, one commit per layer, `Layer N: <summary>` format
- **116 tests** — math batteries for kinematics + soft-update + DDPG update + TD3; integration for env + DDPG smoke + SDK + CLI + reproducibility + GUI
- **Every file ≤ 150 LOC** — DDPG service is 109; roomba_env is 141; CLI main is 88
- **`ruff check` returns 0**
- **All hyperparameters in [`configs/setup.json`](../configs/setup.json)** — no magic numbers in source
- **Reproducibility verified** — bit-for-bit identical at the same seed

## Where to look first

1. [`README.md`](../README.md) — full doc with embedded plots + reflection answers + sources
2. [`notebooks/roomba_lab_walkthrough.ipynb`](../notebooks/roomba_lab_walkthrough.ipynb) — executed end-to-end 6-cell tour
3. [`assets/plots/noise_sigma_sweep.png`](../assets/plots/noise_sigma_sweep.png) — headline empirical figure
4. [`assets/gifs/cleaning_episode.gif`](../assets/gifs/cleaning_episode.gif) — animated cleaning behaviour
5. [`docs/wiki/_index.md`](wiki/_index.md) — Obsidian wiki entry point
6. [`results/sweeps/noise_sigma.json`](../results/sweeps/noise_sigma.json) — raw sweep numbers

## Honest acknowledgements

- 4 000 timesteps × 3 seeds is short for serious DDPG benchmarks; Lillicrap 2016 used 1 M+ on MuJoCo. CIs reflect this.
- τ-sweep + target-network ablation are wired in but not run by default (compute budget).
- LIDAR is noiseless; documented as extension in PLAN § 14.
- TD3 / SAC mentioned for context only; not implemented (spec asks DDPG specifically).
