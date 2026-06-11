# PRD — PPO + GAE Laboratory on MuJoCo Continuous Control (Assignment 4)

> **Course:** Reinforcement Learning with Deep Learning
> **Lecture:** L08 — REINFORCE → TRPO → PPO + GAE (Dr. Yoram Segal, May 2026)
> **Author:** Shaked Kozlovsky (ID 208904839)
> **What this project IS:** a from-scratch implementation of **PPO** (Schulman et al. 2017) and **Generalized Advantage Estimation** (Schulman et al. 2016) trained on **MuJoCo continuous-control benchmarks** (HalfCheetah-v4, Walker2d-v4), with a **mini-Graphify tool** that auto-generates an Obsidian-compatible knowledge graph from the project's own Python AST. The project is organised under the *Active Knowledge Architecture* methodology (Obsidian + Graphify + AI Skills).
> **What this project is NOT:** a pretrained-model wrapper, a wrapper around Stable-Baselines3, or a RLHF system. The algorithm code is original; the only RL libraries we depend on are `gymnasium[mujoco]` and `mujoco` for environments.

---

## 0. PRD evolution log

| Version | Layer | What changed and why |
|---|---|---|
| v1 (Layer 0) | Planning | Initial requirements from the L08 slides + Active Knowledge Architecture methodology document. |

---

## 1. Project goal

Demonstrate the **PPO + GAE pipeline** end-to-end on canonical continuous-control benchmarks, with three layers of rigor:

1. **Algorithm correctness** — verbatim implementation of the clipped-surrogate objective (Eq. 1 below) and the GAE recursion (Eq. 2).
2. **Empirical analysis** — sweeps over the three key hyperparameters (`λ`, `γ`, `clip-ε`) with multi-seed 95 % confidence intervals, demonstrating the **bias-variance ladder** GAE creates (slide 16).
3. **Engineering quality** — same standards as Assignment 3 (≤ 150 LOC per file, ≥ 85 % coverage, ruff clean, `Layer N: <summary>` commits, layered architecture with strict dependency direction).

The grade rewards understanding of *why* PPO works (Trust Region intuition, no hard KL), *why* GAE works (geometric weighting of TD errors as a bias-variance dial), and honest analysis of where each method's assumptions hold. A clean "PPO solves the task" outcome is not the point — the *empirical proof of the slide claims* is.

## 2. Mapping to the L08 lecture slides

The slide deck is Hebrew titled *"הרצאה 7: אופטימיזציית מדיניות מקורבת (PPO) ושיטת GAE"* (lecture-7 by file naming but L08 in the course sequence). The slide numbers below refer to the GAE.pdf I read.

| Slide | Concept | Where it lives in the code |
|---|---|---|
| 3 | Policy Gradient is the starting point — `L^PG(θ) = E[log π_θ(a|s) · Â]` | `services/ppo_service.py::compute_pg_loss` (called as the baseline before clipping) |
| 4 | On-Policy is sensitive to update size | `services/ppo_service.py` enforces `n_epochs_per_update` and `minibatch_size` from config |
| 5 | Policy Collapse failure mode | `services/diagnostics.py::policy_kl_monitor` tracks per-update KL — flags collapse early |
| 6 | Trust Region (TRPO): bound KL between old and new policy | `PRD_ppo.md` discusses TRPO as motivation; not implemented |
| 7 | TRPO accuracy vs cost (Fisher matrix) | Same |
| 8 | **PPO simplifies Trust Region** via clipping | `services/ppo_service.py::ppo_update` |
| 9 | **Probability Ratio** `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` | Computed inside `ppo_update`; logged for diagnostics |
| 10 | **Clipped Objective** `L^CLIP = E[min(r·Â, clip(r, 1-ε, 1+ε)·Â)]` | The headline equation; verbatim in code |
| 11 | Positive advantage case — strengthen action carefully | Unit test asserts the clipped branch wins when `r > 1+ε` and `Â > 0` |
| 12 | Negative advantage case — weaken action carefully | Symmetric test |
| 13 | PPO is "deliberately pessimistic" — picks the more conservative branch | Discussed in `PRD_ppo.md` § "Honest acknowledgement" |
| 14 | **Advantage quality determines PPO stability** | This motivates Layer 4's GAE implementation |
| 15 | **TD error** `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)` is a 1-step advantage | `services/gae.py::td_errors` |
| 16 | **GAE recursion** `Â_t^GAE(γ,λ) = Σ (γλ)^l · δ_{t+l}` and the bias-variance dial via λ | `services/gae.py::compute_gae` — the headline equation |
| 17 | Actor-Critic combined framework | `model/actor_critic_network.py` |
| 18 | PPO training pipeline (rollout → GAE → mini-batch PPO update → repeat) | `services/ppo_service.py::fit` |
| 19 | PPO succeeds on MuJoCo (HalfCheetah, Hopper, Walker2d) | Our chosen environments |
| 20 | PPO extends to RLHF + robotics | Discussed as future work; not implemented |
| 21 | Three stability layers of PPO: Proximal updates · Advantage quality · Practical pipeline | All three asserted by tests |

## 3. The two core equations (verbatim from the slides)

### PPO clipped surrogate (slide 10)

```
L^CLIP(θ) = Ê_t[ min( r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]                  (1)

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

### GAE advantage (slide 16)

```
δ_t^V = r_t + γ·V(s_{t+1}) − V(s_t)                                                (2a)
Â_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}^V                                      (2b)

equivalently, the reverse-recursion form used in code:
Â_t = δ_t + γλ · Â_{t+1}                                                            (2c)
```

The full PPO loss combines the clipped surrogate with a value-function loss and an entropy bonus:

```
L(θ) = −L^CLIP(θ) + c_1 · L^VF(θ) − c_2 · H[π_θ]                                   (3)
```

These three equations are the *spine* of the project — every test, every plot, and every reflection answer references them.

## 4. Environments (MuJoCo continuous control)

| Environment | Observation dim | Action dim | Episode max steps |
|---|---|---|---|
| **HalfCheetah-v4** (primary) | 17 | 6 | 1000 |
| **Walker2d-v4** (secondary) | 17 | 6 | 1000 |

Both are torque-controlled, dense-reward continuous-control benchmarks. They are the canonical PPO benchmarks (slide 19) and let us empirically validate the bias-variance λ-sweep claim (slide 16) on two different dynamics. Action space is `Box(-1, 1, (6,))`, observation space is `Box(-inf, inf, (17,))`.

**Why not Atari?** Discrete actions would not exercise the "PPO for continuous control" claim. Assignment 3 already had discrete actions.

**Why not Hopper?** Identical class to Walker2d (planar bipedal). HalfCheetah + Walker2d span two different morphologies — quadruped vs biped — for diversity.

## 5. Network architecture

| Component | Definition |
|---|---|
| **Actor** | MLP `obs_dim → 64 → 64 → action_dim` with `tanh` activations. Outputs mean of a Gaussian; log-std is a learned **state-independent** parameter vector (the standard SB3 / RLLib convention). |
| **Critic** | MLP `obs_dim → 64 → 64 → 1` with `tanh` activations. Returns scalar value estimate. |
| **Shared trunk** | Off by default (separate networks for actor and critic). Toggleable via `configs/setup.json:actor_critic.shared_trunk` for an ablation. |
| **Action sampling** | `a_t ~ Normal(mu(s_t), exp(log_std))`. `log_std` clamped to `[−5, +2]` to prevent collapse / explosion. |

## 6. Hyperparameters (from `configs/setup.json`)

| Parameter | Value | Source |
|---|---|---|
| γ (discount) | 0.99 | Standard PPO + GAE choice (Schulman 2017) |
| λ (GAE) | 0.95 | Standard PPO + GAE choice (Schulman 2016) |
| Clip ε | 0.2 | Slide 10 default |
| Total timesteps | 1 000 000 | Sufficient for HalfCheetah to plateau; budget verifiable in [`docs/TODO.md`](TODO.md) |
| Rollout length | 2048 steps | Standard PPO config |
| Minibatch size | 64 | Standard PPO config |
| Epochs per update | 10 | Standard PPO config (slide 8) |
| Actor + critic lr | 3e-4 | Shared optimizer, Adam, no warmup |
| Value coef `c_1` | 0.5 | Standard |
| Entropy coef `c_2` | 0.0 | Continuous control rarely needs it; left as ablation variable |
| Max grad norm | 0.5 | Stability — clip global grad norm |

## 7. Empirical study plan (the headline analyses)

Three sweeps, each cross-validated across 3 seeds with 95 % CIs:

### 7.1 The λ-sweep (Layer 10) — *headline*

Sweep `λ ∈ {0.0, 0.5, 0.9, 0.95, 0.99, 1.0}` on HalfCheetah-v4, holding everything else constant. The slide-16 prediction:
- λ → 0: low variance, high bias (TD-only)
- λ → 1: high variance, low bias (Monte-Carlo)
- Middle range (0.9–0.99): sweet spot

The headline plot shows `final_reward(λ)` with a clear inverted-U or monotone shape — empirical proof of the bias-variance dial.

### 7.2 The γ-sweep (Layer 10)

`γ ∈ {0.9, 0.95, 0.99, 0.999}`. Lower γ → shorter effective horizon → less bootstrap noise but reduced credit assignment. Expect monotone or peaked.

### 7.3 The clip-ε sweep (Layer 10)

`clip_eps ∈ {0.1, 0.2, 0.3, 0.5}`. The "Trust Region size" knob. Smaller ε = more conservative = more stable but slower. Larger ε = more aggressive = faster but risks policy collapse (slide 5).

### 7.4 Cross-environment generalisation (Layer 11)

Run the best (γ, λ, ε) trio from § 7.1–7.3 on **Walker2d-v4** and compare. Does the optimum transfer? If yes → algorithm robustness; if no → environment-specific tuning matters.

## 8. The mini-Graphify tool (the methodology differentiator)

The Active Knowledge Architecture document positions Graphify as the "computation engine" that converts raw files → Knowledge Graph → Wiki Markdown → Obsidian Vault. Implementing a small version of this directly **inside the project** is the originality hook.

**What it does** (Layer 9):
1. Walk `src/proximal_lab/` with `ast.parse`.
2. Extract: module dependencies, class definitions, public functions, method-call edges.
3. Emit:
   - `docs/wiki/graph.json` — the knowledge graph (nodes = symbols, edges = imports + calls).
   - `docs/wiki/<module>.md` — one Markdown file per module with Wikilinks to its dependencies.
4. The output folder is an **Obsidian Vault** — opening `docs/wiki/` in Obsidian shows the project's structure as a graph.

This is *not* a serious replacement for Graphify itself; it's a focused implementation of the same idea, applied to our own code. The lecturer's methodology becomes auditable: "show me how the project's modules depend on each other" → run `proximal-lab graphify`.

## 9. KPIs and acceptance criteria

| KPI | Target | How verified |
|---|---|---|
| Test coverage | ≥ 85 % branch | `pytest --cov` gate in `pyproject.toml` |
| `ruff check` | 0 errors | CI-style local invocation |
| File size | ≤ 150 LOC per `.py` | Self-policed; exceptions documented in PLAN.md |
| PPO clip-objective math | Verbatim Eq. (1) | `tests/unit/test_ppo_clip_math.py` |
| GAE recursion | Verbatim Eq. (2c) | `tests/unit/test_gae.py` — closed-form check against a known δ sequence |
| HalfCheetah-v4 convergence | Episode reward > 1500 by 500k steps | The standard PPO benchmark; slide-19 territory |
| λ-sweep monotonicity | Reproducible peaked or monotone curve | `assets/plots/lambda_sweep.png` + JSON in `results/` |
| Reproducibility | Same seed → bit-identical training history | `tests/integration/test_reproducibility.py` |
| Graphify output | `docs/wiki/graph.json` opens in Obsidian; nodes = modules, edges = imports | Visual inspection + JSON-schema test |
| Multi-seed CI | 3 seeds per cell, 95 % normal-approx CIs | `sdk/experiments.py::aggregate_with_ci` |
| Architecture purity | No model/service logic in GUI/CLI; SDK is the only consumer entry | Code review + import graph |

## 10. Non-functional requirements

- **OOP-first**: every responsibility is a class or a focused module-level function.
- **Configuration**: no hardcoding; all hyperparameters in `configs/setup.json`.
- **`uv` is the only package manager.**
- **TDD pairs**: PPO clip math + GAE recursion both written test-first (Red → Green → Refactor).
- **Logging**: through `shared/logger.py`; no `print` in library code.
- **No internet at evaluation time**: MuJoCo runs entirely locally, no API calls.

## 11. Deliverables

- Source under `src/proximal_lab/`
- Tests under `tests/` with ≥ 85 % coverage
- `docs/` with PRD, PLAN, TODO, per-mechanism PRDs (PPO, GAE, ActorCritic, MuJoCo env, Graphify)
- `configs/setup.json`
- `assets/` with screenshots, training plots, comparison charts
- `results/` with experiment outputs (`lambda_sweep.json`, `gamma_sweep.json`, `clip_eps_sweep.json`, `cross_env.json`)
- `notebooks/proximal_lab_walkthrough.ipynb` — end-to-end guided tour
- README at repo root with formulas, plots, reflection answers, audit response
- Pre-trained checkpoints in `saved_models/`
- `docs/wiki/` Obsidian Vault generated by the mini-Graphify tool

## 12. Out of scope

- RLHF (mentioned in slide 20 but a multi-week project on its own)
- TRPO (mentioned in slides 6–7 as motivation; not implemented)
- Vectorised env via shared memory (we use `gymnasium.vector.SyncVectorEnv`)
- Distributional value functions
- Curriculum learning across multiple environments

## 13. Honest acknowledgement

- The slide deck is open-ended ("think about extensions, analyses, and originality"); the domain choice (MuJoCo continuous control) is mine, not specified.
- The Active Knowledge Architecture document is a *methodology* the lecturer recommends, not a topic spec. Implementing the mini-Graphify tool is my interpretation of "originality"; the lecturer may have intended something else.
- Numbers reported in the README come from a 1M-timestep run on a CPU; longer runs would tighten the CIs further but not change the qualitative conclusions.

## 14. Sources

The full bibliography lives in [`docs/PRD_ppo.md`](PRD_ppo.md) § Sources and [`docs/PRD_gae.md`](PRD_gae.md) § Sources. The headline papers are:

- **PPO**: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- **GAE**: J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR, 2016.
- **TRPO** (background): J. Schulman et al., "Trust Region Policy Optimization," ICML, 2015.
- L08 lecture slides — *הרצאה 7: אופטימיזציית מדיניות מקורבת (PPO) ושיטת GAE* (Dr. Yoram Segal, May 2026).
- *Active Knowledge Architecture* — methodology document (Dr. Yoram Segal, May 2026; NotebookLM).
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
