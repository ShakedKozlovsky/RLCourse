# Product Requirements Document — roomba-lab

> **Assignment 5 (June 2026)** — *תרגיל 05 — פיתוח סימולטור רובוט מנקה ויישום DDPG*. Dr. Yoram Segal, RL Course, L09 (DDPG).

## 1. Project goal

Build a complete laboratory for **DDPG on continuous robot control** by:

1. Implementing a **custom 2-D cleaning-robot simulator from scratch** (NO Gymnasium, NO Gazebo — explicitly forbidden by the spec).
2. Running it on **real apartment floorplans from the HouseExpo dataset** [1].
3. Training a **DDPG agent** with actor + critic + soft-target updates + Gaussian exploration noise.
4. Producing empirical evidence that DDPG is the right tool for this continuous-action domain (vs DQN-style discretisation, vs PPO).

The central pedagogical claim under test:

> *L09 slide 9 — "**DDPG: deterministic actor-critic, continuous action space, off-policy, max-Q replay**." This is the only mainstream algorithm in the Table-1 family with the (continuous + deterministic) combination, and the only one that exploits stored data efficiently in this regime.*

## 2. Mapping from L09 lecture slides → code locations

| Slide / chapter | Concept | Where it lives in `roomba-lab` |
|---|---|---|
| § 2 Table 1 — algorithm evolution | DDPG row: continuous-only, deterministic actor-critic, Q-value foundation | [`PRD_ddpg.md`](PRD_ddpg.md), [`services/ddpg_service.py`](../src/roomba_lab/services/ddpg_service.py) |
| § 3 — discretisation explosion | Why we cannot reduce continuous control to DQN tables | README § 5 + `docs/PRD_ddpg.md` § 2 |
| § 4 — DPG gradient ∇θ J = E[∇θ μ · ∇a Q] | Actor gradient via critic chain rule | [`services/ddpg_update.py::actor_loss`](../src/roomba_lab/services/ddpg_update.py) |
| § 5 — DDPG architecture: μ(s\|θ^μ), Q(s,a\|θ^Q), tanh output | Actor + Critic networks | [`model/actor.py`](../src/roomba_lab/model/actor.py), [`model/critic.py`](../src/roomba_lab/model/critic.py) |
| § 6 — Soft target updates: θ' ← τ·θ + (1−τ)·θ' | Polyak averaging | [`model/soft_update.py::polyak_update`](../src/roomba_lab/model/soft_update.py) |
| § 7 — Exploration noise (Gaussian, OU) | `noise/` module | [`noise/gaussian.py`](../src/roomba_lab/noise/gaussian.py), [`noise/ou.py`](../src/roomba_lab/noise/ou.py) |
| § 8 — Training loop, transfer from simulator | DDPG training service | [`services/ddpg_service.py`](../src/roomba_lab/services/ddpg_service.py) |
| § 10 — Practical task (HouseExpo cleaning robot) | The entire project | This PRD |

## 3. The two core equations (verbatim — slide 4 + 6)

### 3.1 Deterministic policy gradient theorem (Silver et al. 2014; slide 4)

$$\nabla_{\theta}\, J(\mu_{\theta}) \;=\; \mathbb{E}_{s\sim\rho^{\mu}}\!\left[\, \nabla_{\theta}\mu_{\theta}(s)\, \nabla_{a} Q^{\mu}(s, a)\big|_{a=\mu_{\theta}(s)} \,\right]$$

This is implemented as the **actor loss**: `−mean( Q(s, μ(s)) )` — auto-diff handles the chain rule. See [`services/ddpg_update.py::actor_loss`](../src/roomba_lab/services/ddpg_update.py).

### 3.2 Soft (Polyak) target update (Lillicrap et al. 2016; slide 6)

$$\theta'_{\mu} \leftarrow \tau\,\theta_{\mu} + (1-\tau)\,\theta'_{\mu}, \quad \theta'_{Q} \leftarrow \tau\,\theta_{Q} + (1-\tau)\,\theta'_{Q}$$

with τ small (default 0.005). Implemented in [`model/soft_update.py::polyak_update`](../src/roomba_lab/model/soft_update.py).

### 3.3 Critic loss (TD-error MSE)

$$L_{Q} \;=\; \mathbb{E}\!\left[\left(r + \gamma\, Q'(s', \mu'(s')) - Q(s, a)\right)^{2}\right]$$

Implemented in [`services/ddpg_update.py::critic_loss`](../src/roomba_lab/services/ddpg_update.py).

## 4. Environment (the custom simulator)

| Item | Value | Rationale |
|---|---|---|
| Map source | HouseExpo JSONs (real apartments) | Spec § 1 |
| Map representation | List of polygons (walls + obstacles) → rasterised to a 2-D occupancy grid | Polygon for ray-casting, grid for coverage tracking |
| Robot model | Differential-drive kinematics: `(x, y, θ)` updated by `(v, ω)` | Slide 3 |
| Action space | `(v_norm, ω_norm) ∈ [−1, 1]²` → scaled to `[v_min, v_max] × [ω_min, ω_max]` | Slide 10 |
| Observation | LIDAR (24 beams × range) + (x, y, θ_sin, θ_cos) + (coverage fraction) → 29-dim float | Spec § 1 |
| Reward | new-cell bonus (+1) + collision penalty (−10) + small step penalty (−0.01) + completion bonus (+100 at 85 % coverage) | Spec § 1 |
| Time step | dt = 0.1 s (10 Hz) | Realistic vacuum |
| Episode horizon | 500 steps (= 50 s sim time) | Limits early-training compute |

## 5. Networks

| Network | Topology | Output |
|---|---|---|
| Actor μ(s) | MLP(29 → 256 → 256 → 2) + `tanh` | Action ∈ [−1, 1]² |
| Critic Q(s, a) | MLP(29+2 → 256 → 256 → 1) (state and action concatenated at input — slide 5) | Scalar V(s, a) |
| Target actor μ′ | Same topology as μ, frozen between Polyak updates | — |
| Target critic Q′ | Same topology as Q, frozen between Polyak updates | — |

## 6. Hyperparameters (justified in PRD § 4 + § 7 of the spec)

All in [`configs/setup.json`](../configs/setup.json). No magic numbers in source.

| Hyperparameter | Value | Justification |
|---|---|---|
| Actor LR | 1e-4 | Lillicrap 2016 standard |
| Critic LR | 1e-3 | Lillicrap 2016 standard — critic needs to track Q faster |
| γ | 0.99 | Standard PPO/DDPG default |
| τ | 0.005 | Lillicrap 2016 + spec § Item 3 ("e.g., 0.005") |
| Batch size | 128 | Lillicrap 2016 standard |
| Replay capacity | 200 000 | Big enough for 50 k-step run not to wrap |
| Warm-up steps | 1 000 | Fill the buffer with random transitions before first update |
| Exploration σ (initial) | 0.2 | Spec § Item 4 ("what is the initial variance you chose") |
| Exploration σ (final) | 0.05 | Linear decay so exploration tapers but never vanishes |
| Hidden sizes | [256, 256] | Lillicrap 2016 (full DDPG paper used [400, 300]; we use a slightly more uniform shape) |

## 7. Empirical study plan

| Study | Cells | Seeds | Headline question |
|---|---|---|---|
| **Noise σ ablation** | {0.0, 0.1, 0.2, 0.4} | 3 | Spec reflection Q2: "what if you remove Gaussian noise entirely?" |
| **τ ablation** | {0.001, 0.005, 0.01, 0.05} | 3 | Spec reflection Q3: "how do soft updates protect the critic?" |
| **Target-network ablation** | hard-copy every N steps vs soft τ=0.005 | 3 | Same as above — exposes the collapse risk explicitly |
| **Cross-apartment transfer** | train on apt-A, eval on apt-B | 3 | DDPG generalisation across morphologies |

## 8. Visualisations (mandatory per spec)

| Required by spec | Implementation |
|---|---|
| **Learning Curve** (reward vs steps) | [`assets/plots/learning_curve.png`](../assets/plots/learning_curve.png) |
| **Critic Loss curve** | [`assets/plots/critic_loss.png`](../assets/plots/critic_loss.png) |
| **Trajectory overlay** on the JSON map | [`assets/plots/trajectory_overlay.png`](../assets/plots/trajectory_overlay.png) |
| **Animation** of the cleaning behaviour | [`assets/gifs/cleaning_episode.gif`](../assets/gifs/cleaning_episode.gif) |

Above-spec polish (Layer 16):

- Coverage **heatmap** on top of the apartment plan (which cells the robot has visited; brighter = visited)
- Noise-σ sweep plot (3 seeds, 95 % CI bands)
- τ-sweep plot (3 seeds, 95 % CI bands)

## 9. Three reflection questions (must answer in the report)

Per the spec § "שאלות ניתוח והבנה" — the report MUST answer:

1. **Why DDPG (vs DQN, vs PPO)?** — Reflection answer in README; argued from Table 1 of L09: continuous-only action space + deterministic actor + max-Q off-policy data reuse.
2. **What happens if you remove Gaussian exploration noise entirely in the early stages?** — Answered with empirical evidence: the noise-σ ablation plot at σ=0.0 shows coverage flat-lining near the spawn point.
3. **How do target networks + soft updates protect the critic from collapse?** — Answered with the target-network ablation: hard-copy vs soft τ=0.005 plot, showing critic-loss divergence under hard updates.

## 10. Originality hooks (above-spec, V3 § 1.4 differentiators)

| Hook | Why it's beyond spec |
|---|---|
| **Mini-Graphify port** (carried forward from Assignment 4) | AST walker → Obsidian Vault for `src/roomba_lab` |
| **PyQt6 GUI** with live trajectory animation + cleaning heatmap | Spec only requires "find a visual way" |
| **Statistical CIs on all sweep plots** (3-seed multi-seed) | Spec asks for two graphs; we produce 4 with proper error bars |
| **OU noise alternative** in `noise/` even though Gaussian is the default | Lets the student SEE why Lillicrap originally chose OU and why later work moved to Gaussian |
| **Coverage-progress curve** overlaid on the reward curve | Connects the learning curve to the actual coverage objective |

## 11. KPIs (Definition of Project Done)

| KPI | Target |
|---|---|
| Lecture-slide coverage | 100 % of L09 § 4–7 has a corresponding code module |
| Custom simulator (no Gym, no Gazebo) | Hard requirement — no Gym imports anywhere |
| HouseExpo data integration | At least 10 real JSON apartments loaded + 1 fully trained-on |
| All `configs/setup.json` keys consumed | Yes — verified by config-coverage test |
| Tests | ≥ 80 unit tests; coverage ≥ 85 % |
| Files ≤ 150 LOC each | All non-vendor source files |
| ruff clean | `ruff check .` returns 0 |
| Reflection answers | All 3 spec questions answered with empirical evidence |
| Headline plot | Learning curve + critic-loss curve with 3-seed CI bands |
| Originality hook | Mini-Graphify port + GUI + OU noise + heatmap |

## 12. Deliverables

1. `src/roomba_lab/` — pure Python source ≤ 150 LOC per file
2. `tests/` — unit + integration tests, coverage ≥ 85 %
3. `configs/setup.json` — every hyperparameter, version field
4. `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` + 5 per-mechanism PRDs (this folder)
5. `notebooks/roomba_lab_walkthrough.ipynb` — 6-cell guided tour, executed end-to-end
6. `assets/plots/*.png` — 5+ figures including the mandatory 2 + 3 above-spec
7. `assets/gifs/cleaning_episode.gif` — animation per spec
8. `docs/PROMPTBOOK.md`, `docs/COSTS.md` — V3 § 20.9 # 1 + # 7
9. CI: `.github/workflows/assignment5-ci.yml` — green badge in README
10. Tag `assignment5-v1.00` after the final layer

## 13. Sources

1. Lillicrap et al., "Continuous control with deep reinforcement learning", ICLR 2016 (DDPG).
2. Silver et al., "Deterministic Policy Gradient Algorithms", ICML 2014 (DPG theorem — slide 4).
3. Li et al., "HouseExpo: A large-scale 2D indoor layout dataset for learning-based algorithms on mobile robots", arXiv 1903.09845, 2019 — github.com/TeaganLi/HouseExpo.
4. Fujimoto et al., "Addressing function approximation error in actor-critic methods", ICML 2018 (TD3) — *cited only for noise/buffer best-practices, not implemented as the main algorithm*.
5. The L09 lecture PDF (`gemini-L09-Deep-Deterministic-Policy-Gradient-DDPG-and-EX05t.pdf`) — Dr. Yoram Segal, June 2026.
6. The exercise sheet `EX05-DDPG-Robot-Simulator.pdf` — Dr. Yoram Segal, June 2026.
