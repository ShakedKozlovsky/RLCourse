# TODO — Layered Implementation Plan (Assignment 4)

> Each layer = one commit. **Definition of Done** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md).

---

## Layer 0 — Scaffold + planning docs

- [x] Create directory tree under `assignment4/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [x] `docs/TODO.md` (this file)
- [ ] Per-mechanism PRDs: `PRD_ppo.md`, `PRD_gae.md`, `PRD_actor_critic.md`, `PRD_mujoco_env.md`, `PRD_graphify.md`
- [ ] `README.md` placeholder
- [ ] `pyproject.toml` + `uv.lock` (lock deferred to Layer 1 after a working `uv sync`)
- [x] `.gitignore`, `.env-example`
- [x] `configs/setup.json`
- [ ] All `__init__.py` files
- [ ] `shared/version.py` with `__version__ = "1.00"`

**DoD:** repo importable (`python -c "import proximal_lab"` works after `uv sync`), docs explain the full plan, no code beyond shared/version.

---

## Layer 1 — Shared layer + MuJoCo environment

Commit: `Layer 1: shared/* + environment/{mujoco_env,vector_env} + smoke tests`

- [ ] `shared/config.py` — JSON loader with version check + dotted access
- [ ] `shared/logger.py` — stdlib logger factory (no print in library)
- [ ] `shared/seed.py` — set_global_seed (Python + NumPy + PyTorch)
- [ ] `shared/types.py` — RolloutBatch, EpisodeMetrics, TrainResult dataclasses
- [ ] `environment/mujoco_env.py` — `gym.make` wrapper with running-mean / running-std obs normaliser
- [ ] `environment/vector_env.py` — SyncVectorEnv wrapper
- [ ] Tests: config loads + version mismatch raises, env produces (17,) obs and (6,) action, vector env produces (n, 17) batched obs

**DoD:** smoke env creation + step + reset works for both HalfCheetah-v4 and Walker2d-v4; tests pass; coverage ≥ 85 % for `shared/` and `environment/`.

---

## Layer 2 — Actor-Critic networks

Commit: `Layer 2: GaussianActor + Critic + ActorCriticNet + orthogonal init + tests`

- [ ] `model/init.py` — orthogonal initialisation helpers (PPO convention: actor head with small gain, critic with 1.0)
- [ ] `model/actor.py` — `GaussianActor(obs_dim, action_dim, hidden_sizes)` with state-independent log_std
- [ ] `model/critic.py` — `Critic(obs_dim, hidden_sizes)` returning scalar V(s)
- [ ] `model/actor_critic_network.py` — `ActorCriticNet` wrapping both; `act(obs) → (action, log_prob, value)`
- [ ] Tests: forward-shape contracts, log-prob differentiable w.r.t. policy params, log_std clamped to `[log_std_min, log_std_max]`

**DoD:** forward and backward pass produce finite gradients on random inputs; networks save/load round-trip preserves outputs.

---

## Layer 3 — Rollout buffer

Commit: `Layer 3: RolloutBuffer + tests`

- [ ] `services/rollout_buffer.py` — `RolloutBuffer(size, obs_dim, action_dim, n_envs)`
- [ ] Methods: `add(s, a, log_prob, r, v, done)`, `compute_returns_and_advantages(last_value, gamma, lam)` (calls into `gae.compute_gae`), `minibatches(size) → iterator`
- [ ] Tests: capacity respected, advantages match the GAE module's output, minibatches sum to full buffer size

**DoD:** buffer can collect 2048 × 4 = 8192 transitions on HalfCheetah and yield 64-sized minibatches with no duplicates.

---

## Layer 4 — GAE module (the headline math)

Commit: `Layer 4: services/gae.py + 4-test math battery`

- [ ] `services/gae.py::compute_gae(rewards, values, last_value, dones, gamma, lam) → ndarray`
- [ ] Reverse recursion implementation (Eq. 2c in PRD)
- [ ] **TDD pair 1**: write tests first using hand-computed δ sequence
- [ ] Tests:
  - λ = 0 → GAE collapses to TD error `δ_t`
  - λ = 1 → GAE collapses to MC-return − V(s)
  - Closed-form check on a 3-step trajectory with known rewards/values
  - Terminal-handling: bootstrap zeroed when `done == True`

**DoD:** 4 math tests pass; the module is a pure function with zero state. 100 % coverage.

---

## Layer 5 — PPO clip math

Commit: `Layer 5: ppo_clip_loss + 4-test sign/window battery`

- [ ] `services/ppo_service.py::ppo_clip_loss(ratio, advantages, clip_eps) → tensor`
- [ ] **TDD pair 2**: tests written first
- [ ] Tests:
  - `r = 1, Â > 0`: loss = `−Â` (unclipped branch)
  - `r > 1+ε, Â > 0`: clipped branch wins (more conservative — smaller surrogate gain)
  - `r > 1+ε, Â < 0`: unclipped branch wins (slide 11–12 intuition)
  - `r < 1−ε, Â < 0`: clipped branch wins

**DoD:** 4 math tests pass; output is differentiable w.r.t. ratio.

---

## Layer 6 — PPO training service

Commit: `Layer 6: PPOService.fit() — end-to-end PPO + GAE training loop`

- [ ] `services/ppo_service.py::PPOService(gamma, lambda, lr, clip_eps, n_epochs, minibatch_size, value_coef, entropy_coef, max_grad_norm, target_kl_stop)`
- [ ] `fit(net, env, total_timesteps) → TrainResult` runs the slide-18 pipeline
- [ ] Per-iteration logging: episode reward, mean KL, clip fraction, explained variance, value loss, policy loss
- [ ] Optional **target-KL early stop** within an epoch (KL > target → break)
- [ ] Tests:
  - Smoke: 1000 timesteps on HalfCheetah produces finite TrainResult
  - Single update changes weights (delta > 1e-6)
  - Per-iteration mean KL bounded under typical settings

**DoD:** 5000-timestep training run on HalfCheetah completes in seconds, with episode reward visibly trending up.

---

## Layer 7 — Evaluation + comparison

Commit: `Layer 7: EvaluationService + ComparisonService + tests`

- [ ] `services/evaluation_service.py::EvaluationService.rollout(net, env, n_episodes=10, deterministic=True)` returns mean ± std of episode returns
- [ ] `services/comparison_service.py::ComparisonService` aggregates per-config results into a `ComparisonReport`
- [ ] Tests: deterministic eval = argmax action; same seed → identical eval reward

**DoD:** evaluate a 5000-timestep-trained policy and report mean episode reward across 10 episodes.

---

## Layer 8 — SDK + CLI

Commit: `Layer 8: ProximalLab SDK + Click CLI + integration tests`

- [ ] `sdk/sdk.py::ProximalLab(config_path)` — facade with `make_env`, `train_ppo`, `evaluate`, `run_sweep`, `graphify`, `predict`
- [ ] `sdk/env_builder.py::build_env(cfg, env_id)` — sized exactly like Assignment 3's
- [ ] `sdk/trainers.py` — `build_ppo_service(cfg)` constructor
- [ ] `interface/cli/main.py` — Click group + short commands
- [ ] `interface/cli/commands.py` — bulky bodies: `train`, `evaluate`, `sweep`, `graphify`, `gui`, `menu`
- [ ] Tests: each CLI subcommand exits 0 on a 200-timestep smoke training run

**DoD:** `uv run proximal-lab --help` lists 7 subcommands; smoke run completes.

---

## Layer 9 — Mini-Graphify tool (the methodology differentiator)

Commit: `Layer 9: tools/graphify — AST walker + Markdown emitter for docs/wiki`

- [ ] `tools/graphify/walker.py` — `ast.parse` visitor extracting modules, classes, public functions, and import edges
- [ ] Optional call-graph: function-level edges within a module
- [ ] `tools/graphify/emitter.py` — writes `docs/wiki/graph.json` (nodes + edges) and one `docs/wiki/<module>.md` per module with Wikilinks
- [ ] `tools/graphify/runner.py` — `proximal-lab graphify` CLI entry
- [ ] Tests: walker produces correct edge count on a synthetic 3-module fixture; emitter writes valid JSON + Markdown with Wikilinks

**DoD:** `proximal-lab graphify` produces `docs/wiki/` that opens in Obsidian; graph view shows the project's module dependency graph.

---

## Layer 10 — Empirical sweeps (λ, γ, clip-ε)

Commit: `Layer 10: ExperimentService — λ + γ + clip-ε sweeps with 3-seed CI`

- [ ] `services/experiment_service.py::ExperimentService` with `run_lambda_sweep`, `run_gamma_sweep`, `run_clip_eps_sweep`
- [ ] Each cell: 200k timesteps × 3 seeds; emit JSON with `final_reward_mean`, `final_reward_ci_95`, full reward curve
- [ ] CLI command `proximal-lab sweep <kind>` writes results to `results/sweeps/<kind>.json`
- [ ] Plots: `assets/plots/lambda_sweep.png`, `gamma_sweep.png`, `clip_eps_sweep.png` — each with error bars
- [ ] Tests: aggregate-with-CI returns correct mean ± 95 % normal-approx CI on synthetic data

**DoD:** λ-sweep shows the predicted bias-variance ladder (slide 16). Result table embedded in README.

---

## Layer 11 — Cross-environment generalisation (Walker2d)

Commit: `Layer 11: best (λ, γ, ε) on HalfCheetah → transfer to Walker2d-v4`

- [ ] Take the best hyperparameter triple from Layer 10
- [ ] Train on Walker2d-v4 with same budget × 3 seeds
- [ ] Plot side-by-side reward curves
- [ ] Discussion: does the HalfCheetah optimum transfer? If λ peak shifts, document why

**DoD:** `assets/plots/cross_env_comparison.png` + `results/layer11/cross_env.json` + README § discussion.

---

## Layer 12 — Notebook walkthrough

Commit: `Layer 12: notebooks/proximal_lab_walkthrough.ipynb + executed outputs`

- [ ] 6-cell notebook: load env → build network → short PPO training → evaluate → run mini-sweep → graphify
- [ ] All cells executed; outputs + embedded plots present in the committed `.ipynb`

**DoD:** notebook opens cleanly on GitHub and shows real numbers; ≤ 200 KB.

---

## Layer 13 — Audit response

Commit: `Layer 13: Claude playing professor — fix surfaced gaps`

- [ ] Run an adversarial audit (same flow as Assignment 3 Layer 11–14)
- [ ] Categorise findings: Critical / Important / Nice-to-have
- [ ] Address every Critical and Important finding
- [ ] Document audit-status table in README § 9

**DoD:** 20-finding audit table all closed, README § audit table embedded.

---

## Layer 14 — PyQt6 GUI

Commit: `Layer 14: PyQt6 GUI with Train + Sweep + Compare tabs + headless smoke test`

- [ ] `interface/gui/main_window.py` — QMainWindow with 3 tabs
- [ ] `interface/gui/plot_widget.py` — matplotlib FigureCanvasQTAgg wrapper
- [ ] `interface/gui/worker.py` — QThread for off-main-thread training
- [ ] Tabs: train_tab (single-config training), sweep_tab (λ/γ/clip-ε sweeps), compare_tab (curve overlay)
- [ ] Headless test under `QT_QPA_PLATFORM=offscreen`

**DoD:** GUI launches; all 3 tabs construct + render; screenshots in `assets/gui/`.

---

## Layer 15 — Final README + plots + reflection + executive summary + sign-off

Commit: `Layer 15: final README + EXECUTIVE_SUMMARY.md + screenshots + final push`

- [ ] README § 1–14 covering: project goal, lecture-slide mapping, environments, networks, PPO + GAE equations, training, sweeps, cross-env, GUI/CLI, quality bar, reflection answers, audit table, sources, honest acknowledgements
- [ ] All plots embedded
- [ ] Architecture diagram as PNG (generated from `docs/wiki/graph.json` — the meta-payoff)
- [ ] Pre-trained checkpoints in `saved_models/`
- [ ] `docs/EXECUTIVE_SUMMARY.md` — 1-pager for grader's first pass
- [ ] `docs/wiki/` Obsidian Vault committed (force-add through `.gitignore`)

**DoD:** every assignment requirement maps to a section in README. Self-grading checklist signed off in the audit-status table.

---

## Cross-cutting (every commit)

- [ ] `ruff check` returns 0
- [ ] Each commit: `Layer N: <summary>` + bullet body
- [ ] No magic numbers — everything in `configs/`
- [ ] No file > 150 LOC
- [ ] After each layer: update this TODO.md, mark items `[x]`
- [ ] Push to RLCourse @ main after each layer

## Future extensions (not in scope)

- [ ] TRPO baseline (the slide-6 motivation)
- [ ] RLHF mini-LM (slide 20)
- [ ] Vectorised env via `AsyncVectorEnv` (shared-memory)
- [ ] Distributional value functions (C51 / QR-DQN)
- [ ] Curriculum learning across multiple MuJoCo environments
