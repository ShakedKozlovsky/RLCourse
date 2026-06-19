# TODO — Layered Implementation Plan (Assignment 5)

> Each layer = one commit. **Definition of Done** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md).

---

## Layer 0 — Scaffold + planning docs

- [x] Create directory tree under `assignment5/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [x] `docs/TODO.md` (this file)
- [x] Per-mechanism PRDs: `PRD_ddpg.md`, `PRD_simulator.md`, `PRD_lidar_sensor.md`, `PRD_soft_target_updates.md`, `PRD_exploration_noise.md`
- [x] `README.md` placeholder
- [x] `pyproject.toml`
- [x] `.gitignore`, `.env-example`
- [x] `configs/setup.json`
- [x] All `__init__.py` files
- [x] `shared/version.py` with `__version__ = "1.00"`

**DoD:** repo importable (`uv run python -c "import roomba_lab"`), docs explain the full plan, no code beyond shared/version.

---

## Layer 1 — Shared layer + HouseExpo loader + 10-map sample

Commit: `Layer 1: shared/* + data/houseexpo_loader + 10-map sample`

- [ ] `shared/config.py` — JSON loader with version check + dotted access (`cfg.get("ddpg.tau")`)
- [ ] `shared/logger.py` — stdlib logger factory (no print in library)
- [ ] `shared/seed.py` — `set_global_seed(int)` for Python + NumPy + PyTorch
- [ ] `shared/types.py` — `Transition`, `EpisodeMetrics`, `StepDiagnostic`, `TrainResult` dataclasses
- [ ] `data/houseexpo_loader.py` — `download_dataset()` + `load_map(map_id) → dict[polygons, metadata]`
- [ ] Download the upstream `json.tar.gz` once, extract the **10 maps listed in `map_id_10.txt`**, commit them as `data/raw/sample_maps/<id>.json`
- [ ] Tests:
  - `test_config.py`: loads, version mismatch raises
  - `test_houseexpo_loader.py`: parsing yields ≥ 1 wall polygon and a bounding box; SHA-256 cache key works

**DoD:** `roomba-lab download-data` is idempotent; the 10 sample apartments load + render to PNG in tests/integration.

---

## Layer 2 — Simulator core (kinematics + world + collision)

Commit: `Layer 2: simulator core (kinematics, world, collision)`

- [ ] `simulator/kinematics.py::step_unicycle(pose, action, dt, max_v, max_w) → new_pose` (pure)
- [ ] `simulator/world.py::World` — holds polygons + occupancy grid + dimensions; `cell_index(x, y) → (i, j)`
- [ ] `simulator/collision.py::is_collision(pose, world, robot_radius) → bool`
- [ ] `simulator/robot.py::Robot` — pose, radius, velocity history
- [ ] Tests:
  - 4-test kinematics battery (zero action, max forward, max turn, combo)
  - Collision boundary inside/outside
  - World grid round-trip (cell ↔ continuous)

**DoD:** 100 % branch coverage in `simulator/`; pure functions throughout.

---

## Layer 3 — LIDAR + custom env + reward

Commit: `Layer 3: lidar + roomba_env + reward`

- [ ] `sensor/lidar.py::LidarSensor.scan(pose, world) → np.ndarray[n_beams]`
- [ ] `environment/reward.py::compute_reward(state, prev_state, world, cfg) → (reward, info)` (pure)
- [ ] `environment/roomba_env.py::RoombaEnv` — `reset()`, `step(action)`, `render()`, **with zero `gym` imports**
- [ ] Tests:
  - LIDAR closed-form on a square room (8 cardinal beams hit at known distances)
  - Reward function: new-cell bonus fires once; collision penalty cancels move; completion bonus fires once
  - Env smoke: 50 steps without errors; obs shape `(29,)`, action shape `(2,)`

**DoD:** `RoombaEnv` is a strict spec match (no gym); 50-step random rollout produces sensible coverage.

---

## Layer 4 — Networks + soft target update

Commit: `Layer 4: actor + critic + soft target update`

- [ ] `model/init.py` — orthogonal-init helper (gain choice documented per layer)
- [ ] `model/actor.py::Actor(obs_dim, act_dim, hidden_sizes)` — MLP + tanh output
- [ ] `model/critic.py::Critic(obs_dim, act_dim, hidden_sizes)` — state+action concat → scalar
- [ ] `model/actor_critic_network.py::ActorCriticNet` — wraps actor, critic, target_actor, target_critic
- [ ] `model/soft_update.py::polyak_update(target, source, tau)` — **TDD 4-test battery**:
  - τ=0 → target unchanged
  - τ=1 → target = source (hard copy)
  - τ=0.5 → target = midpoint
  - Repeated calls converge target → source
- [ ] Tests: forward shape; save/load round-trip; tanh output ∈ [−1, 1]; target params disabled from grad

**DoD:** networks + Polyak update verified by math battery; 100 % coverage on `model/soft_update.py`.

---

## Layer 5 — Replay buffer

Commit: `Layer 5: replay buffer`

- [ ] `memory/replay_buffer.py::ReplayBuffer(capacity, obs_dim, act_dim)` — fixed-size NumPy ring
- [ ] Methods: `push(Transition)`, `sample(batch_size) → dict[str, ndarray]`, `__len__`
- [ ] Tests: capacity wraps; sampling without replacement; raises when sampled before warm-up

**DoD:** 200 000-capacity buffer pushes + samples in microseconds; 100 % coverage.

---

## Layer 6 — Exploration noise (Gaussian + OU + schedule)

Commit: `Layer 6: exploration noise (Gaussian + OU + schedule)`

- [ ] `noise/gaussian.py::GaussianNoise(act_dim, sigma)` — `sample() → ndarray`; `set_sigma(s)`
- [ ] `noise/ou.py::OUNoise(act_dim, theta, mu, sigma)` — Ornstein-Uhlenbeck process
- [ ] `noise/schedule.py::LinearSigmaSchedule(initial, final, decay_steps)` — `at(step) → σ`
- [ ] Tests:
  - Gaussian mean/variance match closed-form (large-N)
  - OU produces autocorrelated sequence (lag-1 corr > 0.8 at θ=0.15)
  - Schedule: at(0)=initial, at(decay_steps)=final, at(decay_steps + 1000)=final (clamped)

**DoD:** all 3 noise utilities pass closed-form tests.

---

## Layer 7 — DDPG update step (the headline math)

Commit: `Layer 7: DDPG update step`

- [ ] `services/ddpg_update.py::critic_loss(net, batch, gamma) → tensor`
- [ ] `services/ddpg_update.py::actor_loss(net, batch) → tensor`
- [ ] `services/ddpg_update.py::apply_update(net, batch, gamma, tau, actor_opt, critic_opt, max_grad_norm) → StepDiagnostic`
- [ ] **TDD pair — write tests first**:
  - Critic-loss gradient flows to critic params, NOT to target params
  - Actor-loss gradient flows to actor params, NOT to critic params (critic frozen during actor step)
  - One update changes weights (Δ > 1e-6)
  - τ = 0.005 Polyak step moves target by exactly that fraction

**DoD:** update step is a pure transformation of (net, batch) → (net', diagnostics); diagnostics include `critic_loss`, `actor_loss`, `mean_q`, `target_drift`.

---

## Layer 8 — DDPG training service

Commit: `Layer 8: DDPG training service`

- [ ] `services/ddpg_service.py::DDPGService(net, env, buffer, noise, schedule, hp)` with `fit(total_timesteps) → TrainResult`
- [ ] Per-step logging: episode reward, critic loss, actor loss, σ, mean Q
- [ ] Optional warm-up phase: random actions before `t > warmup_steps`
- [ ] Save best policy on best eval reward
- [ ] Tests: smoke 1000-step run on a small synthetic world; finite TrainResult; loss decreases (or stays bounded)

**DoD:** 5 000-step training run on a HouseExpo apartment produces a finite TrainResult with non-trivial coverage.

---

## Layer 9 — SDK + CLI

Commit: `Layer 9: SDK + CLI`

- [ ] `sdk/sdk.py::RoombaLab(config_path)` — facade with `make_env`, `train`, `evaluate`, `run_sweep`, `graphify`, `predict`, `record_gif`
- [ ] `sdk/env_builder.py::build_env(cfg, map_id)`
- [ ] `sdk/trainers.py::build_ddpg_service(cfg)` constructor
- [ ] `sdk/experiments.py::ExperimentService` (lives here per ADR-007, not under services/)
- [ ] `interface/cli/main.py` — Click group
- [ ] `interface/cli/commands.py` — bulky bodies: `train`, `evaluate`, `sweep`, `graphify`, `gui`, `download-data`, `record-gif`
- [ ] Tests: each CLI subcommand exits 0 on 200-step smoke training run

**DoD:** `uv run roomba-lab --help` lists 7 subcommands; smoke run completes.

---

## Layer 10 — Mini-Graphify port

Commit: `Layer 10: Mini-Graphify port`

- [ ] Port `tools/graphify/{walker, emitter, runner}.py` from Assignment 4
- [ ] Re-emit for `src/roomba_lab` — verify the AST walker handles `from __future__ import annotations` + dataclasses
- [ ] Update `configs/setup.json` graphify block accordingly
- [ ] Tests: walker produces correct edge count on synthetic 3-module fixture; emitter writes valid JSON + Markdown with Wikilinks

**DoD:** `roomba-lab graphify` produces `docs/wiki/` that opens in Obsidian; graph view shows the project's module dependency graph.

---

## Layer 11 — Empirical sweeps (noise-σ, τ)

Commit: `Layer 11: empirical sweeps (noise-σ + tau)`

- [ ] `sdk/experiments.py::ExperimentService` adds `run_noise_sigma_sweep`, `run_tau_sweep`, `run_target_network_ablation`
- [ ] `scripts/run_noise_sigma_sweep.py` — multi-seed
- [ ] `scripts/run_tau_sweep.py` — multi-seed
- [ ] `scripts/run_target_network_ablation.py` — soft (τ=0.005) vs hard-copy (every N steps)
- [ ] Results saved to `results/sweeps/*.json`
- [ ] Tests: each sweep CLI completes a tiny smoke run

**DoD:** 3 sweep JSONs + 3 PNG plots exist; per-cell 95 % CI computed and reported.

---

## Layer 12 — Visualisations

Commit: `Layer 12: visualisations (trajectory overlay + GIF + coverage heatmap)`

- [ ] `scripts/visualise_trajectory.py` — overlays the trained robot's path on top of the apartment polygon plot → `assets/plots/trajectory_overlay.png`
- [ ] `scripts/record_gif.py` — uses `imageio` to record a cleaning episode → `assets/gifs/cleaning_episode.gif`
- [ ] `scripts/coverage_heatmap.py` — colour-coded grid showing visited cells → `assets/plots/coverage_heatmap.png`
- [ ] `scripts/learning_curve.py` + `scripts/critic_loss_curve.py` — mandatory spec graphs

**DoD:** every spec-mandated visualisation exists on disk; GIF is < 5 s and < 5 MB.

---

## Layer 13 — Audit response

Commit: `Layer 13: audit response (multi-seed sweeps + reproducibility tests)`

- [ ] Self-audit: ≥ 10 findings categorised Critical / Important / Nice-to-have
- [ ] Add 2 integration tests: same-seed → identical trajectories + identical critic-loss curves
- [ ] Bump CI: run sweeps in headless matplotlib
- [ ] Fix any findings flagged Critical

**DoD:** all Critical findings closed; reproducibility tests pass.

---

## Layer 14 — PyQt6 GUI

Commit: `Layer 14: PyQt6 GUI`

- [ ] `interface/gui/main_window.py` — tabbed window
- [ ] `interface/gui/training_tab.py` — start/stop training, live reward + loss plot
- [ ] `interface/gui/visualisation_tab.py` — pick a saved checkpoint, run an eval episode, show trajectory animation
- [ ] Tests: smoke under offscreen Qt (window opens; tabs render; no exceptions)

**DoD:** `roomba-lab gui` opens a window; both tabs functional.

---

## Layer 15 — Notebook walkthrough

Commit: `Layer 15: notebook walkthrough`

- [ ] `notebooks/roomba_lab_walkthrough.ipynb` — 6 cells:
  1. Imports + config
  2. Load a HouseExpo apartment + render
  3. Initialise DDPG agent
  4. Train (smoke 5 000 steps)
  5. Evaluate + visualise trajectory
  6. Embed the headline learning curve
- [ ] Execute end-to-end with `nbconvert --execute`
- [ ] Embed outputs + figures inline

**DoD:** notebook renders without errors; figures embedded.

---

## Layer 16 — Final README + executive summary + reflection answers

Commit: `Layer 16: final docs + reflection answers + sign-off`

- [ ] Rewrite top-level `README.md` with: slide mapping, equations, environment, networks, hyperparameters, headline empirical results, GUI/CLI/SDK, reflection answers, sources
- [ ] `docs/EXECUTIVE_SUMMARY.md` — 1-pager for the grader
- [ ] `docs/REPRODUCIBILITY.md` — exact-replay instructions
- [ ] Reflection answers grounded in `results/sweeps/*.json`
- [ ] Above-spec polish: coverage curve, heatmap, OU vs Gaussian comparison plot

**DoD:** every PRD § 11 KPI ticked; every spec § "שאלות ניתוח והבנה" question answered with empirical evidence.

---

## Layer 17 — V3 final polish (Promptbook + Costs + CI + extension points + v1.00 tag)

Commit: `Layer 17: V3 PDF final-checklist gap close — Promptbook + Costs + Extension points + v1.00 tag`

- [ ] `docs/PROMPTBOOK.md` — AI-assisted methodology log (V3 § 8.3 / § 17.1 / § 20.9 # 1)
- [ ] `docs/COSTS.md` — token cost analysis (V3 § 11 / § 20.9 # 7)
- [ ] `docs/PLAN.md` § 14 — Extension points (V3 § 12.1 / § 20.9 # 8)
- [ ] `.github/workflows/assignment5-ci.yml` — green badge in README
- [ ] Tag `assignment5-v1.00` and push

**DoD:** every V3 § 20.9 checklist item satisfied; CI badge green; tag pushed.
