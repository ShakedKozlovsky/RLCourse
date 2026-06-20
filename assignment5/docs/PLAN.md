# PLAN — Layered architecture, ADRs, and pseudocode

> Companion to [PRD.md](PRD.md). Defines **how** we will build the system. Layered design follows V3 § 2.4 + the prior assignments' `interface → sdk → services → {…} → shared` convention.

## 1. Architecture diagram

```
                       ┌─────────────────────────────────────────────┐
                       │              interface/                     │
                       │   (cli/, gui/) — Click + PyQt6              │
                       └──────────────────────┬──────────────────────┘
                                              │
                       ┌──────────────────────▼──────────────────────┐
                       │                  sdk/                       │
                       │   ProximalLab-style facade (RoombaLab),     │
                       │   env_builder, trainers, experiments        │
                       └──────────────────────┬──────────────────────┘
                                              │
       ┌──────────────────────────────────────▼──────────────────────────────────────┐
       │                                services/                                    │
       │   ddpg_service, ddpg_update, evaluation_service, comparison_service,        │
       │   experiment_service, training_loop                                         │
       └──────────────┬─────────────────────┬──────────────────────┬─────────────────┘
                      │                     │                      │
        ┌─────────────▼────────────┐   ┌────▼──────────┐   ┌──────▼────────────┐
        │       environment/       │   │     model/    │   │     memory/       │
        │  (custom env wrapper +   │   │ actor, critic,│   │  replay buffer    │
        │   reward fn + episode    │   │ soft-update   │   │                   │
        │   API)                   │   │ helpers       │   │                   │
        └─────────────┬────────────┘   └───────────────┘   └───────────────────┘
                      │
        ┌─────────────▼────────────┐   ┌───────────────┐   ┌───────────────────┐
        │      simulator/          │   │    sensor/    │   │     noise/        │
        │  kinematics, world,      │   │ LIDAR ray-    │   │ Gaussian, OU,     │
        │  collision               │   │ caster        │   │ schedule          │
        └─────────────┬────────────┘   └───────────────┘   └───────────────────┘
                      │
        ┌─────────────▼────────────┐   ┌───────────────────────────────────────┐
        │         data/            │   │            shared/                    │
        │  HouseExpo loader,       │   │  config, logger, seed, types,         │
        │  JSON → polygons + grid  │   │  version                              │
        └──────────────────────────┘   └───────────────────────────────────────┘
                                       │     tools/graphify/ — AST→Vault       │
                                       └───────────────────────────────────────┘
```

## 2. Package map (one paragraph per package)

- **`shared/`** — pure-Python utilities. `config.py` loads JSON + version check; `logger.py` makes the standard `logging.Logger`; `seed.py` seeds Python+NumPy+PyTorch; `types.py` holds tiny dataclasses (`Transition`, `EpisodeMetrics`, `TrainResult`); `version.py` exports `__version__`.
- **`data/`** — `houseexpo_loader.py` downloads (Layer 1) and parses HouseExpo JSONs into a list of polygons + a metadata block (room IDs, doors).
- **`simulator/`** — `world.py` holds the static apartment (polygons + occupancy grid for coverage); `robot.py` holds dynamic state (pose, velocity); `kinematics.py` is the pure `(pose, action, dt) → next_pose` function; `collision.py` does point-in-polygon checks via `shapely`.
- **`sensor/`** — `lidar.py` casts N rays from the robot's pose against the world polygons.
- **`environment/`** — `roomba_env.py` is the **Gym-compatible-by-shape (NOT a Gym subclass)** env wrapper: `reset()`, `step(action) → (obs, reward, done, info)`. `reward.py` is the pure reward function.
- **`model/`** — `actor.py` (deterministic, tanh output), `critic.py` (state+action concat), `actor_critic_network.py` (wraps both + targets), `init.py` (PyTorch init helpers), `soft_update.py` (Polyak update).
- **`memory/`** — `replay_buffer.py` is a fixed-capacity ring buffer of `Transition`s.
- **`noise/`** — `gaussian.py` (slide-7 default), `ou.py` (Lillicrap 2016 original), `schedule.py` (linear σ decay).
- **`services/`** — `ddpg_update.py` (single update step: `actor_loss`, `critic_loss`, `apply_grads`); `ddpg_service.py` orchestrates the full training loop; `evaluation_service.py` does deterministic rollouts; `comparison_service.py` aggregates across cells; `experiment_service.py` runs sweeps.
- **`sdk/`** — `sdk.py::RoombaLab` is the single facade for consumers; `env_builder.py`, `trainers.py`, `experiments.py` are the constructor layer.
- **`interface/cli/`** — Click group with `train`, `evaluate`, `sweep`, `graphify`, `gui`, `record-gif`, `download-data`.
- **`interface/gui/`** — PyQt6: training-tab + visualisation-tab.
- **`tools/graphify/`** — ported from Assignment 4: AST walker → Obsidian Vault.

## 3. Class diagram (text)

```
ConfigManager ─── (path, version) ────────────► dict (frozen)
Transition (dataclass) = (state, action, reward, next_state, done)
EpisodeMetrics (dataclass) = (reward, length, coverage, collisions)
TrainResult (dataclass) = (diagnostics: list[StepDiagnostic], final_metrics: EpisodeMetrics)

World(polygons, grid)  ◄─── HouseExpoLoader.load(map_id)
Robot(x, y, theta, radius, dt)
Kinematics.step(pose, action, dt) → new_pose                    (pure)
LidarSensor(n_beams, max_range).scan(pose, world) → np.ndarray  (pure)

RoombaEnv(world, robot, lidar, reward_fn)
    .reset() → obs
    .step(action) → (obs, reward, done, info)
    .render()  (matplotlib + agg backend for headless)

Actor(obs_dim, act_dim, hidden_sizes)        nn.Module
Critic(obs_dim, act_dim, hidden_sizes)       nn.Module
ActorCriticNet(actor, critic, target_actor, target_critic)
polyak_update(target_params, source_params, tau)

ReplayBuffer(capacity, obs_dim, act_dim)
    .push(transition)
    .sample(batch_size) → batch dict

GaussianNoise(act_dim, sigma)  /  OUNoise(act_dim, theta, mu, sigma)
SigmaSchedule(initial, final, decay_steps)

ddpg_update.actor_loss(net, batch)
ddpg_update.critic_loss(net, batch, gamma)
ddpg_update.apply_update(net, batch, gamma, tau, optims)

DDPGService(net, env, buffer, noise, schedule, hyperparams)
    .fit(total_timesteps) → TrainResult
EvaluationService.rollout(net, env, n_episodes, deterministic) → EpisodeMetrics
ComparisonService.aggregate(reports) → ComparisonReport

RoombaLab(config_path)
    .make_env(map_id)
    .train(...)
    .evaluate(...)
    .run_sweep(kind, ...)
    .graphify(...)
    .predict(state)
```

## 4. DDPG pseudocode (slide 8 — annotated)

```text
Initialise:
    Actor μ(s|θ_μ), Critic Q(s,a|θ_Q)         # random weights
    Target Actor μ' ← μ,  Target Critic Q' ← Q  # hard copy at init
    Replay buffer B ← {}
    Noise process N
    σ ← σ_initial

For t = 1..T:
    a_t = clip(μ(s_t) + σ · N, −1, +1)           # slide 7 — exploration
    Execute a_t in env, observe (s_{t+1}, r_t, done_t)
    B.push( (s_t, a_t, r_t, s_{t+1}, done_t) )

    If |B| > warmup AND t mod 1 == 0:
        batch = B.sample(N=128)

        # === Critic update ===
        y = r + γ · (1 − done) · Q'(s', μ'(s'))   # slide 6 — bootstrapped target
        L_Q = MSE( Q(s, a), y )
        ∇_{θ_Q} L_Q ; Adam step                     # slide 5

        # === Actor update ===
        L_μ = − mean( Q(s, μ(s)) )                  # slide 4 — DPG chain rule
        ∇_{θ_μ} L_μ ; Adam step

        # === Soft target updates (slide 6) ===
        θ'_μ ← τ·θ_μ + (1−τ)·θ'_μ
        θ'_Q ← τ·θ_Q + (1−τ)·θ'_Q

    σ ← linear_decay(σ_initial, σ_final, t, decay_steps)

Return θ_μ, θ_Q, diagnostics
```

## 5. Reward function

```
r_t  =  + new_cell_bonus                              per freshly cleaned cell (sparse)
        + coverage_progress_coef × Δcoverage          dense per-step shaping (Layer 18)
        + completion_bonus                            on the first step where coverage ≥ coverage_target
        − collision_penalty                           if the candidate move would clip a wall (move is then cancelled)
        − step_penalty                                every step (encourages efficient cleaning)
```

## 6. Configuration schema

```jsonc
{
  "version": "1.00",                                // hard-fail mismatch with shared.version
  "seed": <int>,
  "device": "cpu" | "cuda",
  "env": { map_ids, pixels_per_metre, max_episode_steps, gamma },
  "robot": { radius_m, max_linear_speed_mps, max_angular_speed_radps, dt, cleaning_radius_m },
  "sensor": { n_lidar_beams, lidar_max_range_m, fov_degrees },
  "reward": { new_cell_bonus, collision_penalty, step_penalty, completion_bonus, coverage_target, coverage_progress_coef },
  "ddpg": { actor_hidden_sizes, critic_hidden_sizes, actor_lr, critic_lr, gamma, tau, batch_size, replay_capacity, warmup_steps, policy_delay, max_grad_norm },
  "noise": { kind, sigma_initial, sigma_final, decay_steps, ou_theta, ou_mu, ou_sigma },
  "training": { total_timesteps, eval_interval, n_eval_episodes, log_interval },
  "experiments": { noise_sigma_sweep, tau_sweep, ablation_seeds },
  "paths": { results_dir, assets_dir, checkpoints_dir, data_dir, wiki_dir },
  "graphify": { src_dir, output_dir, include_docstrings, include_call_graph }
}
```

## 7. Architectural Decision Records (ADRs)

### ADR-001 — Custom env wrapper, NOT a Gymnasium subclass

**Context**: Spec § 1 explicitly bans `Gymnasium` or `Gazebo`. Many DDPG reference implementations subclass `gym.Env`.
**Decision**: `RoombaEnv` is a plain Python class with `reset()` / `step(action) → (obs, reward, done, info)` mirroring the Gym shape **but with zero `gym`/`gymnasium` imports**.
**Consequence**: Code is 100 % spec-compliant; the trainer needs no Gym dependency; we lose a few minor conveniences (vector-env wrapping) that we re-implement ourselves where needed.

### ADR-002 — Shapely for polygon collision, NumPy for raster coverage

**Context**: Two needs: (a) ray-casting against walls (LIDAR), (b) tracking cleaned vs unvisited area (coverage).
**Decision**: Use **shapely** for the polygon ops (line-segment intersection for LIDAR; point-in-polygon for collision). Use a **NumPy occupancy grid** for cleaned-cell tracking — one bit per grid cell.
**Consequence**: One small dependency (shapely is widely shipped); raster is fast for coverage updates; LIDAR stays accurate at sub-grid resolution.

### ADR-003 — Differential-drive kinematics, not holonomic

**Context**: We need a robot model. Holonomic (any (x,y) move) is unrealistic; Ackermann is over-complicated.
**Decision**: Differential-drive: `(v, ω) → (dx, dy, dθ)` via the unicycle equations. Two action dimensions, matching the spec.
**Consequence**: Realistic for a vacuum; matches the spec's "1- 1 בין רציף דו-ממדי כווקטור" wording perfectly.

### ADR-004 — Polyak (soft) target updates from day 1, no hard-copy mode

**Context**: Slide 6 names soft updates as the central stability mechanism. We need at least one ablation that *removes* them.
**Decision**: Default training uses Polyak with τ=0.005. The hard-copy mode is implemented behind a config flag `ddpg.hard_copy_every_n` (0 ⇒ soft); used only for the **reflection-Q3 ablation** experiment.
**Consequence**: The default code path is the canonical DDPG; the ablation is a clean flip of one flag.

### ADR-005 — Gaussian noise is default; OU is implemented for completeness

**Context**: Lillicrap 2016 used Ornstein-Uhlenbeck noise; subsequent work (TD3, SAC) showed Gaussian is at least as good and simpler.
**Decision**: Default is **Gaussian** with σ-decay schedule. OU lives in `noise/ou.py` and is selectable via `noise.kind = "ou"`. The reflection-Q2 ablation toggles σ=0.
**Consequence**: Spec asks "what is the initial variance you chose" — that question only makes sense for Gaussian, so we pick it as default.

### ADR-006 — Replay buffer is NumPy-only (not torch tensors stored)

**Context**: PyTorch tensors on GPU stored in a 200k-capacity buffer leak GPU memory and complicate seeding.
**Decision**: Store transitions as NumPy arrays. Convert to tensors at sample time inside `services/ddpg_update.py`.
**Consequence**: ~negligible perf hit (CPU→GPU each batch is < 1 ms at batch=128); reproducible RNG; no GPU-memory pressure.

### ADR-007 — `services/experiment_service` lives under `sdk/` (not `services/`)

**Context**: Lesson learned in Assignment 4 Layer 17: the experiment service imports the SDK to construct envs, which is a backwards arrow.
**Decision**: Place `experiments.py` directly under `sdk/`. The CLI imports it; services do not.
**Consequence**: Layering arrow stays strict; no Layer-17-style refactor needed mid-build.

### ADR-008 — `data/` package is for HouseExpo I/O only (not training data — that lives in `memory/`)

**Context**: Two unrelated things named "data" in RL: the replay buffer (transitions) and the apartment dataset (maps). Conflating them is confusing.
**Decision**: `data/` is **only** HouseExpo + map I/O. Replay buffer lives in `memory/` (also a more standard ML term — Lillicrap calls it "memory").
**Consequence**: Reads cleanly; matches L09 slide-6 wording ("מאגר הסידור החוזר" — *experience replay buffer*).

### ADR-009 — Map cache key is the SHA-256 of the JSON content, not the filename

**Context**: HouseExpo files are named `<hash>.json`. We could key the cache by filename.
**Decision**: Key by **content SHA-256**. If the user accidentally edits a JSON the cache invalidates.
**Consequence**: Safe even if maps are mutated; tiny extra hash cost paid once per map load.

## 8. Test plan (high level)

| Module | Test type | Coverage focus |
|---|---|---|
| `shared/config.py` | unit | version-mismatch raises |
| `data/houseexpo_loader.py` | unit + integration | sample JSON → polygons OK |
| `simulator/kinematics.py` | unit | pure-function 4-test battery (zero action, max forward, max turn, combination) |
| `simulator/collision.py` | unit | inside/outside boundary, on-edge |
| `sensor/lidar.py` | unit | hit-distance closed-form on a square room |
| `environment/roomba_env.py` | unit + integration | reset shape, step types, collision penalty, coverage bonus |
| `model/{actor,critic}.py` | unit | shape contracts, gradient flows |
| `model/soft_update.py` | unit | **4-test math battery**: τ=0 (no update), τ=1 (hard copy), τ=0.5 (midpoint), repeated calls converge |
| `memory/replay_buffer.py` | unit | capacity wraparound, batch sampling |
| `noise/{gaussian,ou}.py` | unit | mean/variance match closed-form, OU decay correctness |
| `services/ddpg_update.py` | unit + integration | gradient flow; one step changes weights |
| `services/ddpg_service.py` | integration | smoke 1 000-step run reaches finite TrainResult |
| `sdk/sdk.py` | integration | end-to-end `train → evaluate → save → load → evaluate` chain |
| `interface/cli/main.py` | integration | each subcommand exits 0 on smoke |
| `interface/gui/*` | smoke under offscreen Qt | window opens; train + visualise tabs render |

## 9. Build order summary

See [TODO.md](TODO.md) for the full per-layer Definition of Done.

| Layer | Headline | Commit message stem |
|---|---|---|
| 0 | Scaffold + docs | `Layer 0: scaffold + PRD + PLAN + TODO + per-mechanism PRDs` |
| 1 | shared/ + data/houseexpo_loader + 10-map sample | `Layer 1: shared + data/houseexpo_loader + 10-map sample` |
| 2 | simulator/ kinematics + world + collision | `Layer 2: simulator core (kinematics, world, collision)` |
| 3 | sensor/lidar + environment/roomba_env + reward | `Layer 3: lidar + roomba_env + reward` |
| 4 | model/{actor, critic, actor_critic_network, init, soft_update} | `Layer 4: actor + critic + soft target update` |
| 5 | memory/replay_buffer | `Layer 5: replay buffer` |
| 6 | noise/{gaussian, ou, schedule} | `Layer 6: exploration noise (Gaussian + OU + schedule)` |
| 7 | services/ddpg_update (actor/critic loss + Polyak step) | `Layer 7: DDPG update step` |
| 8 | services/ddpg_service (full training loop) | `Layer 8: DDPG training service` |
| 9 | sdk + cli | `Layer 9: SDK + CLI` |
| 10 | tools/graphify (AST walker → docs/wiki) | `Layer 10: Mini-Graphify port` |
| 11 | services/experiment_service + scripts/run_sweeps.py | `Layer 11: empirical sweeps (noise-σ, tau)` |
| 12 | Visualisation: trajectory overlay + GIF + coverage heatmap | `Layer 12: visualisations` |
| 13 | Audit response (multi-seed sweeps, reproducibility tests) | `Layer 13: audit response` |
| 14 | interface/gui (PyQt6) | `Layer 14: PyQt6 GUI` |
| 15 | notebooks/roomba_lab_walkthrough.ipynb (executed) | `Layer 15: notebook walkthrough` |
| 16 | Final README + executive summary + reflection answers | `Layer 16: final docs + reflection answers + sign-off` |

## 10. Out of scope

- Real-robot deployment (we only train + visualise in sim)
- Multi-robot / multi-agent
- 3-D maps (HouseExpo is intentionally 2-D)
- Anything requiring GPU at training time
- TD3, SAC (mentioned for context; not implemented as main agent)

## 14. Extension points (V3 § 12.1 / § 20.9 # 8 — published stable surfaces)

The following surfaces are stable in the `1.x` series. Any breaking change to
them implies a major-version bump.

| # | Surface | Why it's stable | What you'd hook in |
|---|---|---|---|
| 1 | `configs/setup.json` schema | Hard contract — consumed by `ConfigManager` + every constructor | Add new hyperparameters here, not as function arguments |
| 2 | `data.houseexpo_loader.HouseExpoLoader` | Read-only, content-addressed via SHA-256 (ADR-009) | Plug a different floorplan dataset by writing a class with the same `map_ids() + load(id) → HouseMap` interface |
| 3 | `simulator.kinematics.step_unicycle` | Pure function | Swap to Ackermann / holonomic by writing a same-signature alternative |
| 4 | `sensor.lidar.LidarSensor` | Stateless | Add per-beam noise / occlusion by subclassing and overriding `scan()` |
| 5 | `environment.reward.compute_reward` | Pure | Shape new reward terms (frontier exploration, smoothness penalty) by writing a new pure function with the same signature |
| 6 | `noise/` package | `(action_dim, sigma) → sample()` interface | TD3-style noise smoothing fits here as `noise/smoothed_gaussian.py` |
| 7 | `model.actor.Actor` + `Critic` | Plain `nn.Module` | Swap MLP for LSTM / Conv-on-LIDAR by writing same-signature alternates |
| 8 | `model.soft_update.polyak_update` | Pure | Hard-copy ablation uses `hard_copy()`; TD3 delayed-actor pattern lives here |
| 9 | `services.ddpg_update.{actor_loss, critic_loss, apply_update}` | Pure | TD3 = add a second critic + min-of-two in `critic_loss`; rest unchanged |
| 10 | `sdk.experiments.ExperimentService` | Single `run(kind)` entry point | Add new sweep kinds (γ, batch_size, hidden_sizes) by adding `elif kind == ...` |
| 11 | CLI subcommand registration | `interface/cli/main.py` decorators | New CLI subcommands plug in via `@cli.command()` |
| 12 | GUI tab registration | `interface/gui/main_window.py::MainWindow.__init__` | New `QWidget` subclass + `tabs.addTab(YourTab(), "...")` |

### Stability promise

Functions / classes listed in the table above will not break their input or
output contracts within `1.x`. If a breaking change is necessary, the next
release will be `2.00` and the migration guide will live in this section.

### Why we did NOT add lifecycle hooks

The V3 § 12.1 example suggests `beforeCreate` / `afterUpdate` hook patterns.
roomba-lab does not have an event loop; the training loop is a simple
sequential `for step in range(total_timesteps)`. Adding hook plumbing without a
real consumer would be premature abstraction (V3 § 7.2). The 12 extension
points above cover every concrete future request the author can anticipate.
