# PLAN — Architecture & Implementation Plan (Assignment 4)

> Reference: [`PRD.md`](PRD.md), L08 lecture slides, V3 coding rules.

## 1. Layered architecture

Strict layering — inner layers must never import from outer layers. The SDK is the **only** entry point for CLI, GUI, and notebooks. Same principle as Assignment 3.

```
┌──────────────────────────────────────────────────────────┐
│  Interface  (src/proximal_lab/interface/)                │
│    cli/  ·  gui/                                         │
└────────────────────────┬─────────────────────────────────┘
                         │  (only allowed entry point)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  SDK  (src/proximal_lab/sdk/sdk.py)                      │
│    ProximalLab — facade over all services                │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Services  (src/proximal_lab/services/)                  │
│    PPOService · GAE · RolloutBuffer · EvaluationService  │
│    ExperimentService · DiagnosticsService                │
└──────┬────────────┬───────────┬────────────┬─────────────┘
       │            │           │            │
       ▼            ▼           ▼            ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────┐
│ environment/│ │ model/   │ │ tools/       │ │ shared/ │
│ MuJoCoEnv   │ │ Actor    │ │ graphify/    │ │ config  │
│ VectorEnv   │ │ Critic   │ │              │ │ logger  │
│             │ │ ActorCrit│ │              │ │ seed    │
│             │ │ Network  │ │              │ │ types   │
└─────────────┘ └──────────┘ └──────────────┘ └─────────┘
```

### Dependency rule

```
interface → sdk → services → {environment, model, tools} → shared
```

Forward arrows only. `model/` never imports from `services/`. `environment/` never imports from `interface/`. `tools/graphify/` is a leaf — it reads `src/` filesystem only, not runtime state.

## 2. Package map

```
src/proximal_lab/
├── __init__.py                       # exposes __version__
├── shared/
│   ├── version.py                    # __version__ = "1.00"
│   ├── config.py                     # ConfigManager — JSON loader + version check
│   ├── logger.py                     # get_logger(name)
│   ├── seed.py                       # set_global_seed (Python + NumPy + PyTorch)
│   └── types.py                      # RolloutBatch, EpisodeMetrics, TrainResult dataclasses
├── environment/
│   ├── mujoco_env.py                 # gym.make wrapper with observation normalisation
│   └── vector_env.py                 # SyncVectorEnv wrapper + parallel rollout helper
├── model/
│   ├── actor.py                      # GaussianActor — μ(s) + state-independent log_std
│   ├── critic.py                     # Critic MLP — V(s)
│   ├── actor_critic_network.py       # ActorCriticNet — wraps actor + critic
│   └── init.py                       # orthogonal initialisation (PPO convention)
├── services/
│   ├── rollout_buffer.py             # store (s, a, log_prob, r, v, done) per env step
│   ├── gae.py                        # compute_gae — reverse recursion Eq. (2c)
│   ├── ppo_service.py                # fit(net, env, total_timesteps) — main training loop
│   ├── evaluation_service.py         # greedy + stochastic rollout for evaluation
│   ├── comparison_service.py         # cross-config / cross-seed comparison
│   ├── experiment_service.py         # λ / γ / clip-ε sweeps with CI aggregation
│   └── diagnostics.py                # KL monitor + clip-fraction + explained-variance
├── sdk/
│   ├── sdk.py                        # ProximalLab facade
│   ├── env_builder.py                # build_env(cfg, env_id) helper
│   └── trainers.py                   # build_*_service constructors
├── interface/
│   ├── cli/
│   │   ├── main.py                   # Click group + short commands
│   │   └── commands.py               # train / evaluate / sweep / graphify / gui
│   └── gui/
│       ├── main_window.py            # QMainWindow with tabs
│       ├── plot_widget.py            # matplotlib FigureCanvasQTAgg wrapper
│       ├── worker.py                 # QThread off-main-thread trainer
│       ├── train_tab.py
│       ├── sweep_tab.py
│       └── compare_tab.py
└── tools/
    └── graphify/
        ├── walker.py                 # ast.parse + visitor to collect symbols/edges
        ├── emitter.py                # graph.json + per-module Markdown writer
        └── runner.py                 # CLI entry: proximal-lab graphify
```

Every file ≤ 150 LOC. SDK exception only if it goes over after Layer 12 (precedent in Assignment 3).

## 3. Class diagram (textual UML)

```
                       ┌──────────────────────┐
                       │ ProximalLab          │
                       │ (sdk/sdk.py)         │
                       │----------------------│
                       │ make_env(env_id)     │
                       │ train_ppo(timesteps) │
                       │ evaluate(algo)       │
                       │ run_sweep(kind)      │
                       │ graphify()           │
                       └──┬───────────────────┘
                          │ uses
   ┌──────────────────────┼──────────────────────────┐
   ▼                      ▼                          ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ PPOService   │  │ EvaluationService│  │ ExperimentService  │
│ + GAE        │  │ ComparisonService│  │ + Diagnostics      │
│ + Buffer     │  │                  │  │                    │
└──────┬───────┘  └─────────┬────────┘  └─────────┬──────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌──────────────────┐   ┌────────────────┐    ┌──────────────────┐
│ ActorCriticNet   │   │ MuJoCoEnv      │    │ tools/graphify   │
│ (GaussianActor   │   │ VectorEnv      │    │ Walker + Emitter │
│  + Critic)       │   │                │    │ (independent CLI) │
└──────────────────┘   └────────────────┘    └──────────────────┘
```

`PPOService.fit(net, env, total_timesteps)` is the single training entry point. The same `MuJoCoEnv` is used at training and evaluation — that's the comparison's fairness guarantee.

## 4. Data flow (the PPO + GAE pipeline)

This is the slide-18 pipeline rendered as concrete steps:

```
Loop until total_timesteps reached:
  1. RolloutBuffer.reset()
  2. For step in 1..steps_per_rollout:
       a, log_prob_old, V = ActorCriticNet.act(s)              # actor + critic forward
       s', r, done, _, _ = MuJoCoEnv.step(a)
       RolloutBuffer.add(s, a, log_prob_old, r, V, done)
       s = s'
  3. _, V_last = ActorCriticNet.act(s)                          # bootstrap value
  4. compute_gae(buffer.rewards, buffer.values, V_last,         # Eq. (2c)
                buffer.dones, gamma, lambda) → advantages
  5. returns = advantages + buffer.values
  6. For epoch in 1..n_epochs:
       For minibatch in buffer.minibatches(size):
         r_t = exp(new_log_prob - log_prob_old)
         L_clip = -min(r_t·Â, clip(r_t, 1-ε, 1+ε)·Â).mean()    # Eq. (1)
         L_value = 0.5·(V_new - returns).pow(2).mean()
         L_entropy = -dist.entropy().mean()
         L = L_clip + c_1·L_value + c_2·L_entropy              # Eq. (3)
         optimizer.zero_grad(); L.backward(); clip_grad_norm; step()
  7. Log: episode_reward, mean_KL, clip_fraction, explained_variance
```

## 5. Observation / action specification

Both HalfCheetah-v4 and Walker2d-v4 share the same shapes:

| Dim | Component |
|---|---|
| `obs ∈ R^17` | Joint angles + velocities (env-specific encoding) |
| `action ∈ [−1, 1]^6` | Joint torques (clipped) |
| Reward | Forward velocity + control cost (env-defined) |
| Termination | Walker2d: torso angle / height bounds; HalfCheetah: never (timeout only) |

We standardise observations using a **running-mean / running-std** wrapper (the standard PPO recipe) so the actor sees roughly unit-variance inputs.

## 6. PPO clipped surrogate (Eq. 1) — implementation detail

```python
# inside services/ppo_service.py::ppo_update
ratio = torch.exp(new_log_prob - old_log_prob)
clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
value_loss = 0.5 * (new_values - returns).pow(2).mean()
entropy = dist.entropy().sum(-1).mean()  # sum over action dims for continuous
loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
```

**Convention**: `advantages` are normalised per minibatch (`(advantages − mean) / (std + 1e-8)`) — standard PPO trick that empirically helps continuous control.

## 7. GAE (Eq. 2c) — implementation detail

```python
# inside services/gae.py::compute_gae — reverse recursion
def compute_gae(rewards, values, last_value, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages
```

This is the canonical PPO-style GAE implementation. Crucially, `next_non_terminal` zeroes out the bootstrap on terminal transitions — without it the recursion bleeds across episode boundaries.

## 8. Configuration schema (full reference)

`configs/setup.json`:

```jsonc
{
  "version": "1.00",
  "seed": 208904839,
  "env": {
    "id": "HalfCheetah-v4",
    "secondary_id": "Walker2d-v4",
    "max_episode_steps": 1000,
    "n_parallel_envs": 4,
    "gamma": 0.99
  },
  "gae": { "lambda": 0.95 },
  "actor_critic": {
    "hidden_sizes": [64, 64],
    "activation": "tanh",
    "shared_trunk": false,
    "log_std_init": -0.5,
    "log_std_min": -5.0,
    "log_std_max": 2.0
  },
  "ppo": {
    "total_timesteps": 1000000,
    "steps_per_rollout": 2048,
    "minibatch_size": 64,
    "n_epochs_per_update": 10,
    "clip_eps": 0.2,
    "lr": 0.0003,
    "value_coef": 0.5,
    "entropy_coef": 0.0,
    "max_grad_norm": 0.5,
    "target_kl_stop": null
  },
  "experiments": {
    "lambda_sweep": [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
    "gamma_sweep": [0.9, 0.95, 0.99, 0.999],
    "clip_eps_sweep": [0.1, 0.2, 0.3, 0.5],
    "seeds": [0, 1, 2]
  },
  "paths": {
    "results_dir": "results",
    "assets_dir": "assets",
    "checkpoints_dir": "saved_models",
    "wiki_dir": "docs/wiki"
  },
  "graphify": {
    "src_dir": "src/proximal_lab",
    "output_dir": "docs/wiki",
    "include_docstrings": true,
    "include_call_graph": true
  }
}
```

## 9. Architectural Decision Records (ADRs)

- **ADR-001: MuJoCo continuous control, not Atari / CartPole.** Rationale: slide 19 explicitly cites HalfCheetah / Hopper / Walker as PPO's canonical benchmarks; continuous actions exercise the "PPO suits continuous control" claim. Trade-off: requires `mujoco` system library (auto-installs via pip on modern Linux).

- **ADR-002: Separate actor + critic networks by default.** Rationale: avoids the trunk-double-step issue Assignment 3 documented in ADR-007. Shared trunk is opt-in via `shared_trunk: true` for an ablation. Trade-off: more parameters, but cleaner gradient flow.

- **ADR-003: State-independent log-std for the Gaussian actor.** Rationale: matches Schulman 2017 paper + Stable-Baselines3. State-dependent log-std (network-predicted) destabilises early training. Trade-off: less expressive policy, but convergence is more reliable.

- **ADR-004: GAE recursion lives in a standalone module, not a service.** Rationale: it's a pure function over `(rewards, values, dones, γ, λ)` — no state, no side effects, no logger. Trivial to unit-test against a closed-form reference. Trade-off: slight asymmetry with other "services".

- **ADR-005: Rollout buffer is sized to `steps_per_rollout × n_parallel_envs`.** Rationale: the PPO update sees `steps_per_rollout × n_parallel_envs` transitions per iteration; vectorised envs are the standard way to scale wall-clock without losing on-policy semantics. Trade-off: more memory per iteration; not a problem at 2048 × 4.

- **ADR-006: Advantage normalisation per minibatch.** Rationale: standard PPO trick; empirically improves stability on MuJoCo. Documented in PRD_ppo.md so it's not a "magic" line of code. Trade-off: makes the math slightly non-canonical, but every PPO impl does it.

- **ADR-007: Graphify lives under `tools/`, not `services/`.** Rationale: it operates on the filesystem, not runtime state. Mixing it into services would muddle the dependency graph. Trade-off: requires its own tiny CLI entry; worth it for clean separation.

- **ADR-008: Three-sweep empirical analysis (λ, γ, clip-ε).** Rationale: these are the three knobs PPO + GAE explicitly expose to the user (slides 10, 14, 16). Other ablations (entropy coefficient, max-grad-norm, target-KL) are minor. Trade-off: three sweeps × 3 seeds × ~150k timesteps each ≈ 30 min CPU; affordable.

## 10. Testing strategy

- Unit tests mirror `src/` under `tests/unit/`. Integration tests under `tests/integration/`.
- **Coverage gate**: ≥ 85 % statements and branches. Target: ≥ 95 %.
- **TDD pairs**:
  1. `services/gae.py::compute_gae` — write test first using a hand-computed δ sequence with known closed-form GAE values.
  2. `services/ppo_service.py::ppo_clip_loss` — write test first using known `(ratio, advantage)` pairs.
- **Math correctness tests** (the heart of the test suite):
  - GAE recursion at λ = 0 reduces to TD error.
  - GAE recursion at λ = 1 reduces to Monte-Carlo return − value.
  - PPO clip objective with `r > 1+ε ∧ Â > 0` returns the clipped branch.
  - PPO clip objective with `r > 1+ε ∧ Â < 0` returns the unclipped branch (the "negative-advantage outside clip window" case).
- **Smoke training**: 1000-step run on HalfCheetah produces a trained network with non-zero gradients.
- **Reproducibility**: same seed → bit-identical training history.
- **GUI smoke**: under `QT_QPA_PLATFORM=offscreen`.
- **Graphify smoke**: runs against `src/proximal_lab/` and produces valid `graph.json`.

## 11. Implementation order — 16 layers

This maps to `docs/TODO.md`'s layer list. Brief here, detail there.

| Layer | Theme | Deliverable summary |
|---|---|---|
| 0 | Scaffold + planning | This file, PRD, TODO, README, per-mechanism PRDs, configs/setup.json |
| 1 | Shared + Env | shared/{config,logger,seed,types}, environment/{mujoco_env,vector_env}, basic gym smoke test |
| 2 | Actor-Critic networks | GaussianActor, Critic, ActorCriticNet, orthogonal init |
| 3 | Rollout buffer | RolloutBuffer collecting (s, a, log_prob, r, v, done) |
| 4 | **GAE module** | compute_gae() with all 4 math tests passing (λ=0, λ=1, closed-form, terminal handling) |
| 5 | **PPO clip math** | ppo_clip_loss() with all 4 tests passing (sign × in/out clip window) |
| 6 | PPOService end-to-end | fit(net, env, timesteps) with rollout → GAE → minibatch PPO update loop |
| 7 | Evaluation + comparison | EvaluationService greedy + stochastic; ComparisonService for cross-config |
| 8 | SDK + CLI | ProximalLab facade, Click commands (train / evaluate / sweep / graphify / gui) |
| 9 | **Mini-Graphify tool** | ast walker + emitter; produces docs/wiki/ Obsidian Vault |
| 10 | Empirical sweeps | λ / γ / clip-ε sweeps on HalfCheetah, multi-seed CI |
| 11 | Cross-env generalisation | Best config from Layer 10 transferred to Walker2d-v4 |
| 12 | Notebook walkthrough | 6-cell guided tour (data → train → eval → sweep → graphify → compare) |
| 13 | Audit response | Claude playing professor; fix surfaced gaps |
| 14 | PyQt6 GUI | Train tab + Sweep tab + Compare tab + headless smoke test |
| 15 | Final README + plots + reflection + executive summary + commit + push |

## 12. Reproducibility plan

- `seed` from config applied via `shared/seed.set_global_seed` to Python, NumPy, and PyTorch (CPU + CUDA where available).
- `uv.lock` pinned via `uv lock`; CI installs via `uv sync`.
- Every experiment dumps a `config_snapshot.json` + `git_hash.txt` alongside its JSON results.

## 13. Active Knowledge Architecture realisation

The methodology document positions four artefacts (Raw Folder → Pipeline → Wiki Folder → Obsidian Vault). Here's the mapping for this project:

| Doc layer | This project's realisation |
|---|---|
| **Raw Folder** (incoming material) | `docs/raw/` — to-be-curated PDFs (lecture slides, methodology doc) |
| **Pipeline (Graphify)** | `src/proximal_lab/tools/graphify/` — our mini-implementation |
| **Wiki Folder** | `docs/wiki/` — generated Markdown + `graph.json` |
| **Obsidian Vault** | `docs/wiki/` opens directly in Obsidian (same folder; the .obsidian/ subdir is the indicator) |

The lecturer's "תכנון אנדריי קרפטי" (Karpathy-inspired) workflow becomes runnable: anyone with this repo + Obsidian installed gets the same wiki the maintainer sees.
