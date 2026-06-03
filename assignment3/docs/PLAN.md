# PLAN — Architecture & Implementation Plan

> Reference: [`docs/PRD.md`](PRD.md), lecture L07, coding rules V3.

## 1. Layered architecture

Strict layering — inner layers must never import from outer layers. The SDK is the **only** entry point for CLI, GUI, and notebooks.

```
┌──────────────────────────────────────────────────────────┐
│  Interface layer  (src/fitness_rl/interface/)            │
│    cli/             gui/                                 │
└────────────────────────┬─────────────────────────────────┘
                         │  (only allowed entry point)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  SDK  (src/fitness_rl/sdk/sdk.py)                        │
│    FitnessRL — facade over all services                  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Services  (src/fitness_rl/services/)                    │
│    DataService · WorldModelService                       │
│    ReinforceService · A2CService · PPOService            │
│    EvaluationService · ComparisonService                 │
│    WorldModelEvaluator · BaselinePolicies (Layer 12)     │
│    Diagnostics · Recommender (Layer 12/15)               │
│    ExperimentService · ExperimentStudies (Layer 9/13)    │
└──────┬────────────┬───────────┬────────────┬─────────────┘
       │            │           │            │
       ▼            ▼           ▼            ▼
┌─────────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐
│ environment/│ │ model/   │ │ data/  │ │ shared/    │
│ ActionSpace │ │ LSTMWorld│ │ Kaggle │ │ config     │
│ Reward      │ │ PolicyNet│ │ Selector│ │ logger     │
│ ActionMask  │ │ ActorCrit│ │ TrajBld │ │ seed       │
│ WorldEnv    │ │ MuscleCls│ │ FeatEng │ │ types      │
└─────────────┘ └──────────┘ └────────┘ └────────────┘
```

### Dependency rule

```
interface → sdk → services → {environment, model, data} → shared
```

Forward arrows only. `model/` never imports from `services/`. `environment/` never imports from `interface/`.

## 2. Package map

```
src/fitness_rl/
├── __init__.py                       # exposes __version__
├── shared/
│   ├── version.py                    # __version__ = "1.00"
│   ├── config.py                     # ConfigManager — JSON loader + version check
│   ├── logger.py                     # get_logger(name)
│   ├── seed.py                       # set_global_seed
│   └── types.py                      # Action enum, Transition, EpisodeMetrics
├── data/
│   ├── kaggle_loader.py              # read program_summary + detailed CSV
│   ├── preprocessor.py               # clean negatives, parse fields
│   ├── program_selector.py           # pick one program by 8 criteria
│   ├── trajectory_builder.py         # daily aggregation → trajectory
│   └── feature_engineer.py           # state vector construction
├── data/
│   └── muscle_classifier.py          # keyword→muscle group (Layer 1 discovery)
├── environment/
│   ├── action_space.py               # 5-action discrete space
│   ├── reward.py                     # gain − λ·overload − λ·imbalance (Layer 15: REST=0 gain)
│   ├── action_mask.py                # guardrails (3-consecutive-same-group block)
│   └── world_env.py                  # Env wrapping the LSTM transition model
│       (note: state.py was dropped — see PRD evolution v3)
├── model/
│   ├── lstm_world_model.py           # f_φ(s_t, a_t, h_t) → s_{t+1}; also clamps output (Layer 11)
│   ├── policy_network.py             # REINFORCE actor π_θ
│   └── actor_critic_network.py       # A2C/PPO: shared trunk + actor + critic heads
├── services/
│   ├── data_service.py               # end-to-end data pipeline
│   ├── world_model_service.py        # supervised LSTM training
│   ├── world_model_evaluator.py      # persistence + linear baselines + rollout MSE (Layer 12)
│   ├── reinforce_service.py          # REINFORCE training loop
│   ├── a2c_service.py                # A2C training loop
│   ├── ppo_service.py                # PPO clipped surrogate (Layer 15, beyond-spec)
│   ├── evaluation_service.py         # greedy policy rollout + metrics
│   ├── comparison_service.py         # REINFORCE vs A2C side-by-side
│   ├── baseline_policies.py          # Random/RoundRobin/KaggleProgram (Layer 12)
│   ├── diagnostics.py                # greedy trajectory + reward decomposition (Layer 12)
│   ├── recommender.py                # WorkoutRecommender — user-facing (Layer 15)
│   ├── experiment_base.py            # shared train_one, make_sdk, aggregate_with_ci
│   ├── experiment_service.py         # Layer 9: 3 original differentiator experiments
│   └── experiment_studies.py         # Layer 13: 5 audit-driven studies
├── sdk/
│   ├── sdk.py                        # FitnessRL training facade
│   ├── evaluator.py                  # FitnessRLEvaluator — diagnostics companion (Layer 12)
│   ├── env_builder.py                # build_env(cfg, init, world_model) helper
│   └── trainers.py                   # build_*_service constructors (Layer 15)
└── interface/
    ├── cli/main.py                   # Click CLI group
    ├── cli/commands.py               # bulky command bodies (compare/menu/recommend/experiments)
    └── gui/                          # PyQt6 GUI (5 tabs)
        ├── main_window.py
        ├── plot_widget.py            # matplotlib FigureCanvasQTAgg wrapper
        ├── worker.py                 # QThread for off-main-thread training
        ├── data_tab.py
        ├── world_model_tab.py
        ├── algo_tab.py               # shared base for REINFORCE + A2C tabs
        └── compare_tab.py
```

Every file ≤ 150 LOC. When approaching, split by responsibility.

**Documented exception** (Layer 12 / Layer 15): `sdk/sdk.py` is the public facade and is currently 154 LOC. Service constructor logic was factored into `sdk/trainers.py` (Layer 15) to keep this file as close to the cap as possible. Splitting further would either inline tiny accessors as one-liners (hurts readability) or move private helpers to a separate module that re-imports the SDK's private state (anti-pattern). The Layer-12 evaluator / baseline / diagnostic methods all live in `sdk/evaluator.py`. The exception is scoped strictly to `sdk/sdk.py`.

## 3. Class diagram (textual UML)

```
                          ┌──────────────────┐
                          │ FitnessRL        │
                          │ (sdk/sdk.py)     │
                          │------------------│
                          │ prepare_data     │
                          │ train_world_model│
                          │ train_reinforce  │
                          │ train_a2c        │
                          │ train_ppo  (L15) │
                          │ compare          │
                          │ evaluate         │
                          │ predict          │
                          └──┬───────────────┘
                             │ uses
       ┌──────────────────-──┼──────────────────────────────────┐
       ▼                     ▼                                  ▼
┌──────────────┐  ┌──────────────────┐               ┌─────────────────────┐
│ DataService  │  │ WorldModelService│               │ ReinforceService    │
│              │  │ EvaluationService│               │ A2CService          │
│              │  │ ComparisonService│               │ PPOService    (L15) │
└──────┬───────┘  └─────────┬────────┘               └─────────┬───────────┘
       │                    │                                  │
       ▼                    ▼                                  ▼
┌──────────────┐    ┌──────────────────┐               ┌────────────────┐
│ KaggleLoader │    │ LSTMWorldModel   │               │ WorldEnv       │
│ Preprocessor │    │ PolicyNet        │               │ ActionSpace    │
│ Selector     │    │ ActorCriticNet   │               │ RewardFunction │
│ Trajectory   │    │                  │               │ ActionMask     │
│ FeatureEng   │    └──────────────────┘               └────────────────┘
│ MuscleClass. │
└──────────────┘

                          ┌──────────────────────┐
                          │ FitnessRLEvaluator   │     ← Layer 12 companion
                          │ (sdk/evaluator.py)   │
                          │----------------------│
                          │ evaluate_world_model │ → WorldModelEvaluator
                          │ benchmark_baselines  │ → RandomPolicy / RoundRobin / Kaggle
                          │ qualitative_rollout  │ → GreedyTrajectory.as_table()
                          └──────────────────────┘

                          ┌──────────────────────┐
                          │ ExperimentStudies    │     ← Layer 13 audit-driven
                          │----------------------│
                          │ multi_seed_comparison│
                          │ entropy_sweep        │
                          │ reinforce_variant    │
                          │ gamma_ablation       │
                          │ masking_on_lstm_env  │
                          └──────────────────────┘

                          ┌──────────────────────┐
                          │ WorkoutRecommender   │     ← Layer 15 user-facing
                          │ (recommender.py)     │
                          │----------------------│
                          │ recommend(net, env,  │ → WorkoutPlan with per-day
                          │   reward_fn, n_days, │   (gain, overload, imbalance,
                          │   recent_actions)    │    total) breakdown
                          └──────────────────────┘
```

`ReinforceService`, `A2CService`, `PPOService` each implement a `.fit(net, env, episodes)` method returning `list[EpisodeMetrics]`. The same `WorldEnv` instance is used for all three — that's the comparison's fairness guarantee.

## 4. Data flow

```
Kaggle CSV files in data/raw/
   ▼
KaggleLoader.load()                  → raw program_summary + detailed
   ▼
Preprocessor.clean()                 → numeric sanity, type coercion
   ▼
ProgramSelector.pick()               → chosen_program (single row + filter detailed)
   ▼
TrajectoryBuilder.build()            → daily sequence (s_1, ..., s_T)
   ▼
FeatureEngineer.transform()          → state vectors per day
   ▼
WorldModelService.train()            → trained LSTM → world_model.pt
   ▼
WorldEnv(state_dim, world_model)     → fully usable Gym-like env
   ▼
ReinforceService.fit(env)            → policy_reinforce.pt + metrics
A2CService.fit(env)                  → policy_a2c.pt + metrics
   ▼
ComparisonService.compare()          → REINFORCE vs A2C report
EvaluationService.evaluate(policy)   → action distribution + episode metrics
```

## 5. State vector specification

```
state_t = [
  volume_normalized_t,           # 1d total_volume / max_volume_in_traj
  muscle_dist_t (5d),            # normalized volume per group
  session_duration_norm_t,       # 1d session minutes / 120
  week_index_norm_t,             # 1d week / program_length_weeks
  day_in_cycle_t (7d one-hot),   # day of week one-hot
  rest_indicator_t,              # 1d 1.0 if rest day else 0.0
]                                # total = 1 + 5 + 1 + 1 + 7 + 1 = 16 dims
```

State dimension `D_s = 16`. Action dimension `D_a = 5` (one-hot for LSTM input).

## 6. LSTM World Model architecture

```
Input  (B, T, D_s + D_a)  ── concatenated [state, action_one_hot]
   ▼
LSTM(input=D_s + D_a, hidden=64, num_layers=1, batch_first=True)
   ▼
last hidden h_T  (B, 64)
   ▼
Linear(64, D_s)  →  predicted s_{t+1}
```

Training:
- Rolling windows of length `W=7` extracted from the trajectory.
- Input: `(s_{t-W+1}, a_{t-W+1}), ..., (s_t, a_t)`. Target: `s_{t+1}`.
- Loss: MSE on the predicted state vector.
- Optimizer: Adam, lr=1e-3, 100 epochs, early stopping on val loss.

## 7. Policy network (REINFORCE)

```
Input  s_t  (B, 16)
   ▼
Linear(16, 128) → ReLU
Linear(128, 128) → ReLU
Linear(128, 5)            ← actor logits
softmax → π_θ(a|s)
```

Sample `a_t ~ Categorical(π_θ(a|s))` during training; argmax during evaluation.

## 8. Actor-Critic network (A2C)

```
Input  s_t  (B, 16)
   ▼
shared trunk:
  Linear(16, 128) → ReLU
  Linear(128, 128) → ReLU
   ├── Actor head:  Linear(128, 5)   → π_θ(a|s)
   └── Critic head: Linear(128, 1)   → V_ψ(s)
```

Same trunk size as REINFORCE for fair comparison (only the critic head is extra).

## 9. Architectural Decision Records (ADRs)

- **ADR-001: 5-action discrete space.** Rationale: balances diversity (push/pull/legs/cardio/rest) with simplicity; matches the slide's "4-8 categories" example. Trade-off: less fine-grained than per-muscle-group, but easier convergence and clearer analysis.

- **ADR-002: LSTM as world model (not as policy).** Rationale: the slide explicitly distinguishes World Model (Part C) from Policy (Parts D & E). Keeping them separate makes the credit-assignment story clean. Trade-off: extra training step, but the model is reused for both REINFORCE and A2C.

- **ADR-003: One synthetic trainee, fixed trajectory.** Rationale: the assignment §7.2.4 says "build one synthetic trainee". Limiting to one keeps the evaluation fair across REINFORCE and A2C. Trade-off: generalization untested — explicitly noted in §F discussion.

- **ADR-004: 28-day fixed-length episodes.** Rationale: matches the assignment §7.4.1 example. Trade-off: doesn't simulate full multi-month programs; sufficient for algorithmic comparison.

- **ADR-005: Mean-baseline REINFORCE (not learned).** Rationale: keeps REINFORCE as the "simple" baseline that A2C is meant to improve upon. Trade-off: not state-of-the-art REINFORCE; matches the lecture's pedagogical progression.

- **ADR-006: Action Masking as opt-in differentiator.** Rationale: the assignment §F.1 explicitly calls this out as an excellence direction. Implemented but disabled by default so the comparison is "vanilla" REINFORCE vs A2C; enabled in one experiment to demonstrate effect.

- **ADR-007 (Layer 5): A2C trunk owned by actor optimizer only.** Rationale: the naive "trunk in both optimizers" pattern double-steps the trunk with `actor_lr + critic_lr` per update. `ActorCriticNet.actor_params()` returns trunk + actor head; `critic_params()` returns only the critic head. A unit test asserts the partition is disjoint.

- **ADR-008 (Layer 11): Reward imbalance is action-conditional, not state-conditional.** Rationale: with state-based conditioning, a policy could exploit the LSTM's `rest_indicator` prediction to collect zero-imbalance reward regardless of action. Fix: `RewardFunction.compute(state, action)` zeroes imbalance only when `action == REST`. See PRD evolution v6.

- **ADR-009 (Layer 11): LSTM rollouts clamp predicted state to valid feature ranges.** Rationale: the LSTM is an unconstrained regression head; without clamping it can produce negative volumes / muscle distributions summing ≠ 1, feeding OOD states to the policy. `_clamp_state()` in `lstm_world_model.py` snaps volume to [0,1], renormalises muscle distribution, snaps day-of-week to one-hot, clips rest_indicator.

- **ADR-010 (Layer 12): SDK companion class for diagnostics, not method explosion.** Rationale: adding `evaluate_world_model` + `benchmark_baselines` + `qualitative_rollout` to `FitnessRL` would push it well past the 150-LOC cap. They live on `FitnessRLEvaluator(sdk)` instead — same SDK state, separate file.

- **ADR-011 (Layer 13): `ExperimentStudies` as a separate class from `ExperimentService`.** Rationale: the Layer-9 originals (`run_action_masking_ablation`, `run_reward_weight_sweep`, `run_collapse_analysis`) and the Layer-13 audit-driven additions share infrastructure (`train_one`, `make_sdk`, `aggregate_with_ci`) but answer different questions. Split into two classes with shared `experiment_base.py` helpers.

- **ADR-012 (Layer 15): REST earns zero gain.** Rationale: Layer 13's entropy sweep proved the prior reward made "rest forever" optimal — higher entropy bonus diversified actions but lowered reward. Direct fix: `gain = volume if action != REST else 0`. REST still pays the overload penalty so resting still has a cost. See PRD evolution v8.

- **ADR-013 (Layer 15): PPO added as third algorithm, beyond-spec.** Rationale: the assignment requires REINFORCE + A2C only, but the lecture's REINFORCE → A2C arc naturally extends to PPO. Implementing it shows depth without competing with the spec deliverables. Documented as beyond-spec in `PRD_ppo.md` and the executive summary.

## 10. Configuration schema

`configs/setup.json`:

```jsonc
{
  "version": "1.00",
  "seed": 208904839,
  "data": {
    "raw_dir": "data/raw",
    "program_summary_csv": "program_summary.csv",
    "programs_detailed_csv": "programs_detailed_boostcamp_kaggle.csv",
    "equipment_filter": "Full Gym",
    "min_program_weeks": 4,
    "max_program_weeks": 12,
    "min_time_per_workout": 45,
    "max_time_per_workout": 120
  },
  "env": {
    "state_dim": 16,
    "n_actions": 5,
    "episode_length": 28,
    "gamma": 0.99,
    "reward_gain_weight": 1.0,
    "reward_overload_lambda": 0.2,
    "reward_imbalance_lambda": 0.3,
    "action_masking_enabled": false,
    "max_same_group_consecutive": 2,
    "max_rest_consecutive": 2
  },
  "world_model": {
    "hidden_size": 64,
    "num_layers": 1,
    "window_size": 7,
    "epochs": 100,
    "batch_size": 32,
    "lr": 0.001,
    "early_stop_patience": 10
  },
  "reinforce": {
    "episodes": 300,
    "lr": 0.001,
    "use_baseline": true,
    "policy_hidden": 128,
    "entropy_bonus": 0.0
  },
  "a2c": {
    "episodes": 300,
    "actor_lr": 0.0005,
    "critic_lr": 0.001,
    "hidden": 128,
    "entropy_bonus": 0.01
  },
  "ppo": {                                  // Layer 15
    "episodes": 300,
    "lr": 0.0005,
    "hidden": 128,
    "clip_eps": 0.2,
    "n_epochs_per_batch": 4,
    "entropy_coef": 0.01
  },
  "paths": {
    "data_raw_dir": "data/raw",
    "results_dir": "results",
    "assets_dir": "assets",
    "checkpoints_dir": "saved_models"
  }
}
```

## 11. Reproducibility plan

- `seed` from config applied via `shared/seed.set_global_seed` to Python, NumPy, and PyTorch.
- `uv.lock` pinned via `uv lock`; CI installs via `uv sync`.
- A `results/run_<ts>/` directory per run with `config_snapshot.json`, `git_hash.txt`, `metrics.csv`, `plots/*.png`.

## 12. Testing strategy

- Unit tests mirror `src/` under `tests/unit/`. Integration tests under `tests/integration/`.
- Coverage gate: ≥ 85 % statements and branches. **Actual: 97.07 %**.
- TDD pairs documented in README:
  1. `RewardFunction.compute` — write test first (specific reward values for known inputs).
  2. `LSTMWorldModel.forward` — write test first (shape contract + finite outputs).
- Smoke training tests run for 2 episodes on the synthetic trajectory in seconds.
- GUI smoke under `QT_QPA_PLATFORM=offscreen`.
- **Reproducibility test** (Layer 11): two SDK runs with the same seed produce bit-identical training histories — catches silent `torch.manual_seed` regressions.
- **PPO importance-ratio test** (Layer 15): direct check of the clipped surrogate math (`min(r·A, clip(r, 1-ε, 1+ε)·A)`) at the tensor level — independent of training-loop behaviour.

## 13. Audit-driven layers (Layers 11–15)

After Layer 10 the project was subjected to an adversarial review surfacing 20 weaknesses; Layers 11–15 closed every one. Each layer is one commit + one section in README §9 + one set of result files:

| Layer | Theme | Headline deliverable |
|---|---|---|
| 11 | Correctness | Reward action-conditional + LSTM state clamping + reproducibility test + POMDP framing fix |
| 12 | Evaluation infra | `WorldModelEvaluator` (baselines + rollout MSE) + `BaselinePolicies` (random / round-robin / Kaggle) + `Diagnostics` (qualitative + decomposition) |
| 13 | Empirical studies | Multi-seed CI + entropy sweep proving reward mis-spec + REINFORCE chain + γ ablation + masking on LSTM env |
| 14 | Documentation closeout | Architecture diagram + Layer-13 plots + README §9 audit-findings table + coverage 96.6 → 97.5 % |
| 15 | Beyond-spec polish | Reward fix (REST = 0 gain) + PPO + `recommend` feature + 300-episode × 3-seed run + Jupyter notebook + executive summary |

The full mapping of all 20 audit findings to their fixes is in the README's §9.7 status table.
