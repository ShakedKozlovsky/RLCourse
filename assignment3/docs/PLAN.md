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
│    ReinforceService · A2CService                         │
│    EvaluationService · ComparisonService                 │
└──────┬────────────┬───────────┬────────────┬─────────────┘
       │            │           │            │
       ▼            ▼           ▼            ▼
┌─────────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐
│ environment/│ │ model/   │ │ data/  │ │ shared/    │
│ State       │ │ LSTMWorld│ │ Kaggle │ │ config     │
│ ActionSpace │ │ PolicyNet│ │ Selector│ │ logger     │
│ Reward      │ │ ActorCrit│ │ TrajBld │ │ seed       │
│ WorldEnv    │ │ Mask     │ │ FeatEng │ │ types      │
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
├── environment/
│   ├── state.py                      # State dataclass + state_vec()
│   ├── action_space.py               # 5-action discrete space
│   ├── reward.py                     # gain − λ·overload − λ·imbalance
│   ├── action_mask.py                # guardrails (3-consecutive-same-group block)
│   └── world_env.py                  # Env wrapping the LSTM transition model
├── model/
│   ├── lstm_world_model.py           # f_φ(s_t, a_t, h_t) → s_{t+1}
│   ├── policy_network.py             # REINFORCE actor π_θ
│   └── actor_critic_network.py       # A2C: shared trunk + actor + critic heads
├── services/
│   ├── data_service.py               # end-to-end data pipeline
│   ├── world_model_service.py        # supervised LSTM training
│   ├── reinforce_service.py          # REINFORCE training loop
│   ├── a2c_service.py                # A2C training loop
│   ├── evaluation_service.py         # greedy policy rollout + metrics
│   └── comparison_service.py         # REINFORCE vs A2C side-by-side
├── sdk/
│   └── sdk.py                        # FitnessRL facade
└── interface/
    ├── cli/main.py                   # Click CLI with interactive menu
    └── gui/                          # PyQt6 GUI (5 tabs)
        ├── main_window.py
        ├── data_tab.py
        ├── world_model_tab.py
        ├── reinforce_tab.py
        ├── a2c_tab.py
        └── compare_tab.py
```

Every file ≤ 150 LOC. When approaching, split by responsibility.

**Documented exception** (Layer 12): `sdk/sdk.py` is the public facade and is currently 157 LOC. Splitting further would either inline tiny accessors as one-liners (hurts readability) or move private helpers to a separate module that re-imports the SDK's private state (anti-pattern). The Layer-12 evaluator / baseline / diagnostic methods all live in `sdk/evaluator.py` so the SDK proper only carries training, prediction, and the env-construction passthrough. The exception is scoped strictly to `sdk/sdk.py`.

## 3. Class diagram (textual UML)

```
                       ┌──────────────┐
                       │ FitnessRL    │
                       │--------------│
                       │ prepare_data │
                       │ train_world  │
                       │ train_reinf  │
                       │ train_a2c    │
                       │ compare      │
                       │ predict      │
                       └──┬───────────┘
                          │ uses
   ┌──────────────────────┼──────────────────────────┐
   ▼                      ▼                          ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ DataService  │  │ WorldModelService│  │ ReinforceService   │
│ A2CService   │  │ EvaluationService│  │ ComparisonService  │
└──────┬───────┘  └─────────┬────────┘  └─────────┬──────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌──────────────┐    ┌──────────────────┐  ┌────────────────┐
│ KaggleLoader │    │ LSTMWorldModel   │  │ WorldEnv       │
│ Selector     │    │ PolicyNet        │  │ State          │
│ Trajectory   │    │ ActorCriticNet   │  │ ActionSpace    │
│ FeatureEng   │    │                  │  │ RewardFunction │
└──────────────┘    └──────────────────┘  │ ActionMask     │
                                          └────────────────┘
```

`ReinforceService` and `A2CService` share an `RLAgent` interface (acts via `act()`, learns via `update()`). The same `WorldEnv` instance is used for both — that's the comparison's fairness guarantee.

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
- Coverage gate: ≥ 85% statements and branches.
- TDD pairs documented in README:
  1. `RewardFunction.compute` — write test first (specific reward values for known inputs).
  2. `LSTMWorldModel.forward` — write test first (shape contract + finite outputs).
- Smoke training tests run for 2 episodes on the synthetic trajectory in seconds.
- GUI smoke under `QT_QPA_PLATFORM=offscreen`.
