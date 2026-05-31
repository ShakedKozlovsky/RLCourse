# Assignment 3 — REINFORCE + A2C Fitness Recommender

> **Status: 🚧 Layer 0 — scaffolding and planning. Code layers land in subsequent commits.**
>
> The full learning README — with formulas, screenshots, experiment results, and answers to the 5 reflection questions — is built in **Layer 10** (see [`docs/TODO.md`](docs/TODO.md)).

## What this project is

A complete, object-oriented policy-gradient RL system that learns a daily **workout recommendation policy** (Push / Pull / Legs / Cardio / Rest) over an **LSTM-learned world model** built from the Kaggle "600K+ Fitness Exercise & Workout Program" dataset. Two algorithms compared: **REINFORCE** (the original Monte-Carlo policy gradient, Williams 1992) and **A2C** (Advantage Actor-Critic, Mnih et al. 2016).

This is a *learning* project about Reinforcement Learning, **not** a fitness coach or medical recommendation system. See [`docs/PRD.md`](docs/PRD.md) §1.

## Read order

1. **[`docs/PRD.md`](docs/PRD.md)** — problem framing, RL formulation, data spec, KPIs.
2. **[`docs/PLAN.md`](docs/PLAN.md)** — architecture, layers, class map, ADRs.
3. **[`docs/TODO.md`](docs/TODO.md)** — layered task list with Definition of Done per layer.
4. Per-mechanism PRDs:
   - [`PRD_reinforce.md`](docs/PRD_reinforce.md) · the simple policy gradient
   - [`PRD_a2c.md`](docs/PRD_a2c.md) · Actor + Critic with TD-error advantage
   - [`PRD_lstm_world_model.md`](docs/PRD_lstm_world_model.md) · the environment
   - [`PRD_reward.md`](docs/PRD_reward.md) · gain − λ·overload − λ·imbalance
   - [`PRD_env.md`](docs/PRD_env.md) · Gymnasium-style env over the world model
   - [`PRD_data.md`](docs/PRD_data.md) · Kaggle → trajectory
   - [`PRD_action_masking.md`](docs/PRD_action_masking.md) · excellence differentiator

## Quickstart (will become real in Layer 1+)

```bash
uv sync --extra dev
cp .env-example .env

# Place Kaggle CSVs in data/raw/:
#   program_summary.csv
#   programs_detailed_boostcamp_kaggle.csv

uv run fitness-rl menu          # interactive guided menu
```

## Repo layout

```
assignment3/
├── docs/                    # PRD, PLAN, TODO, per-mechanism PRDs
├── src/fitness_rl/          # source (populated layer-by-layer)
│   ├── shared/              # config, logger, seed, types, version
│   ├── data/                # Kaggle loader, preprocessor, selector, trajectory
│   ├── environment/         # state, action_space, reward, action_mask, world_env
│   ├── model/               # lstm_world_model, policy_net, actor_critic_net
│   ├── services/            # data, world_model, reinforce, a2c, evaluation, compare
│   ├── sdk/                 # FitnessRL facade
│   └── interface/cli + gui  # consumers — never touch model directly
├── tests/                   # unit + integration; ≥85% coverage gate
├── configs/                 # setup.json (versioned)
├── data/raw/                # Kaggle CSV files (git-ignored)
├── results/                 # checkpoints, metrics, plots (git-ignored)
├── assets/                  # screenshots, plots, diagrams
├── saved_models/            # committed pre-trained checkpoints
├── notebooks/               # results analysis notebook
├── scripts/                 # reproducibility drivers
├── pyproject.toml           # uv-managed deps
└── README.md                # this file
```

## Commit conventions

One commit per layer (per `docs/TODO.md`), with messages of the form `Layer N: <summary>` + bullet body. The git log reads as a build story from planning → data → world model → REINFORCE → A2C → comparison → README.
