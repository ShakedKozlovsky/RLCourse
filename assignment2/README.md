# Assignment 2 — DQN Trading Agent

> **Status: 🚧 Layer 0 — scaffolding and planning. Code layers will land in subsequent commits.**
>
> This README is intentionally short while the project is in setup. The full learning README — with formulas, screenshots, experiment results, and answers to the 12 reflection questions — is built in **Layer 9** (see [`docs/TODO.md`](docs/TODO.md)).

## What this project is

A complete, object-oriented Deep Q-Network system that learns a daily **trading policy** (Sell / Hold / Buy) on Yahoo Finance stock data. The headline algorithm is **Dueling DQN + Double DQN + Prioritized Experience Replay**, with a vanilla DQN baseline used for the mandatory comparative experiment.

This is a *learning* project about Reinforcement Learning, **not** a trading recommendation, predictor, or production system. See [`docs/PRD.md`](docs/PRD.md) §1.

## How to read this repo right now

The thinking and design come first; code follows the plan layer by layer. Start here:

1. **[`docs/PRD.md`](docs/PRD.md)** — problem framing, RL formulation, dataset spec, KPIs.
2. **[`docs/PLAN.md`](docs/PLAN.md)** — architecture, layers, class map, ADRs.
3. **[`docs/TODO.md`](docs/TODO.md)** — layered task list with Definition of Done per layer.
4. Per-mechanism PRDs for the core algorithmic pieces:
   - [`PRD_dqn.md`](docs/PRD_dqn.md) · vanilla baseline
   - [`PRD_dueling.md`](docs/PRD_dueling.md) · `Q(s,a) = V(s) + A(s,a) − mean A`
   - [`PRD_double_dqn.md`](docs/PRD_double_dqn.md) · decoupled select/evaluate target
   - [`PRD_per.md`](docs/PRD_per.md) · sum-tree prioritized replay
   - [`PRD_reward.md`](docs/PRD_reward.md) · baseline + risk-adjusted variants
   - [`PRD_env.md`](docs/PRD_env.md) · Gymnasium-style env
   - [`PRD_features.md`](docs/PRD_features.md) · the 10 state channels
   - [`PRD_data_pipeline.md`](docs/PRD_data_pipeline.md) · yfinance → tensors

## Quickstart (will become real in Layer 1)

```bash
uv sync
cp .env-example .env
uv run python -m dqn_trader.interface.cli --help
```

## Repository layout

```
assignment2/
├── docs/                    # PRD, PLAN, TODO, per-mechanism PRDs
├── src/dqn_trader/          # source code (populated layer-by-layer)
│   ├── shared/              # config, logger, seed, gatekeeper, version
│   ├── data/                # yfinance client, features, splits, scaling, windows
│   ├── environment/         # TradingEnv, RewardFunction, Portfolio
│   ├── model/               # DuelingDQN, target network
│   ├── memory/              # uniform + prioritized replay
│   ├── services/            # data, training, backtest, inference, experiment
│   ├── sdk/                 # TradingSDK facade
│   └── interface/cli + gui  # consumers — never touch model directly
├── tests/                   # unit + integration; ≥85% coverage gate
├── configs/                 # setup.json, rate_limits.json (versioned)
├── data/raw/                # parquet cache for yfinance (git-ignored)
├── results/                 # checkpoints, metrics, plots (git-ignored)
├── assets/                  # screenshots, architecture diagrams
├── notebooks/               # results analysis notebook (Layer 8)
├── pyproject.toml           # uv-managed deps
└── README.md                # this file
```

## Commit conventions

Each layer in [`docs/TODO.md`](docs/TODO.md) becomes one (or a few small) git commits, with messages of the form `Layer N: <summary>`. The professor and grader can read the git log as a story of the build.
