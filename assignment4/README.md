# Assignment 4 — PPO + GAE Laboratory on MuJoCo Continuous Control

> **Status: 🚧 Layer 0 — scaffolding and planning. Code layers land in subsequent commits.**
>
> The full learning README — with formulas, plots, sweep results, audit response, and reflection answers — is built in **Layer 15** (see [`docs/TODO.md`](docs/TODO.md)).

## What this project is

A from-scratch implementation of **PPO** (Schulman et al. 2017) and **Generalized Advantage Estimation** (Schulman et al. 2016) trained on **MuJoCo continuous-control benchmarks** (HalfCheetah-v4, Walker2d-v4), with a **mini-Graphify tool** that auto-generates an Obsidian-compatible knowledge graph from the project's own Python AST. Organised under the *Active Knowledge Architecture* methodology (Obsidian + Graphify + AI Skills).

This is a *learning* project about Reinforcement Learning, **not** a production controller. See [`docs/PRD.md`](docs/PRD.md) §1.

## Read order

1. **[`docs/PRD.md`](docs/PRD.md)** — problem framing, lecture-slide mapping, algorithm specs, KPIs.
2. **[`docs/PLAN.md`](docs/PLAN.md)** — layered architecture, package map, class diagram, ADRs, config schema.
3. **[`docs/TODO.md`](docs/TODO.md)** — layered task list with DoD per layer.
4. Per-mechanism PRDs:
   - [`PRD_ppo.md`](docs/PRD_ppo.md) — the clipped surrogate objective
   - [`PRD_gae.md`](docs/PRD_gae.md) — the bias-variance dial
   - [`PRD_actor_critic.md`](docs/PRD_actor_critic.md) — continuous Gaussian policy + critic
   - [`PRD_mujoco_env.md`](docs/PRD_mujoco_env.md) — the env wrapper + normalisation
   - [`PRD_graphify.md`](docs/PRD_graphify.md) — the Active Knowledge Architecture realisation

## The two core equations (verbatim from the slides)

**PPO clipped surrogate** (slide 10):
```
L^CLIP(θ) = Ê_t[ min( r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]
```

**GAE advantage** (slide 16):
```
Â_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}^V    where    δ_t^V = r_t + γ·V(s_{t+1}) − V(s_t)
```

## Quickstart (will become real in Layer 1+)

```bash
uv sync --extra dev
cp .env-example .env

# Quick smoke training
uv run proximal-lab train --env-id HalfCheetah-v4 --total-timesteps 50000

# Empirical sweeps (Layer 10)
uv run proximal-lab sweep lambda     # → results/sweeps/lambda.json + plot

# Build the wiki (the Graphify payoff)
uv run proximal-lab graphify         # → docs/wiki/ Obsidian Vault

# Interactive
uv run proximal-lab menu              # numeric menu
uv run proximal-lab gui               # PyQt6 GUI
```

## Repo layout

```
assignment4/
├── docs/                    # PRD, PLAN, TODO, per-mechanism PRDs (Layer 0); wiki/ generated Layer 9
├── src/proximal_lab/        # source (populated layer-by-layer)
│   ├── shared/              # config, logger, seed, types, version
│   ├── environment/         # MuJoCo env + vector env wrappers
│   ├── model/               # GaussianActor, Critic, ActorCriticNet
│   ├── services/            # rollout buffer, GAE, PPO, evaluation, experiments
│   ├── sdk/                 # ProximalLab facade
│   ├── interface/cli + gui  # consumers — never touch model directly
│   └── tools/graphify/      # AST walker + Markdown emitter (Layer 9)
├── tests/                   # unit + integration; ≥ 85 % coverage gate
├── configs/                 # setup.json (versioned)
├── assets/                  # screenshots, plots, diagrams
├── results/                 # experiment outputs (git-ignored; force-added in headline commits)
├── saved_models/            # committed pre-trained checkpoints
├── notebooks/               # walkthrough notebook (Layer 12)
├── scripts/                 # reproducibility drivers
├── pyproject.toml           # uv-managed deps
└── README.md                # this file
```

## Commit conventions

One commit per layer (per `docs/TODO.md`), with messages of the form `Layer N: <summary>` + bullet body. The git log reads as a build story: planning → shared → env → networks → buffer → GAE → PPO clip math → PPO service → evaluation → SDK → Graphify → sweeps → cross-env → notebook → audit → GUI → final.

## Acknowledgements

Methodology inspired by the *Active Knowledge Architecture* document (Dr. Yoram Segal, May 2026; NotebookLM). Algorithm content from L08 lecture slides. Implementation conventions inherited from Assignment 3's coding rules + V3 PDF.
