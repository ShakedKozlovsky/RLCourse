# RLCourse - Reinforcement Learning Course Projects

This repository contains all assignments and projects for the Reinforcement Learning course.

---

## 📚 Assignments

### [Assignment 1: Grid-Based Drone RL Simulator](assignment1/)
**Grid-based drone navigation using Q-Learning**

- **Algorithm:** Tabular Q-Learning with Bellman updates
- **Environment:** 2D grid (20×20) with obstacles, traps, and wind zones
- **Features:** 
  - Interactive visualization with pygame
  - Real-time training metrics
  - Save/load trained agents
  - Fast forward mode for accelerated training
- **Tech Stack:** Python, Gymnasium, Pygame-CE, NumPy
- **Status:** ✅ Complete

**Quick Start:**
```bash
git clone https://github.com/ShakedKozlovsky/RLCourse.git
cd RLCourse/assignment1
./run_grid_simulator.sh
```

---

### [Assignment 3: REINFORCE + A2C Fitness Recommender](assignment3/)
**Policy-gradient daily workout recommendation over an LSTM-learned world model**

- **Algorithms:** REINFORCE (Williams 1992) + A2C (Mnih et al. 2016)
- **World model:** 1-layer LSTM trained supervised on the per-day trajectory (3.2× lower MSE than persistence)
- **Dataset:** Kaggle "600K+ Fitness Exercise & Workout Program" — 2 598 programs, 605 033 detailed rows
- **Action space:** 5 discrete actions — `PUSH / PULL / LEGS / CARDIO / REST`
- **Features:**
  - Full SDK + Click CLI + 5-tab PyQt6 GUI
  - Multi-seed comparison with 95 % CI, entropy sweep, γ ablation, REINFORCE → +baseline → +advantage chain
  - Baseline policies (random, round-robin, Kaggle program) to anchor the trained-agent reward
  - Per-step reward decomposition + qualitative 28-day rollout
  - Action masking as the excellence differentiator
- **Tech Stack:** Python 3.12, PyTorch, NumPy, Gymnasium, PyQt6, Click, matplotlib, ruff, pytest, uv
- **Quality bar:** 216 tests · 97.56 % branch coverage · ruff clean · every src file ≤ 150 LOC
- **Status:** ✅ Complete (14 layers — incl. 4-layer professor's-audit response)

**Quick Start:**
```bash
cd RLCourse/assignment3
uv sync --extra dev
# place Kaggle CSVs in data/raw/
uv run fitness-rl menu          # interactive numeric menu
uv run fitness-rl gui           # PyQt6 GUI
uv run fitness-rl compare --episodes 60
```

---

### [Assignment 4: PPO + GAE on MuJoCo Continuous Control](assignment4/)
**Proximal Policy Optimization + Generalized Advantage Estimation laboratory**

- **Algorithms:** PPO (Schulman et al. 2017) + GAE (Schulman et al. 2016)
- **Environments:** HalfCheetah-v5 (primary) + Walker2d-v5 (secondary) — canonical PPO MuJoCo benchmarks
- **Action space:** Continuous `Box(-1, 1, (6,))` torque control
- **Networks:** Gaussian actor + scalar critic, separate networks, Engstrom 2020 orthogonal init
- **Features:**
  - Verbatim PPO clipped surrogate + GAE recursion with 4-test math batteries each
  - Multi-seed λ/γ/clip-ε empirical sweeps with 95% CIs
  - λ-sweep empirically validates the slide-16 bias-variance dial (peak at λ=0.95)
  - Cross-env transfer (best HalfCheetah config → Walker2d +381 reward in 30k steps)
  - **Mini-Graphify tool** (the originality hook) — walks the project's Python AST and emits an Obsidian-compatible knowledge graph (70 nodes, 128 edges)
  - SDK + Click CLI + 3-tab PyQt6 GUI + executed Jupyter walkthrough
- **Tech Stack:** Python 3.12, PyTorch, MuJoCo, Gymnasium, NumPy, PyQt6, Click, matplotlib, ruff, pytest, uv
- **Quality bar:** 120 tests · 97.25% branch coverage · ruff clean · every src file ≤ 150 LOC
- **Status:** ✅ Complete (15 layers)

**Quick Start:**
```bash
cd RLCourse/assignment4
uv sync --extra dev
uv run proximal-lab train --env-id HalfCheetah-v5 --total-timesteps 50000
uv run proximal-lab sweep lambda                # λ-sweep
uv run proximal-lab graphify                     # build docs/wiki/
uv run proximal-lab gui                          # PyQt6 GUI
```

---

## 🗂️ Repository Structure

```
RLCourse/
├── assignment1/          # Grid-based Drone RL Simulator
│   ├── src/             # Source code (27 modular files)
│   ├── docs/            # Complete documentation
│   ├── configs/         # YAML configuration files
│   └── README.md        # Assignment 1 details
│
├── assignment2/          # DQN Stock Trading Agent
│   ├── src/dqn_trader/  # Dueling + Double DQN + PER
│   ├── docs/            # PRD, PLAN, per-mechanism specs
│   ├── tests/           # 135 tests, 97% coverage
│   ├── configs/         # JSON configuration files
│   ├── assets/          # Plots + GUI screenshots
│   └── README.md        # Assignment 2 details
│
├── assignment3/          # REINFORCE + A2C Fitness Recommender
│   ├── src/fitness_rl/   # SDK, services, models, environment, GUI, CLI
│   ├── docs/             # PRD + 7 per-mechanism PRDs + PLAN + TODO
│   ├── tests/            # 216 tests, 97.56% coverage
│   ├── configs/          # setup.json (versioned)
│   ├── assets/           # plots, GUI screenshots, architecture diagram
│   ├── results/          # experiment outputs (layer12, layer13, experiments)
│   ├── saved_models/     # pre-trained world_model.pt
│   ├── scripts/          # plot + diagram + experiment runners
│   └── README.md         # Assignment 3 details
│
│
├── assignment4/          # PPO + GAE on MuJoCo continuous control
│   ├── src/proximal_lab/ # SDK, services, models, environment, GUI, CLI, tools/graphify
│   ├── docs/             # PRD + 5 per-mechanism PRDs + PLAN + TODO + EXECUTIVE_SUMMARY + wiki/
│   ├── tests/            # 120 tests, 97.25% coverage
│   ├── configs/          # setup.json (versioned)
│   ├── assets/           # plots, GUI screenshots
│   ├── results/          # experiment outputs (sweeps, layer11/cross-env)
│   ├── saved_models/     # pre-trained HalfCheetah + Walker2d checkpoints
│   ├── scripts/          # sweep + plot + GUI capture drivers
│   ├── notebooks/        # executed Jupyter walkthrough
│   └── README.md         # Assignment 4 details
│
└── README.md            # This file
```

---

## 📖 Documentation

Each assignment has its own comprehensive documentation:
- Product Requirements (PRD)
- Technical Implementation Plan
- Architecture Documentation
- Complete TODO list with 1000+ tasks

---

## 🚀 Quick Navigation

| Assignment | Topic | Status | Link |
|------------|-------|--------|------|
| 1 | Grid-based Q-Learning | ✅ Complete | [assignment1/](assignment1/) |
| 2 | DQN Stock Trading Agent | ✅ Complete | [assignment2/](assignment2/) |
| 3 | REINFORCE + A2C over LSTM world model (fitness) | ✅ Complete | [assignment3/](assignment3/) |
| 4 | PPO + GAE on MuJoCo continuous control | ✅ Complete | [assignment4/](assignment4/) |

---

## 👤 Author

**Shaked Kozlovsky**  
Email: shaked1221997@gmail.com  
GitHub: [@ShakedKozlovsky](https://github.com/ShakedKozlovsky)

---

## 📜 License

Educational project for Reinforcement Learning course.

---

**Last Updated:** May 2026 (assignment 3 added)
