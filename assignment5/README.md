# roomba-lab — Custom DDPG cleaning robot on real HouseExpo floorplans

> **Assignment 5** of the RL Course. Built layer-by-layer; this README starts as a placeholder and is rewritten at Layer 16.

The headline mandate of the exercise: build a **2-D cleaning-robot simulator from scratch** (no Gymnasium, no Gazebo), run it on **real apartment layouts from the HouseExpo dataset**, and train a **DDPG agent** to navigate and clean it. The exercise specifically forbids any pre-built simulation platform.

## Documentation entry points

- [`docs/PRD.md`](docs/PRD.md) — main Product Requirements Document
- [`docs/PLAN.md`](docs/PLAN.md) — layered architecture, ADRs, pseudocode
- [`docs/TODO.md`](docs/TODO.md) — 17-layer build plan with DoD per layer
- Per-mechanism PRDs: [DDPG](docs/PRD_ddpg.md), [simulator](docs/PRD_simulator.md), [LIDAR sensor](docs/PRD_lidar_sensor.md), [soft-target updates](docs/PRD_soft_target_updates.md), [exploration noise](docs/PRD_exploration_noise.md)

## Status

Layer 0 — scaffold + docs. Implementation begins at Layer 1.
