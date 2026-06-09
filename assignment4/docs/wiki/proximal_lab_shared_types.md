---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/shared/types.py
module: proximal_lab.shared.types
tags: [type-module]
---

# proximal_lab.shared.types

## Summary
Typed structures shared across layers (rollouts, episode metrics, train results).

## Classes

- `proximal_lab.shared.types.RolloutBatch` — One PPO rollout's worth of transitions, post-GAE.
- `proximal_lab.shared.types.EpisodeMetrics` — Per-episode aggregates emitted by the env wrappers + collector.
- `proximal_lab.shared.types.IterationDiagnostics` — Per-PPO-iteration diagnostics — slide-21 stability pillars in one struct.
- `proximal_lab.shared.types.TrainResult` — Outcome of ``PPOService.fit`` — per-iteration diagnostics for plotting + analysis.

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]