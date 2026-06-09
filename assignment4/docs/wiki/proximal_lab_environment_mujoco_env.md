---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/environment/mujoco_env.py
module: proximal_lab.environment.mujoco_env
tags: [type-module]
---

# proximal_lab.environment.mujoco_env

## Summary
MuJoCo env wrapper + running-mean / running-std observation normaliser.

## Classes

- `proximal_lab.environment.mujoco_env.RunningMeanStd` — Welford's online algorithm — numerically stable streaming mean + variance.
- `proximal_lab.environment.mujoco_env.NormalisedEnv` — ``gym.Env`` wrapper that maintains a ``RunningMeanStd`` over observations.

## Public functions

- `proximal_lab.environment.mujoco_env.make_env` — Construct a single ``NormalisedEnv`` ready for PPO use.

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]