---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/evaluation_service.py
module: proximal_lab.services.evaluation_service
tags: [type-module]
---

# proximal_lab.services.evaluation_service

## Summary
Greedy / stochastic evaluation rollouts for a trained ``ActorCriticNet``.

## Classes

- `proximal_lab.services.evaluation_service.EvaluationResult` — Summary of an n-episode evaluation rollout.
- `proximal_lab.services.evaluation_service.EvaluationService` — Run ``n_episodes`` evaluation rollouts and aggregate the returns.

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]
- [[proximal_lab_environment_mujoco_env|proximal_lab.environment.mujoco_env]]
- [[proximal_lab_model_actor_critic_network|proximal_lab.model.actor_critic_network]]