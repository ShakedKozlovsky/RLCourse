---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/ppo_service.py
module: proximal_lab.services.ppo_service
tags: [type-module]
---

# proximal_lab.services.ppo_service

## Summary
End-to-end PPO + GAE training loop (slide-18 pipeline).

## Classes

- `proximal_lab.services.ppo_service.PPOService` — Train an :class:`ActorCriticNet` with PPO + GAE on a :class:`SyncVectorEnv`.

## Imports

- [[__future__|__future__]]
- [[torch|torch]]
- [[proximal_lab_environment_vector_env|proximal_lab.environment.vector_env]]
- [[proximal_lab_model_actor_critic_network|proximal_lab.model.actor_critic_network]]
- [[proximal_lab_services_ppo_update|proximal_lab.services.ppo_update]]
- [[proximal_lab_services_rollout_buffer|proximal_lab.services.rollout_buffer]]
- [[proximal_lab_shared_logger|proximal_lab.shared.logger]]
- [[proximal_lab_shared_types|proximal_lab.shared.types]]