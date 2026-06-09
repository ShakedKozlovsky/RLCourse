---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/ppo_update.py
module: proximal_lab.services.ppo_update
tags: [type-module]
---

# proximal_lab.services.ppo_update

## Summary
PPO update step + diagnostics aggregation — factored out of ``ppo_service.py``.

## Public functions

- `proximal_lab.services.ppo_update.ppo_update` — Run K-epoch minibatch PPO update on a filled buffer; return diagnostics.

## Imports

- [[__future__|__future__]]
- [[torch|torch]]
- [[proximal_lab_model_actor_critic_network|proximal_lab.model.actor_critic_network]]
- [[proximal_lab_services_ppo_clip|proximal_lab.services.ppo_clip]]
- [[proximal_lab_services_rollout_buffer|proximal_lab.services.rollout_buffer]]