---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/ppo_clip.py
module: proximal_lab.services.ppo_clip
tags: [type-module]
---

# proximal_lab.services.ppo_clip

## Summary
PPO clipped surrogate — the headline math (Schulman et al. 2017 Eq. 7).

## Public functions

- `proximal_lab.services.ppo_clip.ppo_clip_loss` — Compute the PPO clipped surrogate loss.
- `proximal_lab.services.ppo_clip.approx_kl` — Approximate KL divergence per Schulman's blog: ``E[log(p_old) − log(p_new)]``.

## Imports

- [[__future__|__future__]]