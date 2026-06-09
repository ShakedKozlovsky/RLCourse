---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/model/actor.py
module: proximal_lab.model.actor
tags: [type-module]
---

# proximal_lab.model.actor

## Summary
Gaussian actor for continuous action spaces — μ(s) MLP + state-independent log_std.

## Classes

- `proximal_lab.model.actor.GaussianActor` — μ(s) = MLP(obs); log_std is a separate learned parameter vector.

## Imports

- [[__future__|__future__]]
- [[torch|torch]]
- [[torch_distributions|torch.distributions]]
- [[proximal_lab_model_init|proximal_lab.model.init]]