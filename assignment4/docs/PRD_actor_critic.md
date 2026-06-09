# PRD — Actor-Critic Network (continuous action space)

## Theory recap

For continuous action spaces, the policy outputs the *parameters* of an action distribution rather than a categorical distribution over discrete actions. Standard PPO + MuJoCo convention: **diagonal Gaussian** policy.

```
π_θ(a | s) = N(a; μ_θ(s), diag(σ²))
log_std = θ_log_std (state-INDEPENDENT learned parameter vector)
σ = exp(log_std), clamped to [e^{−5}, e^{2}]
```

Where:
- `μ_θ(s)` is the actor MLP output: `obs → 64 → 64 → action_dim` with `tanh` activations.
- `log_std` is a *separate learned tensor* of shape `(action_dim,)` — NOT a network output. This is the SB3 / RLLib convention and the one most pretrained MuJoCo PPO policies use.
- The critic is a separate MLP: `obs → 64 → 64 → 1` with `tanh` activations.

## Why state-independent log_std?

A state-dependent `log_std(s)` lets the network shrink variance in well-known regions and grow it in unfamiliar ones — sounds appealing, but in practice it destabilises early training: the network learns to predict tiny variance in the regions it's already seen, which kills exploration before learning has happened. State-independent `log_std` keeps exploration uniform across observation space.

## Why separate actor and critic networks?

Assignment 3's ADR-007 documented the **trunk double-step** problem: when actor and critic share a trunk and use two optimisers, the trunk gets updated with `actor_lr + critic_lr` per step. Two clean ways to fix it:

1. **Separate networks** (our default here, ADR-002 in PLAN.md) — no shared parameters, no double-step.
2. **Shared trunk with shared optimizer** — combine actor + critic loss with the value coefficient, one Adam.

For continuous control PPO, option 1 is more common because the value function and policy function have different stationarity requirements. Option 2 is ablated via `actor_critic.shared_trunk: true` in config.

## Orthogonal initialisation

PPO conventionally initialises every layer with orthogonal weights and zero bias, with specific gains:
- Hidden layers: `gain = sqrt(2)` (for `tanh`).
- Actor output (mean head): `gain = 0.01` — keep initial actions near zero so the policy doesn't slam the joints.
- Critic output: `gain = 1.0` — values can have any scale.

This recipe is from the *Implementation Matters* paper (Engstrom et al. 2020), which showed the gain choices matter as much as the algorithm itself.

## Network classes

```python
# model/actor.py
class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        ...
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, obs: torch.Tensor) -> Normal:
        mu = self.mlp(obs)
        log_std = self.log_std.expand_as(mu).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, log_std.exp())

# model/critic.py
class Critic(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).squeeze(-1)  # (B,)

# model/actor_critic_network.py
class ActorCriticNet(nn.Module):
    def act(self, obs) -> tuple[action, log_prob, value]:
        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs)
        return action, log_prob, value
```

The `.sum(-1)` collapses the per-action-dim log-probs into a single per-step log-prob — the standard convention for diagonal Gaussians.

## Acceptance criteria

- `test_actor_critic_network.py::test_forward_shapes` — `(B, 17)` obs → mean `(B, 6)`, value `(B,)`.
- `test_actor_critic_network.py::test_log_std_clamped` — `log_std` parameter stays inside `[log_std_min, log_std_max]` after optimisation.
- `test_actor_critic_network.py::test_save_load_round_trip` — saving + loading produces identical outputs.
- `test_actor_critic_network.py::test_separate_optimizer_paths` — actor optimiser does not see critic params and vice versa (per ADR-002).

## Where this lives

- `src/proximal_lab/model/actor.py` — `GaussianActor` ≤ 50 LOC.
- `src/proximal_lab/model/critic.py` — `Critic` ≤ 40 LOC.
- `src/proximal_lab/model/actor_critic_network.py` — `ActorCriticNet` wrapper ≤ 70 LOC.
- `src/proximal_lab/model/init.py` — orthogonal init helpers ≤ 30 LOC.

## Caveats

- Diagonal Gaussian assumes action dimensions are independent. For MuJoCo this is fine; for highly coupled action spaces (e.g. quadrotor full-state) one would consider a full covariance matrix.
- The output `mu(s)` is not bounded to `[−1, 1]` (the env's action space). We rely on the env's `Box` action space clipping at step time. An alternative is `tanh(mu(s))` squashing — adds complexity, not needed here.

## Sources

- L. Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," ICLR, 2020.
- Schulman 2017 PPO paper Appendix B (network architecture details).
- L08 lecture slide 17 (Actor-Critic combined framework).
