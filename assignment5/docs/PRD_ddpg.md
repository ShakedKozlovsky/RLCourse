# PRD — DDPG algorithm (the central pedagogical artifact)

> Per-mechanism deep-dive. Companion to [PRD.md](PRD.md). Mapped against L09 § 4–8 and EX05 § Requirements 1–4.

## 1. Why DDPG (vs Q-Learning, DQN, REINFORCE, A2C, PPO)?

This is **reflection question 1** of the spec; this PRD is the contract that the answer in the final README must come back to.

| Algorithm | Action space | Architecture | Killer property | Why it's wrong for vacuum |
|---|---|---|---|---|
| Q-Learning (tabular) | Discrete tiny | Value table | Optimal in small grids | State explosion (LIDAR is continuous) |
| DQN | Discrete | Deep Q | Solves continuous **state** | Still needs discretised actions; combinatorial blow-up |
| REINFORCE | Discrete + continuous | Single policy net | Stochastic PG | Very high variance |
| A2C / PPO | Discrete + continuous | Actor + critic (stochastic) | On-policy stability | Wastes off-policy data (no replay) |
| **DDPG** | **Continuous only** | **Actor (det.) + Critic (Q)** | **Combines value-based efficiency with continuous-space accuracy via Q-max over deterministic π** | (this is the right tool) |

The spec hint is **"deterministic nature of the physical engines"**: a vacuum that consumes 0.5 m/s should move 0.5 m/s — there is no value in modelling a stochastic outcome at the actuator level.

## 2. Discretisation explosion (slide 3)

If we discretised the 2-D action space `(v, ω) ∈ [−1, 1]²` to 100 bins per axis, we'd have 10 000 actions to evaluate per state. With a 7-DoF arm (hinted in slide 3) at 100 bins each, that's 10¹⁴ — `searching over trillions of possibilities is computationally impossible in real time`.

DDPG sidesteps the explosion: the actor outputs the action directly via `tanh`. No enumeration.

## 3. The DPG theorem (slide 4, equation 1)

$$\nabla_{\theta}\, J(\mu_{\theta}) = \mathbb{E}_{s\sim\rho^{\mu}}\!\left[\, \nabla_{\theta}\mu_{\theta}(s)\, \nabla_{a} Q^{\mu}(s, a)\big|_{a=\mu_{\theta}(s)} \,\right]$$

In code, with autograd, this becomes:

```python
# actor loss = the deterministic policy gradient via the chain rule
actor_loss = -critic(state, actor(state)).mean()
# .backward() lets PyTorch handle ∇θ μ · ∇a Q automatically
```

Slide 4's numerical example (toy 1-D robot) is exactly the unit test we'll write for `services/ddpg_update.py::actor_loss`.

## 4. The architecture — actor + critic (slide 5)

```
        μ(s | θ_μ)               Q(s, a | θ_Q)
       ┌─────────┐              ┌──────────────┐
   s ──►  MLP    ├── tanh ──► a │  concat(s, a)│
       └─────────┘              │     MLP      ├── scalar V(s, a)
                                │              │
                                └──────────────┘
                Hidden sizes: [256, 256]
                Optimiser: Adam (actor lr=1e-4, critic lr=1e-3)
```

Concretely:

- `Actor(s) = tanh(MLP(s))` — output is the deterministic action vector, always in `[−1, +1]ᵈ`.
- `Critic(s, a)` — state and action are concatenated **at the input layer**, per slide 5. The critic outputs a scalar Q-value.

## 5. The training step — slide 8 algorithmic flow

```
1. Sample a batch B = {(s, a, r, s', done)} from replay buffer
2. Target Q  y  ← r + γ (1 − done) Q'(s', μ'(s'))         # bootstrapped target
3. Critic loss = MSE(Q(s, a), y) ; Adam step on θ_Q
4. Actor loss  = − mean( Q(s, μ(s)) )                      # DPG, slide 4
5. Adam step on θ_μ
6. Soft target updates:                                   # slide 6
       θ'_μ ← τ θ_μ + (1 − τ) θ'_μ
       θ'_Q ← τ θ_Q + (1 − τ) θ'_Q
```

Critically: **the target networks are used for the bootstrap term only** (step 2). They are NOT used for the actor's own loss (step 4).

## 6. Hyperparameters (justified)

| Param | Value | Why |
|---|---|---|
| actor lr | 1e-4 | Lillicrap 2016 standard |
| critic lr | 1e-3 | Critic targets are noisier; needs to track Q faster |
| γ | 0.99 | Standard discount; horizon ≈ 100 steps ≈ 10 s sim time |
| τ | 0.005 | Spec § Item 3 explicit example ("e.g. 0.005") + Lillicrap default |
| batch size | 128 | Lillicrap default |
| replay capacity | 200 000 | Big enough for a 50 k-step run not to wrap; ~30 MB in float32 |
| warm-up steps | 1 000 | Fill the buffer with random data before any update |
| hidden sizes | [256, 256] | Common DDPG default; Lillicrap used [400, 300] |
| max grad norm | 1.0 | Clip large gradients for stability |

## 7. Acceptance criteria (Definition of Done for the algorithm)

1. `services/ddpg_update.py::actor_loss` and `critic_loss` are pure functions of (net, batch, γ) — no I/O, no globals.
2. Gradient flow:
   - `critic_loss.backward()` ⇒ critic params have `.grad`, target params do not.
   - `actor_loss.backward()` ⇒ actor params have `.grad`, critic params do not.
3. A single update step changes the actor weights by > 1e-6 and the critic weights by > 1e-6.
4. `polyak_update(target, source, τ=0.005)` moves target by exactly 0.005 (verified by elementwise tolerance test).
5. With Gaussian σ=0.2 noise, the agent reaches > 25 % coverage on the primary map within 30 000 steps over 3 seeds.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Q-value explosion during training | Critic gradient clipping (`max_grad_norm=1.0`); soft target updates |
| Actor saturating at ±1 | `tanh` output naturally bounded; orthogonal init with small gain on the actor head |
| Wrong action scaling | The env multiplies the `[−1, 1]` actor output by `(max_v, max_w)` outside the network |
| Buffer wrap loses early data before learning | Capacity > total_timesteps for the default run |
