# DDPG vs DQN vs PPO — head-to-head for the cleaning-robot domain

> Reflection-Q1 expanded into a full comparison table. The spec asks "why DDPG
> and not DQN or PPO?" — this document is the long-form answer with citations.

## The 9-axis comparison

| Axis | DDPG (chosen) | DQN | PPO |
|---|---|---|---|
| **Action space** | Continuous only | Discrete only | Continuous + discrete |
| **Actor type** | Deterministic μ(s) → action | Implicit (argmax over Q) | Stochastic π(a\|s) |
| **Critic / value** | Q(s, a) — state + action | Q(s, a) per action | V(s) |
| **Policy gradient** | DPG theorem: ∇θ J = E[∇θμ · ∇aQ] | — | Clipped surrogate: L^CLIP |
| **Data efficiency** | Off-policy + replay buffer | Off-policy + replay + PER | On-policy (rollouts) |
| **Stability mechanism** | Soft target networks (Polyak τ) | Hard target networks (every C steps) + double Q | Trust region via clipping |
| **Sample budget for similar tasks** | ~10⁵–10⁶ steps | ~10⁶ steps (Atari) | ~10⁶–10⁷ steps |
| **Hyperparameter sensitivity** | Moderate | Low | High (clip ε, λ, etc.) |
| **Continuous control quality** | Best-in-class for low-dim | N/A (needs discretisation) | Comparable but more sample-hungry |

## Why each alternative fails for *this* task

### DQN

| Issue | Detail |
|---|---|
| Action space mismatch | (v, ω) ∈ [−1, 1]² is continuous; DQN needs a finite action set |
| Discretisation explosion | At 100 bins/axis → 10 000 discrete actions, each needing its own Q-output. L09 § 3 calls this "combinatorial computation explosion" |
| Loss of fine motor control | A vacuum at "0.5 m/s exactly" cannot be expressed as "approximately bin 50 of 100" without compounding error over a 500-step episode |
| **Verdict** | Architecturally wrong for continuous physics |

### PPO

| Issue | Detail |
|---|---|
| On-policy data | After each gradient step, all collected rollouts become stale; replay buffer not used |
| Sample cost | ~3–10× the env steps of DDPG for the same final reward in continuous control (per Schulman 2017 + community benchmarks) |
| Stochastic actor wastes variance budget | The vacuum's physics is deterministic; a stochastic π(a\|s) adds entropy that has to be re-learned |
| Slower convergence on low-dim continuous | DDPG dominates the slide-19 single-task continuous benchmarks |
| **Verdict** | Works correctly but slower; over-engineered for a deterministic physical engine |

### Why DDPG wins

1. **Action space match**: tanh-bounded actor outputs the action directly — no discretisation, no enumeration.
2. **Deterministic actor = deterministic physics**: The vacuum's motor takes (v, ω); the actor outputs (v, ω). No symbolic distance between policy output and actuator input.
3. **Replay buffer reuses data**: With ~12 minutes of simulator time per cell (4 000 LIDAR-heavy steps), throwing away rollouts after one update is unaffordable. The buffer lets us amortise the LIDAR cost.
4. **Soft target nets stabilise the bootstrap**: Critic learns against a stationary y for ~200 steps before the target catches up — the deadly-triad shield (slide 6).

## What if the action space had been discrete?

If the vacuum had been "press one of 8 buttons" (turn left 45°, turn right 45°,
forward, backward, …), DQN would be the obvious choice and DDPG would be
inappropriate (the actor's continuous output adds nothing). The spec is
clear that the action space is continuous, and slide 3 explicitly walks
through why this is the deciding factor.

## What if the env had been longer-horizon?

For 1 000-episode horizons with sparse reward (e.g., "find a specific
object"), PPO's trust-region trick prevents catastrophic policy collapse
that DDPG can suffer from. DDPG works because our cleaning reward is
dense (we even added `coverage_progress_coef` for additional density in
Layer 18) and our horizon is short (500 steps).

## Where DDPG is known to fail and how the literature responds

| Known DDPG weakness | Modern fix |
|---|---|
| Over-estimation bias in single critic | **TD3** (Fujimoto 2018): twin critics + min for target. Implemented in this project as an opt-in module — see [`model/td3_network.py`](../src/roomba_lab/model/td3_network.py) + [`services/td3_update.py`](../src/roomba_lab/services/td3_update.py) (Layer 20). |
| Deterministic actor → no exploration | **SAC** (Haarnoja 2018): stochastic actor with entropy bonus. Not implemented; documented as extension point. |
| Hard sigma decay can be brittle | Parameter-space noise (Plappert 2018). Out of scope. |

## Citations

1. **DDPG**: Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D. (2016). *Continuous control with deep reinforcement learning*. ICLR.
2. **DPG theorem**: Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., Riedmiller, M. (2014). *Deterministic policy gradient algorithms*. ICML.
3. **DQN**: Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518.
4. **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv:1707.06347.
5. **TD3**: Fujimoto, S., van Hoof, H., Meger, D. (2018). *Addressing function approximation error in actor-critic methods*. ICML.
6. **SAC**: Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). *Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor*. ICML.
