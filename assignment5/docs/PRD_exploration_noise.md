# PRD — Exploration noise (Gaussian + OU + decay)

> Mechanism PRD for slide 7 of L09 and Requirement 4 of EX05.

## 1. Why exploration noise at all?

DDPG's actor is **deterministic** — `μ(s)` returns a single action for every state. If we executed `μ(s)` as-is during data collection, the replay buffer would only ever see one action per state. No data diversity → critic cannot learn the gradient of Q w.r.t. *other* actions → actor never moves off its initial behaviour.

Slide 7's solution: add an external noise process to the actor output **at action time** (NOT at training time):

```
a_t = clip( μ(s_t) + N_t, −1, +1 )
```

## 2. Two noise families

### 2.1 Gaussian (the default — ADR-005)

```
N_t ~ Normal(0, σ²·I)
```

Independent across time. Simple, well-understood, no internal state.

Default σ schedule: linear decay from `0.2` (initial) to `0.05` (final) over the first 50 000 steps. After that, σ is clamped at 0.05 — exploration tapers but does not vanish.

> *Reflection-Q2: "What is the initial variance you chose?" → **σ_initial = 0.2**, because the action space is `[−1, 1]` and σ = 0.2 means the typical noise magnitude (1σ) equals 10 % of the full action range. Larger σ produces too many wall collisions early; smaller σ stalls the actor near init.*

### 2.2 Ornstein-Uhlenbeck (Lillicrap 2016 original)

```
N_{t+1} = N_t + θ (μ − N_t) Δt + σ √Δt · ξ,    ξ ~ Normal(0, I)
```

Produces **temporally-correlated** noise — useful when actions are tightly coupled to angular velocity (steering noise spikes that immediately reverse are unrealistic). Lillicrap noted OU helped in mountain-car-style envs.

We implement OU for completeness and pedagogical comparison, but default to Gaussian (TD3, SAC, and most production implementations agree OU is unnecessary).

## 3. Schedule

```python
# noise/schedule.py
class LinearSigmaSchedule:
    def __init__(self, initial: float, final: float, decay_steps: int) -> None:
        ...
    def at(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.decay_steps))
        return self.initial + (self.final - self.initial) * frac
```

The schedule guarantees:

- `at(0) == initial`
- `at(decay_steps) == final`
- `at(any step > decay_steps) == final`

## 4. Reflection answer for the spec (Q2)

> *"What would happen if you stopped adding Gaussian exploration noise from the actor output entirely in the early training stages? Explain how this affects the coverage map."*

Without exploration noise:

1. The actor is initialised randomly (orthogonal-init around a small action magnitude). Its initial deterministic policy is some near-arbitrary fixed direction.
2. Every episode the robot follows the same trajectory from any given spawn (because the env is deterministic, the actor is deterministic, the LIDAR is deterministic).
3. The replay buffer fills with copies of the same trajectory.
4. The critic learns Q(s, a) only on this one narrow ridge of (s, a) pairs.
5. The actor gradient `∇μ Q(s, μ(s))` has no signal in directions the buffer never explored.
6. **Coverage flatlines near the spawn radius** — the agent never discovers the rest of the apartment because it never tries any other action.

**Empirical evidence**: the noise-σ sweep at σ=0.0 (Layer 11) should show final coverage near `2π · spawn_radius² / apartment_area` and zero improvement over training. We will plot this on top of the σ=0.2 baseline to make the contrast visceral.

## 5. Test plan

In `tests/unit/test_noise.py`:

| Test | Setup | Assert |
|---|---|---|
| Gaussian large-N mean ≈ 0 | sample 10 000 noise vectors | |mean| < 3σ / √N |
| Gaussian large-N variance ≈ σ² | sample 10 000 noise vectors | std within 5 % of σ |
| OU lag-1 autocorrelation > 0.8 | θ=0.15, σ=0.2, length 1000 | corr(N_t, N_{t+1}) > 0.8 |
| OU mean reverts to μ | μ=0.5, run 5000 steps | |mean − 0.5| < 0.05 |
| Schedule clamps at decay_steps | initial=0.2, final=0.05, decay=1000 | at(1001) == 0.05 |

## 6. Acceptance criteria

1. `noise/gaussian.py` and `noise/ou.py` are both implemented and selectable via config.
2. The schedule is implemented and tested.
3. The training service uses `schedule.at(step)` to set σ on every step.
4. The σ=0.0 ablation cell exists in `results/sweeps/noise_sigma.json` with 3-seed CIs.
5. The reflection-Q2 answer in README is grounded in this ablation.

## 7. Non-goals

- Parameter-space noise (NoisyNets, Plappert et al. 2018)
- Action-conditioned noise schedules
- Per-action-dim heteroskedastic noise (each action gets the same σ in the default config)
