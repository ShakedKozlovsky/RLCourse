# PRD — MuJoCo Continuous-Control Environment Wrapper

## Theory recap

MuJoCo (Multi-Joint dynamics with Contact) is the physics simulator that defines the canonical continuous-control benchmarks for RL: HalfCheetah, Walker2d, Hopper, Ant, Humanoid, etc. Since 2022 MuJoCo is open-source and shipped via `pip install mujoco`. Gymnasium provides the standard `gym.Env` wrappers (`HalfCheetah-v4`, `Walker2d-v4`, etc.).

## Why these two environments?

| Environment | Observation dim | Action dim | Termination | Why chosen |
|---|---|---|---|---|
| **HalfCheetah-v4** | 17 | 6 | timeout only | Slide 19's primary PPO benchmark; smooth dense reward; never terminates → tests the bootstrapping path in GAE |
| **Walker2d-v4** | 17 | 6 | torso height + angle bounds | Bipedal walking; episodes terminate on falling → tests the `done` handling in GAE recursion |

The two together span:
- **Quadruped vs biped** — different morphologies.
- **Always-on vs terminating** — exercises both branches of `next_non_terminal` in GAE.
- **Same dims** — fair comparison; one network architecture works for both.

Hopper-v4 is omitted because it's structurally identical to Walker2d (planar bipedal). HumanoidStandup-v4 would test PPO at scale but takes 10× the compute.

## Observation normalisation

MuJoCo observations have heterogeneous scales (joint angles in radians vs velocities in rad/s vs forces in N·m). PPO traditionally wraps the env with a **running-mean / running-std normaliser**:

```python
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0001

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

def normalise(obs: np.ndarray) -> np.ndarray:
    return (obs - rms.mean) / np.sqrt(rms.var + 1e-8)
```

The normaliser is updated at training time and **frozen at evaluation time** (otherwise the eval observation distribution drifts during evaluation, biasing the rollout).

## Vectorised environments

PPO collects `steps_per_rollout` transitions per iteration. With a single env, that's 2048 sequential `env.step()` calls. With `gymnasium.vector.SyncVectorEnv(4)`, it's 512 calls × 4 envs in parallel → same total transitions, ~4× wall-clock speedup.

```python
# environment/vector_env.py
def make_vector_env(env_id: str, n: int, seed: int) -> SyncVectorEnv:
    def _make(i):
        def _thunk():
            env = gym.make(env_id)
            env.action_space.seed(seed + i)
            return env
        return _thunk
    return SyncVectorEnv([_make(i) for i in range(n)])
```

## Action space handling

Both envs have `Box(-1, 1, (6,))` action spaces (after Gymnasium's automatic normalisation). The Gaussian actor outputs unbounded `μ`, so we rely on the env's automatic clipping. Alternative: `tanh` squashing inside the actor; standard PPO doesn't do this for MuJoCo.

## Reward shaping (intentionally minimal)

Both envs ship dense rewards from MuJoCo: forward velocity − control cost. We do **not** modify the reward. This is a key contrast with Assignment 3, where the reward function was hand-designed; here we want pure algorithm comparison on the canonical reward.

## Acceptance criteria

- `test_mujoco_env.py::test_halfcheetah_step_shapes` — `step(action)` returns obs `(17,)`, reward `float`, done `bool`.
- `test_mujoco_env.py::test_walker2d_step_shapes` — same.
- `test_vector_env.py::test_batched_obs_shape` — `step(actions)` with `n=4` returns obs `(4, 17)`.
- `test_obs_norm.py::test_running_stats_consistent` — RMS update equals a NumPy reference (Welford's algorithm).
- `test_obs_norm.py::test_eval_norm_frozen` — at eval time the normaliser does not update.

## Where this lives

- `src/proximal_lab/environment/mujoco_env.py` — `make_env(env_id, seed)` + `RunningMeanStd` ≤ 80 LOC.
- `src/proximal_lab/environment/vector_env.py` — `make_vector_env(env_id, n, seed)` ≤ 40 LOC.

## Caveats

- MuJoCo determinism is not bit-perfect across hardware. Same seed → same trajectories on the same machine, but different machines may diverge after thousands of steps. Reproducibility tests run on a single machine.
- Walker2d terminates early in many episodes during training — the rollout buffer has to handle variable-length episodes via `done` flags. This is why GAE's terminal handling matters.
- Action clipping at the env boundary means the policy can output values outside `[−1, 1]` without penalty, which can encourage extreme `μ` values. Diagnostic: log `|action|` mean per iteration.

## Sources

- E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A physics engine for model-based control," IROS, 2012.
- M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," 2024.
- Schulman 2017 PPO paper § 6 (Experiments on MuJoCo).
- L08 lecture slide 19.
