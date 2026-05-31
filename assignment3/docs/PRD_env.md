# PRD — WorldEnv (Gymnasium-style)

## Responsibilities

- Maintain the agent's current state `s_t` and the LSTM's hidden state `h_t`.
- Step through fixed-length episodes (28 days by default).
- Translate `action` → next state via the trained LSTM world model.
- Compute reward via `RewardFunction`.
- Track per-step diagnostics (current muscle, recent rest count, total volume) for both reward and (optional) action masking.

## API (Gymnasium-compatible)

```python
class WorldEnv:
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    @property
    def action_space(self) -> spaces.Discrete: ...   # = Discrete(5)
    @property
    def observation_space(self) -> spaces.Box: ...    # = Box(shape=(16,))
```

`info` dict carries: `step_idx`, `last_action`, `recent_volume_7d`, `muscle_distribution`, `action_was_masked` (bool, when masking enabled).

## State assembly at step t

The state vector `s_t` is the 16-dim feature produced by `FeatureEngineer` (PLAN.md §5). On `reset()`, `s_0` is the first day of the synthetic trajectory; on `step(a)`, the LSTM produces `s_{t+1}` from `(s_t, a)` and the maintained hidden state `h_t`.

## Episode lifecycle

```
reset() → s_0, h_0=zeros
   for t in 0..T-1:
     action_t = policy(s_t)
     reward_t = RewardFunction.compute(s_t, action_t, history)
     s_{t+1}, h_{t+1} = LSTMWorldModel.step(s_t, action_t, h_t)
     yield (s_t, action_t, reward_t, s_{t+1})
   done = True at t = T-1
```

## Termination

`done = True` when `step_idx == episode_length - 1`. `truncated` is `False` for now (we use full fixed-length episodes); reserved for future variable-length episodes.

## Why this is not gym.make-registered

We instantiate `WorldEnv` directly rather than via `gym.envs`. Keeps the world-model dependency explicit and the environment trivially mockable in tests. The training services take a `WorldEnv` instance, not a `gym.Env` string.

## Acceptance criteria

- `test_world_env.py::test_reset_shape` — `(16,)` state, `info == {}` initially.
- `test_world_env.py::test_step_advances_one` — `info["step_idx"]` increments by 1.
- `test_world_env.py::test_done_at_episode_length` — `done` becomes True at exactly `episode_length - 1`.
- `test_world_env.py::test_deterministic_with_seed` — same seed → same trajectory.
- `test_world_env.py::test_action_masking_when_enabled` — when consecutive same-group limit is hit, the chosen action is masked and not executed.

## Why this isn't `gym.Env` exactly

Following the Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)` but exposing the `np.ndarray` directly. Training services don't call `gym.make`; this keeps the env trivially injectable for tests and replaceable for future world-model improvements.
