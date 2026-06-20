# L09 slide-by-slide → code line map

> For every concept in the L09 lecture, point at the exact file + line where it
> appears in `roomba-lab`. The V3 § 2.7 traceability requirement, taken to its
> strict interpretation. Citations use the format `file.py:Nxxx-Nyyy`.

## § 1 — Introduction (autonomous AI journey)

Lecture framing. No direct code analogue; the L09 introduction motivates the
whole project. See [`docs/PRD.md`](PRD.md) § 1.

## § 2 — Algorithm evolution table (Q-Learning → DQN → REINFORCE → A2C / PPO → DDPG)

The table itself becomes the **DDPG vs prior-algorithms comparison** in this
project's README § "DDPG vs DQN vs PPO". The takeaway — DDPG is the only entry
with continuous-only action space + deterministic actor + Q-foundation — is
quoted in [`docs/PRD.md`](PRD.md) § 1 + [`docs/PRD_ddpg.md`](PRD_ddpg.md) § 1.

## § 3 — Discretisation explosion

> *"For a 7-DoF arm with 100 bins per joint, 10¹⁴ discrete actions is
> impossible to enumerate in real time."*

Mapped to our concrete 2-D vacuum: 100 bins × 100 bins = 10 000 actions —
already worse than directly outputting the (v, ω) continuous vector. Discussed
in [`docs/PRD_ddpg.md`](PRD_ddpg.md) § 2.

## § 4 — DPG theorem (Silver 2014, equation 1)

$$\nabla_{\theta}\, J(\mu_{\theta}) = \mathbb{E}_{s\sim\rho^{\mu}}\!\left[\, \nabla_{\theta}\mu_{\theta}(s)\, \nabla_{a} Q^{\mu}(s, a)\big|_{a=\mu_{\theta}(s)} \,\right]$$

Implemented in [`services/ddpg_update.py:55-59`](../src/roomba_lab/services/ddpg_update.py):

```python
def actor_loss(net: ActorCriticNet, batch: dict, ...) -> torch.Tensor:
    ...
    return -net.critic(b["state"], net.actor(b["state"])).mean()
```

The minus sign and the chain rule via `net.actor()` inside `net.critic()` make
this the deterministic policy gradient — autograd handles ∇θμ · ∇aQ
automatically.

## § 5 — Architecture: μ(s|θ_μ) tanh-bounded, Q(s,a|θ_Q) state-action concat

| Concept | File |
|---|---|
| Actor: MLP → tanh → action ∈ [-1,1]^d | [`model/actor.py:32`](../src/roomba_lab/model/actor.py) |
| Critic: concat(state, action) → MLP → scalar Q | [`model/critic.py:30-32`](../src/roomba_lab/model/critic.py) |
| Wrapped + targets | [`model/actor_critic_network.py:21-30`](../src/roomba_lab/model/actor_critic_network.py) |

## § 6 — Soft target updates θ' ← τ·θ + (1-τ)·θ'

[`model/soft_update.py:17-25`](../src/roomba_lab/model/soft_update.py):

```python
def polyak_update(target_params, source_params, tau):
    if not 0.0 <= tau <= 1.0:
        raise ValueError(...)
    with torch.no_grad():
        for t, s in zip(target_params, source_params, strict=True):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)
```

The 4-test math battery in [`tests/unit/test_soft_update.py`](../tests/unit/test_soft_update.py)
proves τ=0 freezes the target, τ=1 hard-copies, τ=0.5 is exact midpoint, and
repeated calls converge target → source.

## § 7 — Exploration noise (Gaussian and Ornstein-Uhlenbeck)

| Family | File |
|---|---|
| Gaussian (default, ADR-005) | [`noise/gaussian.py:28-30`](../src/roomba_lab/noise/gaussian.py) |
| Ornstein-Uhlenbeck (Lillicrap 2016 original) | [`noise/ou.py:31-38`](../src/roomba_lab/noise/ou.py) |
| Linear σ schedule | [`noise/schedule.py:19-22`](../src/roomba_lab/noise/schedule.py) |

Action-time injection happens at
[`services/ddpg_service.py:58-62`](../src/roomba_lab/services/ddpg_service.py):

```python
action = self.net.actor(obs_t).cpu().numpy()[0]
action += self.noise.sample()
return np.clip(action, -1.0, 1.0).astype(np.float32)
```

## § 8 — Training pipeline (full step ordering)

The slide-8 pseudocode → [`services/ddpg_service.py:70-99`](../src/roomba_lab/services/ddpg_service.py):

| Step | Code |
|---|---|
| Observe / select action | line 77 — `action = self._select_action(obs, step)` |
| Env step | line 78 — `next_obs, reward, done, info = self.env.step(action)` |
| Push transition | line 81 — `self.buffer.push(Transition(...))` |
| Warmup gate | line 85 — `if len(self.buffer) >= max(batch_size, warmup_steps):` |
| Sample batch | line 86 — `batch = self.buffer.sample(self.hp.batch_size)` |
| Apply update (critic + actor + Polyak) | line 87 — `apply_update(self.net, batch, ...)` |
| σ schedule | line 76 — `self.noise.set_sigma(self.schedule.at(step))` |
| Episode reset | line 102 — `obs = self.env.reset(seed=seed + step)` |

## § 9 — Glossary

The lecture's glossary closes the algorithmic story. No direct code; concepts
mapped via the docstrings in each module.

## § 10 — Practical task: HouseExpo cleaning robot

The entire project. Mapping:

| Spec item | File |
|---|---|
| "no Gymnasium / no Gazebo — custom 2-D sim" | [`environment/roomba_env.py`](../src/roomba_lab/environment/roomba_env.py) imports zero gym packages — `grep "import gym" src/` returns nothing |
| HouseExpo JSON data | [`data/houseexpo_loader.py`](../src/roomba_lab/data/houseexpo_loader.py) + [`data/raw/sample_maps/`](../data/raw/sample_maps/) |
| 2-dim continuous action ∈ [-1, 1]² | [`environment/roomba_env.py:48-53`](../src/roomba_lab/environment/roomba_env.py) `action_dim = 2` |
| LIDAR-style virtual sensors | [`sensor/lidar.py`](../src/roomba_lab/sensor/lidar.py) |
| Reward: +cleaning / -collision | [`environment/reward.py`](../src/roomba_lab/environment/reward.py) |
| Polyak soft-update lines (spec § Item 2) | [`model/soft_update.py:17-25`](../src/roomba_lab/model/soft_update.py) |
| Hyperparameter justification (spec § Item 3) | [`README.md`](../README.md) hyperparameter table |
| Initial Gaussian variance (spec § Item 4) | `configs/setup.json::noise.sigma_initial = 0.2` |
| Learning curve (spec § graphs a) | [`assets/plots/learning_curve_tuned.png`](../assets/plots/learning_curve_tuned.png) |
| Critic loss curve (spec § graphs b) | [`assets/plots/critic_loss_tuned.png`](../assets/plots/critic_loss_tuned.png) |
| Trajectory overlay on map | [`assets/plots/trajectory_overlay_tuned.png`](../assets/plots/trajectory_overlay_tuned.png) |
| Animation | [`assets/gifs/cleaning_episode.gif`](../assets/gifs/cleaning_episode.gif) |

## § 11 — Bibliography

| Paper | Usage in code |
|---|---|
| Silver et al. 2014, DPG theorem | [`services/ddpg_update.py::actor_loss`](../src/roomba_lab/services/ddpg_update.py) docstring cites it |
| Lillicrap et al. 2016, DDPG | All of `model/` + the τ default + actor/critic LR defaults |
| Li et al. 2019, HouseExpo | [`data/houseexpo_loader.py`](../src/roomba_lab/data/houseexpo_loader.py) module docstring |
