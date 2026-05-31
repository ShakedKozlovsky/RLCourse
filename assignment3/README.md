# Assignment 3 — REINFORCE + A2C Fitness Recommender

> **Course:** Reinforcement Learning with Deep Learning · **Author:** Shaked Kozlovsky (ID 208904839)
>
> A complete, object-oriented policy-gradient RL system that learns a daily **workout-recommendation policy** (`PUSH / PULL / LEGS / CARDIO / REST`) over an **LSTM-learned world model** built from the Kaggle 600 K+ Fitness Exercise & Workout Program dataset. Two algorithms compared: **REINFORCE** (Williams 1992) and **A2C** (Mnih et al. 2016).
>
> This is a *learning* project about Reinforcement Learning, **not** a fitness coach or medical recommendation. See [§ Honest acknowledgements](#13-honest-acknowledgements).

## Table of contents

1. [Project goal & RL framing](#1-project-goal--rl-framing)
2. [Lecture-slide mapping](#2-lecture-slide-mapping)
3. [Data pipeline (Part B)](#3-data-pipeline-part-b)
4. [State, action, reward, env (Part A + B)](#4-state-action-reward-env-part-a--b)
5. [LSTM world model (Part C)](#5-lstm-world-model-part-c)
6. [REINFORCE (Part D)](#6-reinforce-part-d)
7. [A2C (Part E)](#7-a2c-part-e)
8. [Comparison & experiments](#8-comparison--experiments)
9. [GUI / CLI / SDK](#9-gui--cli--sdk)
10. [Quality bar — tests, ruff, coverage](#10-quality-bar--tests-ruff-coverage)
11. [Five reflection answers (Part F)](#11-five-reflection-answers-part-f)
12. [Sources](#12-sources)
13. [Honest acknowledgements](#13-honest-acknowledgements)

## 1. Project goal & RL framing

Demonstrate the full policy-gradient learning chain:

1. **Formulate** workout recommendation as an MDP.
2. **Build** a synthetic per-day trajectory from a chosen Kaggle program.
3. **Learn** an LSTM transition model `s_{t+1} ≈ f_φ(s_t, a_t, h_t)`.
4. **Train REINFORCE** over the learned world model.
5. **Train A2C** over the same world model.
6. **Compare** the two: stability, action diversity, reward variance.

The grade rewards understanding of `REINFORCE → +baseline → +advantage → Actor-Critic`, the role of the baseline/advantage in variance reduction, and honest analysis. A clean "policy beats random" outcome is *not* the point; controlled experiments + algorithmic comparison are.

## 2. Lecture-slide mapping

| Lecture concept | Slide | Where it lives in this repo |
|---|---|---|
| Policy-gradient flow `π → a → r → ∇` | 1 | The whole pipeline. |
| `J(θ) = E_τ[R(τ)]` | 4 | `services/reinforce_service.py` |
| Estimator `∇ log π · R` | 5 | Same. |
| Baseline subtraction `(R − b)` | 6 | `ReinforceService(use_baseline=True)` (mean baseline). |
| Reward-to-go `G_t = Σ γ^{k−t} r_k` | 7 | `ReinforceService.reward_to_go` (reverse pass). |
| Advantage `A_t ≈ δ_t` | 7, 15 | `services/a2c_service.py::td_error` |
| Actor + critic with two heads | 9 | `model/actor_critic_network.py` |
| TD-error / actor / critic updates | 10 | `A2CService._run_episode` |
| A2C stability vs REINFORCE | 21 | [§ Comparison & experiments](#8-comparison--experiments) |

## 3. Data pipeline (Part B)

The dataset has **2 598 programs** and **605 033 detailed exercise rows**. There is **no `target_muscle` column** — only an `exercise_name` string. We added a keyword-based [`MuscleClassifier`](src/fitness_rl/data/muscle_classifier.py) with word-boundary regex to infer muscle group; documented as a heuristic in [`docs/PRD_data.md`](docs/PRD_data.md).

| Step | Module |
|---|---|
| Load CSVs | [`data/kaggle_loader.py`](src/fitness_rl/data/kaggle_loader.py) |
| Clean negative reps (≤ 600 → seconds, else NaN) | [`data/preprocessor.py`](src/fitness_rl/data/preprocessor.py) |
| Pick one program by 4 criteria | [`data/program_selector.py`](src/fitness_rl/data/program_selector.py) |
| Build per-day trajectory + rest-day insertion | [`data/trajectory_builder.py`](src/fitness_rl/data/trajectory_builder.py) |
| 16-dim state vector | [`data/feature_engineer.py`](src/fitness_rl/data/feature_engineer.py) |
| Orchestrate end-to-end | [`services/data_service.py`](src/fitness_rl/services/data_service.py) |

On the real Kaggle data the pipeline picks **"(MASS MONSTER) High Intensity 4 Day Upper Lower Program"** (12 weeks, 90 min/session), yielding an **84-day trajectory** with action distribution `PUSH=23, PULL=3, LEGS=22, REST=36`.

## 4. State, action, reward, env (Part A + B)

| Component | Definition | Where |
|---|---|---|
| State `s_t` (16-d) | `[volume_norm(1), muscle_dist(5), session_dur(1), week(1), day_one_hot(7), rest_indicator(1)]` | [`data/feature_engineer.py`](src/fitness_rl/data/feature_engineer.py) |
| Action `a_t` | 5 discrete: `PUSH=0, PULL=1, LEGS=2, CARDIO=3, REST=4` | [`shared/types.py`](src/fitness_rl/shared/types.py) |
| Reward `r_t` | `gain_t − λ_1·overload_t − λ_2·imbalance_t` (rolling 7-day window for overload; entropy-based imbalance, zeroed on rest days) | [`environment/reward.py`](src/fitness_rl/environment/reward.py) |
| Transition | LSTM `f_φ(s_t, a_t, h_t)` when trained; identity stub for unit tests | [`environment/world_env.py`](src/fitness_rl/environment/world_env.py) |
| Action mask | Pre-softmax `-inf` for triples of same-group or rest | [`environment/action_mask.py`](src/fitness_rl/environment/action_mask.py) |
| Episode | 28 steps (one month); γ = 0.99 | `configs/setup.json:env` |

## 5. LSTM world model (Part C)

```
input (B, W=7, 21) ← state(16) + action_one_hot(5)
   │
LSTM(21 → 64, 1 layer, batch_first)
   │
take last hidden state
   │
Linear(64 → 16) → predicted s_{t+W}
```

Trained supervised with **MSE + Adam(lr=1e-3) + early stopping (patience 10)** over rolling 7-day windows of the 84-day trajectory, chronological 80/20 train/val split (no leakage). The trained model exposes `as_transition_fn(window_size, warmup_state)` returning a stateful `(state, action) → next_state` closure that plugs straight into `WorldEnv` — see [`model/lstm_world_model.py`](src/fitness_rl/model/lstm_world_model.py).

**Real-data result (30 epochs):** best val MSE **0.0560** at epoch 29.

![LSTM world-model loss](assets/plots/world_model_loss.png)

## 6. REINFORCE (Part D)

Williams (1992). Each episode: sample a rollout, compute reward-to-go `G_t = r_t + γ·G_{t+1}` in reverse, subtract the **mean baseline** (`b = mean(G_t)`), and apply `∇_θ log π(a|s) · (G − b)` via Adam. Optional **entropy bonus** + optional **action masking** at the logits level — see [`services/reinforce_service.py`](src/fitness_rl/services/reinforce_service.py).

The policy is a 2-layer MLP `16 → 128 → 128 → 5 logits` — the same actor architecture as A2C, so the comparison isolates the *learning rule* from network capacity.

![REINFORCE reward curve](assets/plots/reinforce_reward.png)

**Real-data result (60 episodes over the trained LSTM):** final-episode reward **6.36**, final-30 % mean **6.17**, final-30 % CV **0.024**.

## 7. A2C (Part E)

Mnih et al. (2016). Per transition `(s, a, r, s')`:
```
δ = r + γ·V(s') − V(s)        # TD error = advantage
actor_loss  = −log π(a|s) · δ.detach() − β · H(π)
critic_loss = ½ δ²
```
The shared trunk + actor head + critic head live in [`model/actor_critic_network.py`](src/fitness_rl/model/actor_critic_network.py). The **trunk is stepped exactly once per update at `actor_lr`** — the critic optimizer only owns the critic head — to avoid the double-step that the naive "trunk in both optimizers" pattern produces. A unit test asserts this partition. Implementation: [`services/a2c_service.py`](src/fitness_rl/services/a2c_service.py).

![A2C reward curve](assets/plots/a2c_reward.png)

**Real-data result (60 episodes over the trained LSTM):** final-episode reward **6.91**, final-30 % mean **7.06**, final-30 % CV **0.011**.

## 8. Comparison & experiments

### REINFORCE vs A2C (60 episodes each)

![REINFORCE vs A2C](assets/plots/comparison.png)

| Metric | REINFORCE | A2C | Winner |
|---|---|---|---|
| Final-30 % mean reward | **6.17** | **7.06** | A2C |
| Final-30 % CV (std / |mean|) | 0.024 | **0.011** | A2C — ~2× less variable |
| Overall std over the whole run | **0.17** | 0.26 | REINFORCE (mid-training A2C has more swing) |

**The theory holds:** A2C's per-step TD error is a lower-variance update than REINFORCE's end-of-episode `G_t`. By the end of training A2C is **both higher and more stable** — exactly the prediction of slide § 21.

### Action distribution per algorithm

![Action distribution](assets/plots/action_distribution.png)

REINFORCE produces a much more *diverse* action mix (≈ 16 / 23 / 25 / 14 / 22 across PUSH / PULL / LEGS / CARDIO / REST). A2C concentrates **57 % on REST** — under our reward (`gain − overload − imbalance`), REST has lowest overload penalty and zero imbalance penalty (rest days bypass imbalance), so a critic that learns this fast collapses the actor toward REST despite the entropy bonus. This is a real, expected failure mode of A2C when the critic converges faster than the actor; the assignment's slide § 22 mentions exactly this hazard.

### Three differentiator experiments — [`results/experiments/`](results/experiments/)

`uv run fitness-rl experiments --episodes 20 --out-dir results/experiments` writes three JSON files:

1. **Action-masking ablation** ({REINFORCE, A2C} × {mask on, mask off}): on the identity-transition env the rewards are deterministic per state so the absolute reward is the same across cells, but action distributions differ in the expected direction (mask-on never produces 3-in-a-row).
2. **Reward-weight sweep** (3 × 3 grid of `λ_overload × λ_imbalance`): **monotone**, mean reward drops from **+6.84** at `(0, 0)` to **−6.47** at `(0.5, 0.6)`. The reward function is well-formed: heavier penalties hurt achievable reward as expected.
3. **Collapse analysis**: with only 20 episodes on the *identity*-transition env (no LSTM), both algos collapse to a single action (REINFORCE → CARDIO, A2C → PUSH). With the trained LSTM at 60 episodes (the [§ 8 comparison run](#reinforce-vs-a2c-60-episodes-each)) A2C still concentrates on REST but stays multi-modal; REINFORCE stays balanced.

## 9. GUI / CLI / SDK

### Five-tab PyQt6 GUI — [`interface/gui/`](src/fitness_rl/interface/gui/)

| Tab | Screenshot |
|---|---|
| Data | ![Data tab](assets/gui/tab_0_data.png) |
| World model | ![World model tab](assets/gui/tab_1_world_model.png) |
| REINFORCE | ![REINFORCE tab](assets/gui/tab_2_reinforce.png) |
| A2C | ![A2C tab](assets/gui/tab_3_a2c.png) |
| Compare | ![Compare tab](assets/gui/tab_4_compare.png) |

Each tab runs the long-running SDK call on a `QThread` worker (see [`interface/gui/worker.py`](src/fitness_rl/interface/gui/worker.py)) so the UI stays responsive. All tabs share a single `FitnessRL` SDK instance — data loaded in the Data tab is visible from the REINFORCE tab without reloading.

### Click CLI — [`interface/cli/main.py`](src/fitness_rl/interface/cli/main.py)

```bash
uv run fitness-rl --help
uv run fitness-rl prepare-data
uv run fitness-rl train-world
uv run fitness-rl train-reinforce --episodes 60
uv run fitness-rl train-a2c --episodes 60
uv run fitness-rl compare --episodes 60 --out results/compare.json
uv run fitness-rl predict --algo a2c --episodes 10
uv run fitness-rl experiments --episodes 20 --out-dir results/experiments
uv run fitness-rl gui     # launches the PyQt6 GUI
uv run fitness-rl menu    # interactive numeric menu
```

### FitnessRL SDK — [`sdk/sdk.py`](src/fitness_rl/sdk/sdk.py)

```python
from fitness_rl.sdk.sdk import FitnessRL

sdk = FitnessRL(config_path="configs/setup.json")
sdk.prepare_data()
sdk.train_world_model()
hist_r = sdk.train_reinforce(episodes=60)
hist_a = sdk.train_a2c(episodes=60)
report = sdk.compare()
action = sdk.predict(state, algo="a2c")
```

## 10. Quality bar — tests, ruff, coverage

- **183 tests** (unit + integration + headless-Qt GUI smoke), all green.
- **96.63 % branch coverage** (gate is 85 %).
- `ruff check src/ tests/` returns 0.
- **Every source file is ≤ 150 LOC.**
- No magic numbers in source — everything lives in [`configs/setup.json`](configs/setup.json).
- Every commit is `Layer N: <summary>` + bullet body — the [git log](https://github.com/ShakedKozlovsky/RLCourse/commits/main) reads as a build story planning → data → environment → world model → REINFORCE → A2C → eval → SDK → GUI → experiments → README.

Reproduce the full quality gate:
```bash
uv run pytest tests/ -q --cov=src/fitness_rl --cov-report=term
uv run ruff check src/ tests/
```

## 11. Five reflection answers (Part F)

### F.1 — How realistic is the synthetic trainee?

The trainee is **structurally** realistic and **physiologically** not. The 84-day trajectory mirrors a real Kaggle hypertrophy program's *muscle-group cadence* (week-of-cycle, day-of-week, per-day volume). It does **not** model fatigue, recovery, injury risk, sleep, or progressive overload — those are physiological outcomes the dataset doesn't contain. So the policy learns to navigate a *program-shaped state space*, which is a meaningful exercise in policy gradient theory but not a medical recommendation. The reward function operationalises three constructs (gain, overload, imbalance) that *correlate* with good training but were chosen by the author, not validated against any biological signal.

### F.2 — Why an LSTM world model instead of model-free RL?

Two reasons. **Sample efficiency:** an 84-day trajectory is far too small to train a deep policy gradient method directly — we'd see almost no signal per epoch. By learning the dynamics first (supervised, dense per-step signal) and then training the policy *inside* the model (effectively unlimited rollouts), we sidestep the sample-budget problem. **Pedagogical:** it's the cleanest illustration of *what a world model actually does*. The LSTM is exactly Ha & Schmidhuber's idea — a recurrent dynamics model with a hidden state encoding history — bolted to a policy gradient learner. A feed-forward state-to-state regressor would be the simpler baseline; the LSTM's win is the hidden-state continuity across the rollout.

### F.3 — When did REINFORCE beat A2C and when didn't it?

REINFORCE's only structural advantage is **lower overall variance during training** (its update is a sum over an episode that smooths step-to-step noise). After 60 episodes it has **higher overall std** than A2C (0.17 vs 0.26) but **higher final-30 % CV** (0.024 vs 0.011) — i.e. mid-training A2C is jittery (the critic is still learning), but by the tail A2C settles and REINFORCE is the noisier one. A2C also wins on **action diversity** away from collapse early in training; conversely it concentrates 57 % on REST by the end. The takeaway matches the lecture: A2C is the better long-run learner but is more sensitive to actor / critic balance and reward shape.

### F.4 — What goes wrong when the critic learns too fast?

Three failure modes show up in our runs:

1. **Actor collapse** — when the critic accurately predicts the (deterministic-per-state) reward early, `δ ≈ 0` for "average" actions, so the actor only sees signal on outliers. With low entropy bonus + a reward where REST is locally low-risk, the policy concentrates on REST.
2. **Trunk double-step** — the naive "put the trunk in both optimisers" pattern updates the trunk with `actor_lr + critic_lr` per step. Our [`actor_critic_network.py::critic_params`](src/fitness_rl/model/actor_critic_network.py) excludes the trunk from the critic optimiser; a unit test asserts the partition.
3. **Stale-target chase** — the actor optimises against `δ_t` whose target `r + γ·V(s')` shifts every step. The PRD's lower `actor_lr` (5e-4) vs `critic_lr` (1e-3) is the standard mitigation — slow actor, fast critic.

### F.5 — What would you change with another week?

In priority order:
1. **GAE(λ)** advantage instead of one-step TD — the obvious next variance reducer.
2. **Constrained policy** — replace the entropy bonus with a KL trust region (TRPO/PPO style) so the actor can't collapse on a single critic update.
3. **Replace mean baseline with a learned state-value baseline in REINFORCE** — then the only remaining difference between REINFORCE and A2C is *update frequency* (episodic vs per-step), which would be the cleanest possible ablation of the slide's claim.
4. **Real physiological data** — pull HR, RPE, or recovery proxies into the state vector so the reward could meaningfully reward biology rather than program structure (this is more of a *next project* than a *next week*).
5. **Reward-weight learning** — meta-learn `(λ_1, λ_2)` rather than hand-set them.

## 12. Sources

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.*
- Mnih, V., et al. (2016). *Asynchronous methods for deep reinforcement learning* (A3C / A2C).
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* Chapter 13 (policy gradient methods).
- Ha, D., & Schmidhuber, J. (2018). *World Models.*
- Huang, S., & Ontañón, S. (2022). *A closer look at invalid action masking in policy gradient algorithms.*
- L07 lecture slides — REINFORCE, A2C, Exercise 3 spec.
- Kaggle dataset — *600K+ Fitness Exercise & Workout Program Dataset* (Adnan Elouardi).

## 13. Honest acknowledgements

- The dataset is workout *programs*, not physiological *outcomes*. The LSTM learns the temporal pattern of a chosen program — not muscle growth, fatigue, or injury risk. Treat the system as "structurally realistic recommender", not a medical simulator.
- `MuscleClassifier` is a keyword heuristic (with word-boundary regex). It misclassifies a small share of exercises; the alternative would be a tagged dataset, which we don't have.
- All hyperparameters live in [`configs/setup.json`](configs/setup.json); the README's reported numbers come from `scripts/generate_plots.py` with `world_model.epochs=30`, `reinforce.episodes=60`, `a2c.episodes=60`. Larger budgets generally improve A2C's final reward and REINFORCE's variance.

---

Built layer-by-layer per [`docs/TODO.md`](docs/TODO.md); each layer is a single commit on [`main`](https://github.com/ShakedKozlovsky/RLCourse). 183 tests · ruff clean · 96.63 % coverage.
