# PRD — REINFORCE + A2C Fitness Recommender (Assignment 3)

> **Course:** Reinforcement Learning with Deep Learning
> **Lecture:** L07 — REINFORCE, A2C and Exercise 3
> **Author:** Shaked Kozlovsky (ID 208904839)
> **What this project IS:** an educational, full-stack policy-gradient RL system that learns a daily *workout recommendation policy* via REINFORCE and A2C, operating over an LSTM-learned world model.
> **What this project is NOT:** a fitness coach, a medical recommendation, or a production health app. The data is a workout-program catalogue, not physiological measurements of real trainees.

---

## 0. PRD evolution log

The PRD evolves as the project advances; each layer's findings feed back into the requirements.

| Version | Layer | What changed and why |
|---|---|---|
| v1 (Layer 0) | Planning | Initial requirements from the assignment PDF §7 + lecture slides + coding rules V3. |

---

## 1. Project goal

Demonstrate the full policy-gradient learning pipeline:

1. **Formulate** workout recommendation as an MDP with sequential decisions.
2. **Build** a synthetic per-day trajectory from one chosen Kaggle workout program.
3. **Learn** an LSTM-based World Model `s_{t+1} ≈ f_φ(s_t, a_t, h_t)` via supervised training on the trajectory.
4. **Train** a REINFORCE policy over the learned world model.
5. **Train** an A2C policy (Actor + Critic) over the same world model.
6. **Compare** the two algorithms: stability, sample efficiency, action diversity, reward variance.

The grade rewards understanding of REINFORCE → Actor-Critic, the role of baseline/advantage in variance reduction, and honest analysis of where each method works and fails. A clean "the policy beats random" outcome is not the point — the *process* (controlled experiments + algorithmic comparison) is.

## 2. Mapping to the lecture slides

| Lecture concept | Slide | Where it appears in this project |
|---|---|---|
| Policy-gradient learning flow | 1 | The full pipeline: Policy → Action → Reward → Update by Return/Advantage. |
| REINFORCE objective `J(θ) = E_τ[R(τ)]` | 4 | `services/reinforce_agent.py::compute_loss`. |
| Policy-gradient estimator with `∇ log π · R(τ)` | 5 | Same module; gradient computed via PyTorch autograd. |
| Baseline subtraction `(R − b)` | 6 | Optional baseline (mean return) in REINFORCE; documented as variance-reduction. |
| Reward-to-Go `G_t = Σ γ^{k−t} r_k` | 7 | Replaces raw return for better credit assignment. |
| Advantage `A_t = G_t − V(s_t)` | 7, 15 | `services/a2c_agent.py` uses `A_t ≈ δ_t = r + γV(s') − V(s)`. |
| Actor-Critic two networks | 9 | `model/actor_critic_network.py` has two heads (policy + value). |
| Actor-Critic core equations (TD error, actor update, critic update) | 10 | Directly implemented in `services/a2c_agent.py`. |
| Closed Actor-Critic cycle | 11 | `_run_episode()` matches the slide's 5-step loop. |
| Stability advantage of A2C over REINFORCE | 21 | The §"Experiments" section in README reports variance/return curves for both. |

## 3. RL problem formulation (Part A)

The MDP is the contract between the fitness domain and policy gradient learning:

| Component | This project |
|---|---|
| **State `s_t`** | Vector of trainee features at day `t`: 7-day rolling volume, muscle-distribution vector (5 groups), session duration, week index, day in cycle. Approximate Markov state via LSTM hidden representation. |
| **Action `a_t`** | One of **5 discrete actions**: `PUSH=0, PULL=1, LEGS=2, CARDIO=3, REST=4`. Chosen by π_θ(a\|s). |
| **Reward `r_t`** | `gain_t − λ_1 · overload_penalty_t − λ_2 · imbalance_penalty_t`. Defined explicitly in `environment/reward.py`. |
| **Transition `P(s_{t+1}\|s_t, a_t)`** | Unknown in reality. **Approximated by an LSTM**: `s_{t+1} ≈ f_φ(s_t, a_t, h_t)` trained supervised on the synthetic trajectory. |
| **Episode** | A fixed-length 28-day rollout starting from the trajectory's first state (or a sampled mid-trajectory state). |
| **Policy** | Stochastic `π_θ(a\|s) = softmax(actor_logits(s))`. Sampled during training; argmax during evaluation. |
| **Return** | `G_t = Σ_{k=t}^T γ^{k-t} · r_k` with γ from config (default 0.99). |

**Why "sequential decision-making":** the trainee's state at day `t+1` depends on the cumulative history of workouts. A good action today affects what's possible tomorrow (muscle recovery, accumulated fatigue, training volume). REINFORCE and A2C are designed exactly for this credit-assignment problem.

## 4. Dataset (Part B)

| Parameter | Mandatory value | Source |
|---|---|---|
| Source | Kaggle (manual download) | https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset |
| Files | `program_summary.csv`, `programs_detailed_boostcamp_kaggle.csv` | Placed in `data/raw/` |
| Total exercise rows | 605,033 (assignment-stated) | Verified at load |
| Total unique programs | 2,598 (assignment-stated) | Verified at load |
| Data sanity | Some negative `sets`/`reps` cells must be filtered or interpreted as seconds | `data/preprocessor.py::clean_numeric_fields` |

### Synthetic trainee construction (Part B.4)

1. **Program selection** (`data/program_selector.py`): pick exactly one program from `program_summary.csv` matching the 8 criteria:
   - `equipment = "Full Gym"`
   - `program_length` in a reasonable range (e.g., 4–12 weeks)
   - `time_per_workout` in 45–120 minutes
   - Records as `chosen_program`.
2. **Detailed-row filter** (`data/program_selector.py`): keep only rows belonging to `chosen_program`.
3. **Daily aggregation** (`data/trajectory_builder.py`): group by `week, day` and compute per-day:
   - `total_volume_t = Σ_i (sets_i · reps_i)`
   - `muscle_distribution_t` — normalized volume per muscle group (5-dim)
   - `session_duration_t`, `week_index_t`, `day_in_cycle_t`
   - **Rest Days**: weeks/days with no training entries → zero-volume Rest Day row
4. **Trajectory**: a sequence `(s_1, s_2, ..., s_T)` where T is the program's day count. This is the *one synthetic trainee's trajectory*.

## 5. LSTM World Model (Part C)

| Item | Decision |
|---|---|
| Input | `(s_t, a_t)` concatenated, plus the hidden state `h_t` |
| Output | Predicted `s_{t+1}` (same shape as `s_t`) |
| Architecture | 1-layer LSTM (`hidden_size=64`) + `Linear(64, state_dim)` head |
| Training | Supervised on rolling windows of the trajectory: input window of length `W`, target = next state |
| Loss | Mean Squared Error on continuous state features |
| Split | 80/20 train/val by chronological window position |

**Why the LSTM is the environment, not the recommender:** the LSTM approximates the *transition function* — what state comes next given current state + action. It does **not** choose actions. The policy (REINFORCE / A2C) chooses actions, and the world model tells us the consequence.

## 6. REINFORCE (Part D)

```
Policy π_θ(a|s) = softmax( PolicyNet(s) )

For each episode τ:
    rollout T steps using the LSTM world model
    compute G_t = Σ_{k=t}^T γ^{k-t} · r_k       (reward-to-go)
    optionally subtract baseline b = mean(G_t)  (variance reduction)
    θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b)
```

- **Policy network:** 2-layer MLP, state → hidden(128) → ReLU → action_logits(5).
- **Baseline option:** mean-over-episode baseline (no learned baseline) — kept simple to highlight A2C's contribution.
- **Episodes:** 300, 28 days each, γ = 0.99, lr = 1e-3.

## 7. A2C (Part E)

```
Actor:  π_θ(a|s) = softmax( ActorHead(s) )
Critic: V_ψ(s)   = ValueHead(s)

For each step (s, a, r, s'):
    δ_t = r + γ · V_ψ(s') − V_ψ(s)
    Actor  update: θ ← θ + α · ∇_θ log π_θ(a|s) · δ_t
    Critic update: ψ ← ψ + β · δ_t · ∇_ψ V_ψ(s)
```

- **Shared trunk:** state → hidden(128) → ReLU → [actor head, critic head].
- **Updates:** one-step TD; bootstrapped, low-variance per step but biased by V's accuracy.
- **Loss:** policy loss + value loss (MSE on TD error) + (optional) entropy bonus for exploration.

## 8. Reward function

```
r_t = gain_t − λ_1 · overload_penalty_t − λ_2 · imbalance_penalty_t
```

| Term | Definition |
|---|---|
| `gain_t` | "Useful training volume" — reward proportional to `total_volume_t` when within healthy bounds. |
| `overload_penalty_t` | Penalty for high cumulative recent volume (proxy for over-training risk). |
| `imbalance_penalty_t` | Penalty for repeating the same muscle group, or skewed muscle distribution. |

Defaults: `λ_1 = 0.2`, `λ_2 = 0.3`. All in `configs/setup.json`. The values are pedagogical, not medical.

## 9. Action Masking (Excellence — §F.1)

To inject expert knowledge: forbid 3 consecutive same-group days, and forbid REST after 2 consecutive REST days. Implemented by setting the logits of forbidden actions to `-∞` before softmax. Documented as the "expert guardrails" mechanism in README. Enabled via `configs/setup.json:env.action_masking_enabled`.

## 10. KPIs and acceptance criteria

| KPI | Target | Where |
|---|---|---|
| Test coverage | ≥ 85% | `pytest --cov` |
| `ruff check` | 0 errors | linter gate |
| File size | ≤ 150 lines per .py file | enforced |
| LSTM converges | Validation MSE drops monotonically over training | `assets/plots/lstm_loss.png` |
| REINFORCE converges | Mean episode reward trends up over training | `assets/plots/reinforce_reward.png` |
| A2C converges | Mean episode reward trends up; **lower variance than REINFORCE** | `assets/plots/a2c_reward.png` + variance comparison |
| Policy not collapsed | Action distribution shows ≥ 3 of 5 actions in non-trivial fractions on test | `assets/plots/action_distribution.png` |
| Reproducibility | Same seed → same results within tolerance | `configs/setup.json:seed = 208904839` |
| Architecture purity | No policy/model logic in GUI/CLI; SDK is the only consumer entry | code review |

## 11. Non-functional requirements

- **OOP-first:** every responsibility is a class.
- **Configuration:** no hardcoding; all in `configs/setup.json`.
- **uv** is the only package manager.
- **TDD:** Red → Green → Refactor for at least 2 components (Reward + LSTM forward).
- **Logging:** through `shared/logger.py`; no print in library code.
- **No internet at evaluation time:** Kaggle CSV is downloaded once into `data/raw/`.

## 12. Deliverables

- Source under `src/fitness_rl/`
- Tests under `tests/` with ≥85% coverage
- `docs/` with PRD, PLAN, TODO, per-mechanism PRDs
- `configs/setup.json`
- `assets/` with screenshots, training plots, comparison charts
- `results/` with experiment outputs
- README with formulas, plots, 5 reflection answers (Part F)
- Pre-trained checkpoints in `saved_models/`

## 13. Out of scope

- Real physiological metrics (heart rate, RPE, recovery markers)
- Multi-trainee personalization
- Continuous action space
- Off-policy methods (PPO, SAC, DDPG)
- Online learning from real users
