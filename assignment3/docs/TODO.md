# TODO — Layered Implementation Plan (Assignment 3)

> Each layer = one or more commits. **Definition of Done** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md).

---

## Layer 0 — Scaffold + planning docs

- [~] Create directory tree under `assignment3/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [~] `docs/TODO.md` (this file)
- [ ] Per-mechanism PRDs: `PRD_reinforce.md`, `PRD_a2c.md`, `PRD_lstm_world_model.md`, `PRD_reward.md`, `PRD_env.md`, `PRD_data.md`, `PRD_action_masking.md`
- [ ] `README.md` placeholder
- [ ] `pyproject.toml` + `uv.lock`
- [ ] `.gitignore`, `.env-example`
- [ ] `configs/setup.json`
- [ ] All `__init__.py` files
- [ ] `shared/version.py` with `__version__ = "1.00"`

**DoD:** repo importable (`python -c "import fitness_rl"` works), docs explain the full plan, no code beyond shared/version.

---

## Layer 1 — Data pipeline (Part B) ✅

- [x] `shared/config.py` — JSON loader with version check + dotted access
- [x] `shared/logger.py` — stdlib logger factory (no print in library)
- [x] `shared/seed.py` — set_global_seed (Python + NumPy + PyTorch)
- [x] `shared/types.py` — Action enum (5 actions), MuscleGroup enum, DailyStep, EpisodeMetrics
- [x] `data/kaggle_loader.py` — RawDataset with `summary` + `detailed` DataFrames
- [x] `data/preprocessor.py` — clean negative sets/reps (interpret as seconds if < 600), drop NaN
- [x] `data/muscle_classifier.py` — **(new, not in original PRD)** keyword-based exercise→muscle classifier with word-boundary regex
- [x] `data/program_selector.py` — pick one program by 4 criteria (equipment + program length + time per workout); reproducible by sorted title
- [x] `data/trajectory_builder.py` — per-day aggregation: total_volume, muscle_distribution (5-dim), session_duration, Rest Day insertion, dominant_muscle
- [x] `data/feature_engineer.py` — 16-dim state vector per day
- [x] `services/data_service.py` — end-to-end orchestrator: load → clean → select → trajectory → features → action labels
- [x] **50/50 tests pass**, **92% coverage**, **ruff clean**, all files ≤ 75 LOC
- [x] **Verified on real Kaggle data**: 84-day trajectory from the chosen program "(MASS MONSTER) High Intensity 4 Day Upper Lower Program" (12 weeks); action distribution PUSH=23, PULL=3, LEGS=22, REST=36

**Discovery noted in PRD evolution v2**: the dataset has no explicit muscle-group column, only `exercise_name`. Added `MuscleClassifier` with word-boundary regex matching. Documented as a heuristic limitation in PRD_data.md.

**DoD met**: pipeline runs end-to-end offline (via fixtures) and on real Kaggle data. Coverage of `data/` ≥ 92%.

---

## Layer 2 — Environment (state, action, reward, mask) ✅

Commit: `Layer 2: ActionSpace + RewardFunction + ActionMask + WorldEnv + tests`

- [x] `environment/action_space.py` — Discrete(5) with named actions
- [x] `environment/reward.py` — gain − λ_1·overload − λ_2·imbalance, stateful rolling window
- [x] `environment/action_mask.py` — guardrails: no 3-consecutive-same-group, no 3-consecutive-rest (excellence differentiator)
- [x] `environment/world_env.py` — Gymnasium-style env over an injectable `TransitionFn` callable (LSTM plugs in at Layer 3)
- [x] Tests for reward arithmetic, action masking, env reset/step/terminate, mask propagation
- [x] **88/88 tests pass**, **94% total coverage** (all four environment modules at 100%), **ruff clean**

**Scope note**: dropped the originally-planned `environment/state.py` — the 16-dim state is built by `data/feature_engineer.py` from `DailyStep` (Layer 1) and the env never owns a richer State object than a flat `np.ndarray`.

**DoD met**: env reset/step contract tested with a stub identity transition + a stub bump transition; reward unit tests pass; coverage 100% for `environment/`.

---

## Layer 3 — LSTM World Model (Part C) ✅

Commit: `Layer 3: LSTMWorldModel + WorldModelService + transition-fn adapter + tests`

- [x] `model/lstm_world_model.py` — LSTM(input=state+action=21, hidden=64, batch_first) + Linear(64, 16) head, `save`/`load` round-trip, `encode_inputs` static helper
- [x] `model/lstm_world_model.py::as_transition_fn` — stateful closure that matches Layer 2's `TransitionFn` callable signature so the trained LSTM plugs straight into `WorldEnv`
- [x] `services/world_model_service.py` — `build_windows`, supervised training (Adam + MSE), chronological 80/20 split, early stopping on val loss with patience
- [x] `TrainResult` dataclass — per-epoch train/val loss arrays for plotting in the GUI
- [x] Tests: forward shape, finiteness, encode-inputs, save/load round-trip, transition-fn adapter, window construction + targets, training reduces loss on synthetic data, early stopping triggers
- [x] **105/105 tests pass**, **95.56% total coverage** (lstm_world_model 100%, world_model_service 98%), **ruff clean**

**DoD partially met**: smoke training works (`test_training_reduces_loss`); checkpoint round-trip works. The actual `world_model.pt` save + loss curve PNG will land in Layer 7 SDK (training entry-point) rather than the model layer itself.

---

## Layer 4 — REINFORCE (Part D) ✅

Commit: `Layer 4: PolicyNet + ReinforceService + episodic policy gradient + tests`

- [x] `model/policy_network.py` — 2-layer MLP (16 → 128 → 128 → 5 logits), accepts batched or single states
- [x] `services/reinforce_service.py` — episode rollout via `WorldEnv`, reward-to-go `G_t`, mean-baseline subtraction, ∇_θ log π · (G−b) update via Adam
- [x] Optional **action masking** (uses `WorldEnv.get_mask()`, adds to logits pre-softmax) and **entropy bonus** in the same service — Layer 9 ablation just toggles flags
- [x] Per-episode `EpisodeMetrics`: total reward, mean entropy, action distribution
- [x] Tests: forward shape, softmax sums to 1, reward-to-go arithmetic, single-step weight change, masking respected during rollout, entropy-bonus effect
- [x] **120/120 tests pass**, **96.02% total coverage** (policy_network 100%, reinforce_service 98%), **ruff clean**

**DoD partially met**: REINFORCE update + masked rollout both verified. The 50-episode training run on the LSTM env + reward-curve PNG + checkpoint save will land in Layer 7 SDK alongside the A2C entry-point.

---

## Layer 5 — A2C (Part E)

Commit: `Layer 5: ActorCriticNet + A2CService + TD-Advantage + tests`

- [ ] `model/actor_critic_network.py` — shared trunk + actor head + critic head
- [ ] `services/a2c_service.py` — one-step TD updates: δ = r + γV(s') − V(s), actor and critic both updated
- [ ] Optional entropy bonus
- [ ] Per-episode metrics + variance comparison vs REINFORCE
- [ ] Tests: AC forward shapes, TD error sign on synthetic transitions, one update step

**DoD:** A2C trains for 50 episodes; reward curve PNG; checkpoint saved; variance comparison plot vs REINFORCE.

---

## Layer 6 — Evaluation + Comparison

Commit: `Layer 6: EvaluationService + ComparisonService + tests`

- [ ] `services/evaluation_service.py` — greedy policy rollout, action distribution, total reward
- [ ] `services/comparison_service.py` — side-by-side REINFORCE vs A2C: reward curves, variance, final mean reward
- [ ] Trajectory analysis: did the policy choose balanced workouts? Detect collapse.
- [ ] Tests on synthetic policies

**DoD:** running `compare()` on the two trained policies produces a comparison JSON + plot.

---

## Layer 7 — SDK + CLI

Commit: `Layer 7: FitnessRL SDK + CLI with interactive menu + tests`

- [ ] `sdk/sdk.py` — facade exposing prepare_data, train_world, train_reinforce, train_a2c, compare, predict
- [ ] `interface/cli/main.py` — Click subcommands + interactive `menu` command
- [ ] `tests/integration/test_sdk.py` — end-to-end smoke
- [ ] `tests/integration/test_cli.py` — Click runner

**DoD:** `uv run fitness-rl --help` and `uv run fitness-rl menu` work; smoke tests pass.

---

## Layer 8 — PyQt6 GUI

Commit: `Layer 8: PyQt6 GUI with 5 tabs on top of SDK`

- [ ] `interface/gui/main_window.py` — QMainWindow with 5 tabs
- [ ] `interface/gui/data_tab.py` — load data, show synthetic trajectory shapes
- [ ] `interface/gui/world_model_tab.py` — train LSTM, plot loss
- [ ] `interface/gui/reinforce_tab.py` — train REINFORCE, plot reward
- [ ] `interface/gui/a2c_tab.py` — train A2C, plot reward
- [ ] `interface/gui/compare_tab.py` — comparison plot + action distribution
- [ ] Plot widget + QThread workers (off-thread training)
- [ ] Headless smoke test under `QT_QPA_PLATFORM=offscreen`

**DoD:** GUI launches; all 5 tabs render; screenshots in `assets/gui/`.

---

## Layer 9 — Experiments + Action Masking (Excellence)

Commit: `Layer 9: action masking experiment + reward-weight sweep + collapse detection`

- [ ] Action masking experiment: REINFORCE+A2C with masking on/off → compare
- [ ] Reward-weight sweep: vary λ_1, λ_2 → policy behavior
- [ ] Action distribution analysis per episode (detect collapse to single action)
- [ ] Plots for each experiment

**DoD:** all three differentiator experiments produce JSON + PNG outputs documented in README.

---

## Layer 10 — Final README + reflection answers

Commit: `Layer 10: full README + 5 reflection answers + comparison tables + screenshots`

- [ ] README §1–13 covering: project goal, RL framing, slide mapping, data spec, state/action/reward, LSTM world model, REINFORCE, A2C, comparison, GUI/CLI usage, tests, **5 reflection questions** from Part F, sources
- [ ] All plots embedded inline
- [ ] Architecture + class diagrams as PNG
- [ ] Pre-trained checkpoints in `saved_models/`
- [ ] Executive summary doc for fast grader read

**DoD:** every requirement in the assignment maps to a section in README. Self-grading checklist signed off.

---

## Cross-cutting (every commit)

- [ ] `ruff check` returns 0
- [ ] Each commit: `Layer N: <summary>` + bullet body
- [ ] No magic numbers — everything in `configs/`
- [ ] No file > 150 LOC
- [ ] After each layer: update this TODO.md, mark items `[x]`

## Future extensions (not in scope)

- [ ] PPO over the world model
- [ ] Multi-trainee personalization (different program × policy combinations)
- [ ] Real physiological data integration (HR, RPE, recovery)
- [ ] Continuous action space (workout intensity as a real number)
