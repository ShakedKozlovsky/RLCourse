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

## Layer 1 — Data pipeline (Part B)

Commit: `Layer 1: Kaggle loader + program selector + trajectory builder + tests`

- [ ] `shared/config.py` — JSON loader with version check
- [ ] `shared/logger.py` — stdlib logger factory
- [ ] `shared/seed.py` — set_global_seed
- [ ] `shared/types.py` — Action enum, State, EpisodeMetrics
- [ ] `data/kaggle_loader.py` — read `program_summary.csv` + `programs_detailed_boostcamp_kaggle.csv`
- [ ] `data/preprocessor.py` — clean negative sets/reps, parse fields
- [ ] `data/program_selector.py` — pick one program by 8 criteria
- [ ] `data/trajectory_builder.py` — daily aggregation: total_volume, muscle_distribution, etc.; Rest Day insertion
- [ ] `data/feature_engineer.py` — produce 16-dim state vector per day
- [ ] `services/data_service.py` — end-to-end orchestrator
- [ ] Tests for every module + synthetic CSV fixtures

**DoD:** pipeline runs offline on a minimal synthetic CSV fixture; tests pass; coverage ≥ 90% for `data/`.

---

## Layer 2 — Environment (state, action, reward, mask)

Commit: `Layer 2: State + ActionSpace + RewardFunction + WorldEnv + tests`

- [ ] `environment/state.py` — State dataclass and state-vector construction
- [ ] `environment/action_space.py` — Discrete(5) with named actions
- [ ] `environment/reward.py` — gain − λ_1·overload − λ_2·imbalance
- [ ] `environment/action_mask.py` — guardrails: no 3-consecutive-same-group, no 3-consecutive-rest
- [ ] `environment/world_env.py` — Gymnasium-style env wrapping the world model (placeholder until Layer 3)
- [ ] Tests for reward arithmetic, action masking, env reset/step shapes

**DoD:** env reset/step contract tested with a stub world model that returns identity transitions; reward unit tests pass; coverage ≥ 90% for `environment/`.

---

## Layer 3 — LSTM World Model (Part C)

Commit: `Layer 3: LSTMWorldModel + supervised training service + tests`

- [ ] `model/lstm_world_model.py` — LSTM(input=state+action, hidden=64) + linear head
- [ ] `services/world_model_service.py` — supervised training on rolling windows from trajectory
- [ ] Loss curve plot + early stopping by validation
- [ ] Tests: forward shape, training one step reduces loss on synthetic data, save/load round-trip

**DoD:** smoke training on synthetic trajectory completes; `world_model.pt` saved; loss curve PNG in `assets/plots/`.

---

## Layer 4 — REINFORCE (Part D)

Commit: `Layer 4: PolicyNet + ReinforceService + episodic policy gradient + tests`

- [ ] `model/policy_network.py` — 2-layer MLP returning logits
- [ ] `services/reinforce_service.py` — episode rollout via WorldEnv, reward-to-go G_t, mean-baseline subtraction, ∇_θ log π · (G−b) update
- [ ] Per-episode metrics: total reward, mean entropy, action distribution
- [ ] Tests: policy forward shape, episode rollout shape, one update reduces a synthetic loss

**DoD:** REINFORCE trains for 50 episodes on the synthetic LSTM env; reward curve PNG; checkpoint saved.

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
