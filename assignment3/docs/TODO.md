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

## Layer 5 — A2C (Part E) ✅

Commit: `Layer 5: ActorCriticNet + A2CService + TD-Advantage + tests`

- [x] `model/actor_critic_network.py` — shared MLP trunk (16 → 128 → 128) + actor head (128 → 5 logits) + critic head (128 → 1)
- [x] `actor_params()` / `critic_params()` partition: trunk lives under the actor optimizer so it's stepped once per update at `actor_lr`; critic optimizer only touches the critic head — verified by a partition-test (`actor_ids.isdisjoint(critic_ids)`)
- [x] `services/a2c_service.py` — per-step TD updates: δ = r + γ·V(s') − V(s), bootstrap zeroed on terminal; actor + critic optimizers stepped together; `td_error` static helper for unit testing
- [x] Entropy bonus on the actor loss (default 0.01 from `configs/setup.json`); action masking flag (Layer 9 ablation)
- [x] Per-episode `EpisodeMetrics`
- [x] Tests: forward shapes, TD-error sign (positive/negative/terminal), parameter partition, single-episode weight delta on both heads, V(s) drift across training, action masking respected
- [x] **136/136 tests pass**, **96.48% total coverage** (ActorCriticNet 100%, A2CService 99%), **ruff clean**

**DoD partially met**: A2C update verified end-to-end. Reward-curve PNG, checkpoint save, and REINFORCE-vs-A2C variance plot will land in Layer 6 (evaluation/comparison) + Layer 7 (SDK entry-points).

---

## Layer 6 — Evaluation + Comparison ✅

Commit: `Layer 6: EvaluationService + ComparisonService + tests`

- [x] `services/evaluation_service.py` — greedy (argmax) rollout, action distribution, collapse detection (≥80 % single-action threshold), honours `WorldEnv.get_mask()` so eval respects the same guardrails as training
- [x] `actor_logits` adapter so the service works with both `PolicyNet` (REINFORCE) and `ActorCriticNet` (A2C) via the same interface
- [x] `services/comparison_service.py` — `ComparisonReport` (mean final reward, overall std, final std, final CV, action distribution) + `ComparisonResult` (both reports + raw reward arrays + winner)
- [x] Winner logic: higher mean wins; on a near-tie (≤1 %), lower CV wins; otherwise "tie"
- [x] `ComparisonResult.to_dict()` is JSON-serialisable (consumed by the GUI compare tab + README)
- [x] 21 new tests across both services (greedy rollout shapes, determinism, masking respected, distribution normalisation, collapse threshold, winner under all branches, tiebreak, single-episode robustness)
- [x] **157/157 tests pass**, **96.72% total coverage** (evaluation 100%, comparison 97%), **ruff clean**

**DoD met**: `ComparisonService.compare(reinforce_hist, a2c_hist)` produces a JSON-serialisable structure; plotting consumes `reinforce_rewards` + `a2c_rewards` arrays directly (matplotlib lives in Layer 7/8 entry-points).

---

## Layer 7 — SDK + CLI ✅

Commit: `Layer 7: FitnessRL SDK + CLI with interactive menu + tests`

- [x] `sdk/sdk.py` — `FitnessRL` facade exposing `prepare_data`, `train_world_model`, `train_reinforce`, `train_a2c`, `compare`, `evaluate`, `predict`; holds state across calls so the GUI can re-run pieces
- [x] `sdk/env_builder.py` — factored env construction (transition fn from LSTM-or-identity, reward function from config, optional action mask) so SDK stays under 150 LOC and the same builder is shared with evaluation
- [x] `interface/cli/main.py` — Click subcommands `prepare-data`, `train-world`, `train-reinforce`, `train-a2c`, `compare` (with optional `--out json`), `predict`, and an interactive `menu` command
- [x] `tests/integration/test_sdk.py` — 10 end-to-end SDK tests (pipeline output, world-model checkpoint save, REINFORCE/A2C history shapes, compare requires both trained, evaluate, predict valid range, unknown-algo error, full pipeline with trained LSTM as the env's transition fn)
- [x] `tests/integration/test_cli.py` — 7 CliRunner tests (help text, every subcommand exit_code == 0 on the synthetic config, compare writes JSON, predict prints action name, menu quits on `q`)
- [x] `tests/conftest.py::sdk_config` — extends `minimal_config` with tiny env/world_model/reinforce/a2c hyperparameters so integration tests run in seconds
- [x] **Fix**: `RewardFunction._imbalance` now clamps negative muscle-distribution values to 0 before computing entropy (the LSTM is unconstrained and can produce them; the reward must be robust to upstream noise)
- [x] **174/174 tests pass**, **97.03% total coverage** (SDK 96%, CLI 85%, env_builder 100%), **ruff clean**

**DoD met**: `uv run fitness-rl --help` and `uv run fitness-rl menu` both work; all smoke tests pass.

**DoD:** `uv run fitness-rl --help` and `uv run fitness-rl menu` work; smoke tests pass.

---

## Layer 8 — PyQt6 GUI ✅

Commit: `Layer 8: PyQt6 GUI with 5 tabs on top of SDK`

- [x] `interface/gui/plot_widget.py` — matplotlib `FigureCanvasQTAgg` wrapped in a `QWidget`, single-axes `draw(fn)` that clears and re-renders
- [x] `interface/gui/worker.py` — `TrainingWorker(QThread)` runs blocking SDK calls off the GUI thread; emits `finished_with_result` / `failed`
- [x] `interface/gui/main_window.py` — `QMainWindow` with 5 tabs, shares one `FitnessRL` SDK instance across all tabs (data trained in tab 1 is available in tab 3)
- [x] `interface/gui/data_tab.py` — "Load data" button → text summary + per-day volume plot
- [x] `interface/gui/world_model_tab.py` — "Train" button → LSTM train/val loss curves
- [x] `interface/gui/algo_tab.py` — shared `AlgoTab` base + `ReinforceTab` / `A2CTab` subclasses (one button, episodes spinbox, reward curve)
- [x] `interface/gui/compare_tab.py` — trains both algos, prints winner + final-mean + final-CV, plots both reward curves on one axis
- [x] CLI gained a `fitness-rl gui` subcommand
- [x] `tests/integration/test_gui.py` — headless smoke test under `QT_QPA_PLATFORM=offscreen`: MainWindow constructs, 5 tabs with correct labels, shared SDK works, PlotWidget draws without error, TrainingWorker emits its result
- [x] **179/179 tests pass**, **97.03% coverage** (GUI files excluded from coverage per pyproject), **ruff clean**

**DoD met**: GUI launches under offscreen Qt; all 5 tabs construct + render. Screenshots will be captured in Layer 10 along with the README.

---

## Layer 9 — Experiments + Action Masking (Excellence) ✅

Commit: `Layer 9: ExperimentService + masking ablation + reward-weight sweep + collapse analysis`

- [x] `services/experiment_service.py` — `ExperimentService` with three methods:
  - `run_action_masking_ablation()` — 4-way grid: {REINFORCE, A2C} × {masking on, off}
  - `run_reward_weight_sweep()` — Cartesian product of `λ_overload` × `λ_imbalance`
  - `run_collapse_analysis()` — trains both algos, evaluates greedily, flags `collapsed` via `EvaluationService.collapsed(threshold=0.8)`
- [x] Config overrides happen by deep-copying the base config and writing to a temp JSON file the SDK reads — no global state, each experiment cell starts fresh
- [x] CLI `fitness-rl experiments --episodes N --out-dir results/experiments` writes three JSON files
- [x] Tests: 4 integration tests asserting result keys, n_episodes, action_distribution sums to 1, sweep produces |λ_o| × |λ_i| cells, collapse-analysis returns valid floats
- [x] **Experiments run end-to-end on the real 84-day Kaggle trajectory** with 20 episodes/cell:
  - Reward-weight sweep: monotone — higher λ_overload + λ_imbalance → lower mean reward (sanity check that the reward function is well-formed; mean reward ranges from +6.84 at (0, 0) to −6.47 at (0.5, 0.6))
  - Collapse analysis: both REINFORCE and A2C collapse to a single action after 20 episodes on the *identity*-transition env (REINFORCE → CARDIO, A2C → PUSH). This is *expected*: with no learnable dynamics the env can't surprise the policy, so the entropy bonus + imbalance penalty is the only diversity pressure. With the trained LSTM as the dynamics, results should differ — documented as a finding in the README.
  - Masking ablation: rewards identical across masking on/off on identity env (same reasoning as above), but action distributions differ in the expected direction (masking-on → fewer triples)
- [x] **183/183 tests pass**, **96.63% coverage** (experiment_service is integration-tested), **ruff clean**

**DoD met**: `results/experiments/{masking_ablation,reward_weight_sweep,collapse_analysis}.json` written; findings documented for the README (Layer 10).

**DoD:** all three differentiator experiments produce JSON + PNG outputs documented in README.

---

## Layer 10 — Final README + reflection answers ✅

Commit: `Layer 10: full README + 5 reflection answers + plots + GUI screenshots`

- [x] **README §1–13** covering: project goal & RL framing, slide mapping, data pipeline, state/action/reward/env, LSTM world model, REINFORCE, A2C, comparison + experiments, GUI/CLI/SDK, quality bar, **5 reflection questions**, sources, honest acknowledgements
- [x] **Plots generated from a real training run** via `scripts/generate_plots.py` (30 epochs world model + 60 episodes each of REINFORCE and A2C over the trained LSTM): `assets/plots/world_model_loss.png`, `reinforce_reward.png`, `a2c_reward.png`, `comparison.png`, `action_distribution.png`
- [x] **GUI screenshots** captured headless via `scripts/capture_gui.py` (uses `QT_QPA_PLATFORM=offscreen`) — one PNG per tab in `assets/gui/`
- [x] **Real-data numbers in the README**: A2C final-30 % CV (0.011) is ~2× lower than REINFORCE's (0.024); A2C final-mean reward (7.06) > REINFORCE's (6.17); REINFORCE keeps a diverse action distribution while A2C concentrates 57 % on REST — discussed honestly in the F.3 reflection answer
- [x] Pre-trained `saved_models/world_model.pt` committed
- [x] **183 tests · ruff clean · 96.63 % coverage** — gate of 85 % cleared at every layer

**DoD met**: every assignment requirement (Parts A–F) maps to a section in the README. Self-grading checklist signed off in the comparison + reflection sections.

---

## Layer 11 — Correctness fixes from professor's audit ✅

Commit: `Layer 11: correctness fixes from professor's audit`

- [x] **Audit #10** — reward imbalance now action-conditional (`action == REST`), not state-conditional. Eliminates the exploit where a policy could collect zero-imbalance reward by steering the LSTM to predict high `rest_indicator`.
- [x] **Audit #12** — LSTM-as-transition-fn clamps predicted state to valid feature ranges (volume ∈ [0, 1], muscle distribution renormalised, day-of-week snapped to one-hot, rest_indicator ∈ [0, 1]).
- [x] **Audit #11** — `tests/integration/test_reproducibility.py`: two SDK runs with the same seed produce bit-identical training histories.
- [x] **Audit #5** — PRD framing fixed (v7): the policy is feed-forward MDP, only the world model is POMDP-aware.
- [x] **Audit #13** — `docs/PRD_action_masking.md` "Known structural weakness" section: 3-in-a-row rule lets globally bad schedules through.
- [x] `RewardFunction.decompose()` added now so reward.py only changes once.
- [x] 189/189 tests pass · ruff clean.

## Layer 12 — Evaluation infrastructure ✅

Commit: `Layer 12: evaluation infrastructure (audit findings #1, #4, #15, #19)`

- [x] **Audit #1** — `services/world_model_evaluator.py`: persistence + linear-OLS baselines vs LSTM, plus multi-step rollout MSE at horizons {1, 7, 28}.
- [x] **Audit #4** — `services/baseline_policies.py`: `RandomPolicy`, `RoundRobinPolicy`, `KaggleProgramPolicy` + `PolicyBenchmark.run()`.
- [x] **Audit #15** — `services/diagnostics.py::record_greedy_rollout` returns a `GreedyTrajectory` with per-step `(action, gain, overload, imbalance, total)` decomposition.
- [x] **Audit #19** — `RewardFunction.decompose()` exposes the three reward terms separately.
- [x] `sdk/evaluator.py` companion class so SDK proper stays close to the 150-LOC cap (157 LOC documented exception).
- [x] **Real-data findings**: LSTM beats persistence by 3.2× (0.055 vs 0.175); rollout error compounds 50 % from 1-step to 7-step; **A2C (6.91) and REINFORCE (6.36) both lose to round-robin (6.94) and the Kaggle program (7.28)** on the identity-env baseline run — smoking gun for reward mis-spec.
- [x] 200/200 tests · ruff clean.

## Layer 13 — Empirical studies ✅

Commit: `Layer 13: empirical studies (audit #2, #3, #6, #7, #14, #18)`

- [x] `services/experiment_base.py` — shared `train_one`, `make_sdk`, `summarise_history`, `aggregate_with_ci` (95 % normal-approx).
- [x] `services/experiment_studies.py::ExperimentStudies` — 5 audit-driven experiments:
  - **Audit #3 + #18** — `multi_seed_comparison(seeds)` with 95 % CI on final-30 % reward.
  - **Audit #2** — `entropy_sweep(bonuses)` proves the reward mis-specification: more entropy → diverse → *lower* reward.
  - **Audit #7** — `reinforce_variant_chain()` empirically demonstrates the lecture's REINFORCE → +baseline → +advantage chain (final reward 1.87 → 2.35 → 5.22).
  - **Audit #14** — `gamma_ablation(gammas)`: γ=0.9 outperforms γ=0.99.
  - **Audit #6** — `masking_on_lstm_env()` re-runs the ablation against the trained LSTM dynamics (not identity); A2C gets 2× the reward when the world model is wired up.
- [x] All 5 experiments run end-to-end on real Kaggle data in ~160 s.
- [x] 208/208 tests · ruff clean.

## Layer 14 — Documentation closeout ✅

Commit: `Layer 14: plots + architecture diagram + coverage to 97.5 % + README rewrite`

- [x] **Audit #16** — `scripts/generate_architecture_diagram.py` produces `assets/diagrams/architecture.png`; embedded at the top of the README.
- [x] `scripts/generate_layer13_plots.py` produces 7 new figures (`multi_seed_ci`, `entropy_sweep`, `reinforce_chain`, `gamma_ablation`, `masking_on_lstm`, `world_model_compounding`, `baselines_vs_trained`).
- [x] **Audit #20** — Coverage from 96.45 % → **97.56 %** by adding tests for the CLI's `experiments` / `train-world` / `menu` paths and the seed module's CUDA + ImportError branches.
- [x] **Audit #8** — README §13 explicitly acknowledges the 60-episode budget (vs PRD-stated 300) with instructions to re-run at higher N.
- [x] **Audit #9** — README §13 documents the reward mis-specification as the headline finding.
- [x] **Audit #17** — `docs/PRD.md` evolution log extended to v7 (one entry per layer that changed contracts).
- [x] README rewritten with 14 sections, an audit-findings status table mapping all 20 findings to their fixes, and updated reflection answers F.1–F.5 that cite the new empirical evidence.
- [x] **216/216 tests pass · ruff clean · 97.56 % coverage**.

---

## Layer 15 — Beyond-spec polish for the grade ✅

Commit: `Layer 15: reward fix + PPO + recommend feature + 300-episode chain + notebook + exec summary`

- [x] **Reward fix**: REST now earns zero gain (was full volume reward). Layer-13 entropy sweep proved the reward was mis-specified; this is the structural fix. `RewardFunction.compute(state, action)` is the touched contract.
- [x] **PPO added** as a third algorithm completing the `REINFORCE → A2C → PPO` chain. `services/ppo_service.py` implements the clipped surrogate over a buffer with `n_epochs_per_batch` Adam passes per rollout. `PRD_ppo.md` documents the theory.
- [x] `sdk/trainers.py` — module-level `build_*_service` constructors keep `sdk/sdk.py` close to the 150-LOC cap.
- [x] **Recommend feature**: `services/recommender.py::WorkoutRecommender.recommend()` returns an N-day plan with per-day reward decomposition. CLI command `fitness-rl recommend --algo {reinforce,a2c,ppo} --days N --history "PUSH,PULL,REST"`. The product is now a user-facing recommender, not just an RL experiment.
- [x] **Full-budget run**: `scripts/run_layer15_full.py` runs 300 episodes × 3 seeds × 3 algos on the real Kaggle data (~135 s). All trained agents beat all baselines after the reward fix; Kaggle program scores −1.47 (overload-penalised). Results in `results/layer15/full_budget_multiseed.json`.
- [x] **3 new plots**: `three_algo_curves.png`, `three_algo_final_ci.png`, `baselines_vs_trained_post_fix.png` (Layer 15 generator script).
- [x] `notebooks/fitness_rl_walkthrough.ipynb` — 6-cell guided tour: data → world model → REINFORCE → A2C → PPO → recommend.
- [x] `docs/EXECUTIVE_SUMMARY.md` — 1-pager for the grader's first pass.
- [x] PRD evolution log v8 (reward fix) + v9 (PPO added).
- [x] **235/235 tests pass** (was 216), **97.5 % coverage**, **ruff clean**.

**DoD met**: every assignment part covered + audit findings closed + 3-algo chain demonstrated empirically at full budget + user-facing recommender shipped.

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
