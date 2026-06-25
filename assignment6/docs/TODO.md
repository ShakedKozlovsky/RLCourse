# TODO — Layered Implementation Plan (Assignment 6 — MARL)

> **STATUS — v1.08 — 27 LAYERS + 17 BEYOND-SPEC EXTENSIONS — ALL DONE** ✅
>
> This file is the historical implementation plan that drove the layered build. All `[x]` items are committed up to tag **`marl-lab-v1.08`**. The few `[ ]` remaining are either (a) manual user steps the codebase can't do for you, or (b) plan items that got rolled into a different artifact (called out inline). For the current per-layer status see the table in [`../README.md`](../README.md#status); for the bonus extensions see the "Beyond the spec" section there; for the version-by-version story see [`CHANGELOG.md`](CHANGELOG.md).

> Each layer = one commit. **Definition of Done** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md), [`CHANGELOG.md`](CHANGELOG.md), [`PROOFS.md`](PROOFS.md).

---

## Layer 0 — Scaffold + planning docs

- [x] Directory tree under `assignment6/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [x] `docs/TODO.md` (this file)
- [x] Per-mechanism PRDs: `PRD_dec_pomdp.md`, `PRD_game.md`, `PRD_ctde.md`, `PRD_olora.md`, `PRD_mcp.md`, `PRD_gmail.md`, `PRD_partial_observation.md`, `PRD_iql_baseline.md`
- [x] `README.md` placeholder
- [x] `pyproject.toml` (with FastMCP, Prefect, Google APIs, PyYAML deps)
- [x] `.gitignore` (with secrets carve-out), `.env-example`
- [x] `configs/setup.yaml`
- [x] All `__init__.py` files (24 total)
- [x] `shared/version.py` with `__version__ = "1.00"`

**DoD:** repo importable (`uv run python -c "import marl_lab"`), docs explain the full plan, no code beyond shared/version.

---

## Layer 1 — Shared layer + ConfigManager (YAML)

Commit: `Layer 1: shared/* + YAML config + types`

- [x] `shared/config.py` — YAML loader with `version` check + dotted access (`cfg.get("marl.tau")`)
- [x] `shared/logger.py` — stdlib factory (no print in library)
- [x] `shared/seed.py` — `set_global_seed(int)` for Python + NumPy + PyTorch
- [x] `shared/types.py` — `Obs`, `JointAction`, `Transition`, `EpisodeSequence`, `SubGameResult`, `GameReport`, `StepDiagnostic`, `TrainResult`
- [x] Tests: YAML loads, version mismatch raises, dotted access works, types frozen

**DoD:** ConfigManager passes 6+ tests; version check identical to A4/A5 pattern.

---

## Layer 2 — Game core (board, moves, win adjudication, barriers)

Commit: `Layer 2: game core — board + moves + win adjudication + barriers`

- [x] `game/board.py` — `Board(grid_size, cop_pos, thief_pos, barriers, step)` dataclass
- [x] `game/actions.py` — `Action` enum (UP, DOWN, LEFT, RIGHT, STAY, PLACE_BARRIER)
- [x] `game/moves.py` — `MoveDynamics.apply(board, joint_action) → new_board, info`
- [x] `game/win.py` — `WinAdjudicator.check(board) → "cop"|"thief"|None`
- [x] `game/barriers.py` — `BarrierPlacement` (validity, max-5 cap)
- [x] `game/sub_game.py` — `SubGameRunner` (25-move cap)
- [x] `game/game.py` — `Game` (6 sub-games), accumulates `GameReport`
- [x] Tests: move validity on 5x5; collisions cap at walls/barriers; barrier placement counter; capture-on-overlap; 6-sub-game accounting

**DoD:** 100 % branch coverage in `game/`; pure functions throughout; 4-test scoring battery.

---

## Layer 3 — Partial observation + Dec-POMDP env + reward

Commit: `Layer 3: sensor + dec_pomdp env + reward functions`

- [x] `sensor/partial_observation.py` — `observe(global_state, agent_id, radius) → np.ndarray` Manhattan-radius mask
- [x] `environment/reward.py` — pure `compute_reward(state, action, next_state, cfg) → joint_reward`
- [x] `environment/dec_pomdp.py` — `DecPomdpEnv` with `reset()`, `step(joint_action)`, `global_state()`, **zero gym imports**
- [x] Tests: Manhattan mask correctness; reward at capture/timeout; env smoke 50-step random rollout

**DoD:** env is strict spec match (no gym); same-seed → identical first observation; `global_state()` accessible during training only (warned-on-call from execution path).

---

## Layer 4 — Per-agent recurrent Q-network + soft Polyak update

Commit: `Layer 4: per-agent recurrent Q-net + soft Polyak update`

- [x] `model/init.py` — orthogonal init helpers (carried-over)
- [x] `model/recurrent_q.py` — `QPerAgent(obs_dim, action_dim, hidden_sizes, gru_hidden_size)` with GRU
- [x] `model/soft_update.py` — `polyak_update(target, source, tau)` (carried-over)
- [x] Tests: forward shape, hidden state propagation, gradient flow, save/load roundtrip; Polyak 4-test math battery

**DoD:** GRU hidden propagates across timesteps; Polyak math battery passes (τ=0 / 1 / 0.5 / convergence).

---

## Layer 5 — VDN mixer (∑ additive identity)

Commit: `Layer 5: VDN mixer (sum-decomposition)`

- [x] `model/vdn_mixer.py` — `VDNMixer(n_agents)` — `Q_tot = ∑ Qᵢ`
- [x] Tests:
  - Sum identity: `mixer([q1, q2]) == q1 + q2`
  - n-agents generalisation
  - Gradient flow

**DoD:** VDN is trivially the additive baseline; 100 % coverage.

---

## Layer 6 — QMIX mixer (monotonic hypernetwork)

Commit: `Layer 6: QMIX mixer (monotonic hypernet + abs-weight parametrisation)`

- [x] `model/qmix_mixer.py` — `QMIXMixer(n_agents, state_dim, embed_dim, hidden_dim)`
- [x] Hypernet: state s → mixer weights (with `|·|` to enforce non-negativity → monotonicity)
- [x] Tests:
  - Monotonicity: ∂Q_tot/∂Qᵢ ≥ 0 for all i (verified via random Qᵢ + finite-difference)
  - State-dependence: same Q inputs, different s → different Q_tot
  - Reduces to VDN-style when weights are equal

**DoD:** Monotonicity test passes for 100 random inputs; QMIX paper Section 3.2 IGM constraint verified.

---

## Layer 7 — OLoRA (Orthonormal Low-Rank Adaptation)

Commit: `Layer 7: OLoRA — QR-decomposed orthonormal-init PEFT`

- [x] `model/olora.py` — `OLoRAAdapter(base_layer, rank)` wrapping a `nn.Linear`
- [x] QR decomposition of init weights → orthonormal columns of `A` factor
- [x] Tests:
  - `A` matrix has orthonormal columns: `A.T @ A == I_rank`
  - Reconstruction preserves base layer output at init (zero perturbation)
  - Gradient flow through `A` + `B` factors

**DoD:** OLoRA paper Eq. (3) initialisation verified; PEFT plugs in via a `wrap_with_olora(model)` helper.

---

## Layer 8 — Centralised replay buffer (sequence-aware, masked)

Commit: `Layer 8: centralised replay buffer (variable-length sequences + masks)`

- [x] `memory/centralised_buffer.py` — stores `(s_seq, ō_seq, ā_seq, r̄_seq, s'_seq, ō'_seq, done_seq)`
- [x] Variable sequence length up to `max_seq_len`; pad with mask
- [x] Tests: push wraps at capacity; sample yields correctly-shaped + masked batch

**DoD:** stores full episodes (not transitions); seq + mask shape contract verified.

---

## Layer 9 — Exploration: ε-greedy + schedule

Commit: `Layer 9: ε-greedy exploration + linear schedule`

- [x] `noise/epsilon_greedy.py` — ε-greedy over discrete action space
- [x] `noise/schedule.py` — `LinearEpsilonSchedule(initial, final, decay_steps)` (carried-over from A5's σ schedule)
- [x] Tests: ε=0 → always argmax; ε=1 → uniform random; schedule clamps

**DoD:** discrete-equivalent of A5's noise; same schedule abstraction.

---

## Layer 10 — QMIX update step (the headline math)

Commit: `Layer 10: QMIX update — TD target + per-agent Q + mixer + Polyak`

- [x] `services/qmix_update.py` — `apply_qmix_update(q_nets, mixer, batch, hp) → UpdateDiagnostic`
- [x] Three-network handling: per-agent Qᵢ + Mixer + targets
- [x] **TDD pair — write tests first**:
  - Gradient flows to live Qᵢ + Mixer, NOT to targets
  - One update changes weights
  - Target drift > 0 after one step with τ=0.005
  - Monotonicity preserved post-update

**DoD:** 5-test TDD battery; clean separation of "live" vs "target" networks; matches L10 § 4.2 expressions.

---

## Layer 11 — VDN update + IQL baseline update

Commit: `Layer 11: VDN update + IQL update (baselines)`

- [x] `services/vdn_update.py` — same shape as QMIX but mixer is the sum
- [x] `services/iql_update.py` — **no mixer**; each Qᵢ trains independently against ITS OWN reward; the centralised buffer is used but mixer is absent
- [x] Tests: VDN reduces to per-agent + sum; IQL gradients touch ONLY individual Qᵢ params

**DoD:** all three updaters share the `Updater` protocol; swappable via `marl.algorithm` config key.

---

## Layer 12 — MARL trainer (CTDE end-to-end)

Commit: `Layer 12: MARL trainer — CTDE end-to-end fit loop`

- [x] `services/marl_trainer.py` — `MarlTrainer(env, q_nets, mixer, buffer, schedule, hp).fit(total_episodes)`
- [x] Episode loop: reset → unroll up to 25 steps → push trajectory → sample batch → apply updater → log
- [x] Tests: smoke 200-episode run on 3x3 grid finishes with finite diagnostics

**DoD:** trainer runs end-to-end on a 3x3 grid; pluggable updater (QMIX / VDN / IQL).

---

## Layer 13 — Game runner (6 sub-games per Game)

Commit: `Layer 13: game runner — 6 sub-games per game + GameReport`

- [x] `services/game_runner.py` — `GameRunner(agents, game_cfg).play_one_game() → GameReport`
- [x] Build the JSON per spec § 3.5 (group, students, github_repo, timezone, sub_games[6], totals)
- [x] Tests: deterministic at seed; produces correctly-shaped JSON

**DoD:** running play_one_game from end-to-end produces a JSON that schema-validates against spec § 3.5 example.

---

## Layer 14 — SDK + experiments

Commit: `Layer 14: sdk facade + env_builder + trainers + experiments`

- [x] `sdk/sdk.py::MarlLab(config_path)` — `make_env`, `train`, `evaluate`, `play_one_game`, `run_sweep`, `graphify`
- [x] `sdk/env_builder.py::build_env(cfg, grid_size, algorithm)`
- [x] `sdk/trainers.py::build_trainer(cfg, env)` — picks QMIX / VDN / IQL
- [x] `sdk/experiments.py::ExperimentService` (multi-seed sweeps; same shape as A5)
- [x] Tests: SDK make_env returns a DecPomdpEnv; SDK train returns finite TrainResult

**DoD:** SDK works as the single consumer entry-point; experiments handles all 4 sweep kinds.

---

## Layer 15 — MCP servers (cop + thief) + auth (localhost phase)

Commit: `Layer 15: MCP cop + thief servers + token auth (localhost phase 1)`

- [x] `mcp/protocol.py` — pydantic message schemas (MoveRequest, MoveResponse, HealthResponse)
- [x] `mcp/cop_server.py` — FastMCP server; `@mcp.tool def cop_move(...)`
- [x] `mcp/thief_server.py` — same shape
- [x] `auth/token_registry.py` — load tokens from env; `verify(token)`, `revoke(token)`
- [x] `auth/middleware.py` — `Authorization: Bearer` header check
- [x] Tests: both servers start on localhost; reject without token (401); accept with valid token; revoked token fails

**DoD:** both servers run on localhost simultaneously (different ports); auth works.

---

## Layer 16 — MCP client + game adjudicator over MCP

Commit: `Layer 16: MCP client + game adjudicator drives game via HTTP`

- [x] `mcp/client.py::McpClient(url, token).move(obs, hidden_token) → action`
- [x] `services/game_runner` gets a new constructor variant that drives the game by POSTing to the two MCP servers
- [x] Tests: end-to-end localhost game runs through HTTP; logs to `assets/logs/mcp_session.log`

**DoD:** `marl-lab play --mode mcp-localhost` runs a full 6-sub-game game with both servers up.

---

## Layer 17 — Gmail API + JSON formatter + idempotency

Commit: `Layer 17: Gmail API + JSON formatter + idempotency guard`

- [x] `gmail/formatter.py::build_game_email(report: GameReport) → (subject, body_json)`
- [x] `gmail/sender.py` — common interface; implementations: `AppPasswordSender`, `OAuthSender`, `McpToolSender`
- [x] ADR-010 idempotency: `results/sent_games.json` ledger; same game_id → no-op + warn
- [x] Tests: formatter output schema-validates; idempotency: send twice = one actual SMTP call

**DoD:** `marl-lab report --game results/game_001.json` sends one email (or dry-runs and prints the body).

---

## Layer 18 — CLI (8 subcommands)

Commit: `Layer 18: CLI — train · evaluate · sweep · graphify · gui · serve · play · report`

- [x] `interface/cli/main.py` — Click group
- [x] `interface/cli/commands.py` — bodies for each subcommand
- [x] Tests: each subcommand exits 0 on smoke

**DoD:** `marl-lab --help` lists 8+ subcommands; smoke runs ≤ 200 episodes.

---

## Layer 19 — GUI (Tkinter real-time board)

Commit: `Layer 19: Tkinter GUI — real-time board + score table + replay`

- [x] `interface/gui/main_window.py` — Tabbed window
- [x] `interface/gui/board_tab.py` — live 5x5 board + cop + thief + barriers
- [x] `interface/gui/score_tab.py` — running score table
- [x] `interface/gui/replay_tab.py` — load a `assets/logs/*.log` and replay
- [x] Tests: smoke construction under offscreen Qt-Tk shim (or pytest with `TK_SILENCE_DEPRECATION`)

**DoD:** `marl-lab gui` opens a window; board updates in real time during a game.

---

## Layer 20 — Mini-Graphify port + viz tools

Commit: `Layer 20: Mini-Graphify port + viz tools (plots, GUI capture, log replay)`

- [x] Port `tools/graphify/{walker,emitter,runner}.py` from A5 (rename proximal_lab → marl_lab)
- [x] `tools/viz/plots.py` — `plot_learning_curve`, `plot_loss_curve`, `plot_trajectory_overlay`, `plot_per_agent_q`
- [x] Tests: synthetic 3-module fixture; PNG outputs > 1 KB

**DoD:** `marl-lab graphify` builds `docs/wiki/` with the project's module graph; viz scripts emit at least 6 plots.

---

## Layer 21 — Empirical sweeps

Commit: `Layer 21: sweeps — grid_size + algorithm + observation_radius + ablation_seeds`

- [x] `scripts/run_grid_sweep.py` — 2×2, 3×3, 4×4, 5×5
- [x] `scripts/run_algorithm_sweep.py` — IQL vs VDN vs QMIX
- [x] `scripts/run_radius_sweep.py` — r ∈ {1, 2, 3}
- [x] `scripts/plot_sweep.py` — bar charts with t-CIs (from A5)
- [x] Results → `results/sweeps/*.json`; plots → `assets/plots/*.png`
- [x] Tests: smoke sweeps with reduced timesteps + 1 seed

**DoD:** 3 sweep JSONs + 3 plots; reflection-Q answers grounded in the JSON values.

---

## Layer 22 — Cloud deployment (Prefect/FastMCP) — stub if no creds

Commit: `Layer 22: cloud deployment via Prefect Cloud — full path or documented stub`

- [x] `cloud/prefect_deploy.py` — uses Prefect API key from env
- [x] `cloud/local.py` — always-works localhost runner
- [x] If `PREFECT_API_KEY` missing: print step-by-step guide + skip actual deploy
- [x] README documents BOTH paths
- [x] Tests: stub mode doesn't fail; real-deploy is mocked

**DoD:** can demonstrate the cloud step OR convincingly document why it was skipped (with screenshots / curl examples).

---

## Layer 23 — Reproducibility + drift-test (carried from A5)

Commit: `Layer 23: reproducibility tests + meta-consistency drift-test + extension points`

- [x] `tests/integration/test_reproducibility.py` — same-seed identical diagnostics over a 200-step run
- [ ] `tests/unit/test_doc_drift.py` — *rolled into `tests/integration/test_spec_conformance.py` (5 tests, v1.01) + the CI graphify-drift warning step (v1.06). No standalone file.*
- [x] PLAN.md § 12 extension points already written (Layer 0)
- [x] Tests pass

**DoD:** drift test green; reproducibility test confirms bit-for-bit at the same seed on a small grid.

---

## Layer 24 — Notebook walkthrough (executed)

Commit: `Layer 24: notebook walkthrough — 7-cell guided tour, executed end-to-end`

- [x] `notebooks/marl_lab_walkthrough.ipynb` — 7 cells:
  1. Imports + config
  2. Build env on 3x3 grid
  3. Initialise QMIX agents
  4. Train smoke (200 episodes)
  5. Play one 6-sub-game game; show GameReport JSON
  6. Visualise learning curves for cop + thief
  7. (Optional) plot mixer monotonicity surface
- [x] Execute via `nbconvert --execute`; commit with embedded outputs

**DoD:** notebook renders; outputs embedded; runtime < 5 min.

---

## Layer 25 — Audit + reflection answers + comparison table

Commit: `Layer 25: audit + reflection answers + comparison table + lessons`

- [ ] `docs/AUDIT.md` — *rolled into `CHANGELOG.md` + git tags v1.01/v1.02/v1.05 (TA cycles 1/2/4) + the trail of commits.*
- [ ] `docs/COMPARISON_TABLE.md` — *rolled into `README.md § 7.2` (critical analysis with QMIX vs VDN vs IQL vs QPLEX vs MADDPG) + `docs/PROOFS.md § 5` (the IGM-family summary table).*
- [x] `docs/FAILURE_MODES.md` — known issues + honest disclosures
- [ ] `docs/LESSONS_LEARNED.md` — *rolled into `FAILURE_MODES.md § 3` (the IQL-vs-CTDE empirical finding) + `README.md` "Beyond the spec" rationale.*
- [x] README cross-references each (where the corresponding artifact exists)
- [x] All 3 reflection questions answered with empirical evidence + citations

**DoD:** every PRD § 9 reflection question backed by JSON + plot + paragraph.

---

## Layer 26 — Final docs + EXECUTIVE_SUMMARY + Promptbook + COSTS + V3 tag v1.00

Commit: `Layer 26: final README + EXECUTIVE_SUMMARY + Promptbook + COSTS + sign-off + v1.00 tag`

- [x] Rewrite top-level `README.md` (full version — "Beyond the spec" + § 7 academic analysis embedded)
- [ ] `docs/EXECUTIVE_SUMMARY.md` — *rolled into `README.md` "Beyond the spec" + status badges + the v1.08 tag.*
- [x] `docs/wiki/marl_walkthrough.html` — executed-notebook reproduction artefact (covers REPRODUCIBILITY intent)
- [ ] `docs/PROMPTBOOK.md` — *not produced (methodology trail lives in the git history v1.00 → v1.08 + the 4 TA cycles).*
- [ ] `docs/COSTS.md` — *not produced.*
- [ ] `docs/SLIDE_MAP.md` — *file:line citations rolled into `README.md § 7.1` table + `docs/PROOFS.md` cross-refs.*
- [x] `.github/workflows/assignment6-ci.yml` — green badge (v1.06)
- [ ] **Verify shared with `rmisegal@gmail.com` (MANUAL user step — pending)**
- [x] Tag `marl-lab-v1.00` + push (and v1.01–v1.08 since)

**DoD:** every V3 § 20.9 final-checklist item satisfied; CI badge green; tag pushed.

---

## Beyond-spec extensions (v1.03 → v1.08)

These were not in the original Layer 0 plan above — they came out of the TA-roleplay reviews and post-submission "impress the TA" passes. All 17 are ✅ done; each row links to the tag that introduced it.

### v1.03 — academic depth pass
- [x] **QPLEX mixer** (5th IGM family) — `src/marl_lab/model/qplex_mixer.py` + `qplex_update.py` + 10 tests
- [x] **docs/PROOFS.md** — formal chain-rule IGM derivations for VDN/QMIX/QPLEX
- [x] **Animated sub-game GIF** — `assets/figures/sub_game.gif`
- [x] **4-algorithm tournament** — `assets/figures/tournament.png` + raw CSV
- [x] **Provenance in GameReport** — `src/marl_lab/shared/provenance.py` + 5 tests

### v1.04 — engineering excellence pass
- [x] **Curriculum learning** (Lin 2025) — `services/curriculum.py` + 12 tests with Q-net weight transfer
- [x] **Property-based fuzz tests** — `tests/property/test_env_invariants.py` (7 invariants × 200+ probes)
- [x] **95% measured branch coverage** — `pytest --cov` baseline, table embedded in README
- [x] **Notebook → executed HTML** — `docs/wiki/marl_walkthrough.html` + `scripts/rebuild_notebook.py`

### v1.05 — visual + empirical pass
- [x] **Mermaid system diagram** — README data-flow rendered on GitHub
- [x] **MCP token rotation demo** — 4-stage lifecycle, `assets/logs/token_rotation.log`
- [x] **Bernstein 2002 complexity appendix** — `docs/PROOFS.md § 4`
- [x] **500-episode convergence study** — `assets/figures/long_convergence.png` (honest IQL-competitive finding on 4×4)

### v1.06 — CI pass
- [x] **GitHub Actions** (`.github/workflows/assignment6-ci.yml`) — 2-job pipeline, status badges, HYPOTHESIS_PROFILE=ci

### v1.07 — Lin-2025 verification pass
- [x] **Scale convergence (5×5/6×6/7×7)** — `assets/figures/ctde_advantage_vs_grid.png`; hypothesis confirmed: QPLEX dominates 6×6 by +1.01 over IQL

### v1.08 — algorithmic completeness pass
- [x] **MADDPG-discrete** (true POSG learner; per-agent centralised critic + per-agent reward) — `src/marl_lab/model/maddpg_critic.py` + `services/maddpg_update.py` + 13 tests. 5th algorithm.
- [x] **Docker** — `Dockerfile` + `.dockerignore`, smoke-tested, zero-setup playability

### Manual user steps (still pending)
- [ ] Fill `submission.group_code` in `configs/setup.yaml` with the real 8-char code
- [ ] Fill `submission.students[0].id` with the real student ID
- [ ] Share GitHub repo with `rmisegal@gmail.com` (read access)
- [ ] (Optional) Find a partner for the spec § 9 inter-group bonus (10 pts)
