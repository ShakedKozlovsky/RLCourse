# TODO — Layered Implementation Plan (Assignment 6 — MARL)

> Each layer = one commit. **Definition of Done** is explicit per layer: code + tests + docs updated. Status uses `[ ]` pending, `[~]` in progress, `[x]` done.

Reference: [`PRD.md`](PRD.md), [`PLAN.md`](PLAN.md).

---

## Layer 0 — Scaffold + planning docs

- [x] Directory tree under `assignment6/`
- [x] `docs/PRD.md` written
- [x] `docs/PLAN.md` written
- [x] `docs/TODO.md` (this file)
- [ ] Per-mechanism PRDs: `PRD_dec_pomdp.md`, `PRD_game.md`, `PRD_ctde.md`, `PRD_olora.md`, `PRD_mcp.md`, `PRD_gmail.md`, `PRD_partial_observation.md`, `PRD_iql_baseline.md`
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

- [ ] `shared/config.py` — YAML loader with `version` check + dotted access (`cfg.get("marl.tau")`)
- [ ] `shared/logger.py` — stdlib factory (no print in library)
- [ ] `shared/seed.py` — `set_global_seed(int)` for Python + NumPy + PyTorch
- [ ] `shared/types.py` — `Obs`, `JointAction`, `Transition`, `EpisodeSequence`, `SubGameResult`, `GameReport`, `StepDiagnostic`, `TrainResult`
- [ ] Tests: YAML loads, version mismatch raises, dotted access works, types frozen

**DoD:** ConfigManager passes 6+ tests; version check identical to A4/A5 pattern.

---

## Layer 2 — Game core (board, moves, win adjudication, barriers)

Commit: `Layer 2: game core — board + moves + win adjudication + barriers`

- [ ] `game/board.py` — `Board(grid_size, cop_pos, thief_pos, barriers, step)` dataclass
- [ ] `game/actions.py` — `Action` enum (UP, DOWN, LEFT, RIGHT, STAY, PLACE_BARRIER)
- [ ] `game/moves.py` — `MoveDynamics.apply(board, joint_action) → new_board, info`
- [ ] `game/win.py` — `WinAdjudicator.check(board) → "cop"|"thief"|None`
- [ ] `game/barriers.py` — `BarrierPlacement` (validity, max-5 cap)
- [ ] `game/sub_game.py` — `SubGameRunner` (25-move cap)
- [ ] `game/game.py` — `Game` (6 sub-games), accumulates `GameReport`
- [ ] Tests: move validity on 5x5; collisions cap at walls/barriers; barrier placement counter; capture-on-overlap; 6-sub-game accounting

**DoD:** 100 % branch coverage in `game/`; pure functions throughout; 4-test scoring battery.

---

## Layer 3 — Partial observation + Dec-POMDP env + reward

Commit: `Layer 3: sensor + dec_pomdp env + reward functions`

- [ ] `sensor/partial_observation.py` — `observe(global_state, agent_id, radius) → np.ndarray` Manhattan-radius mask
- [ ] `environment/reward.py` — pure `compute_reward(state, action, next_state, cfg) → joint_reward`
- [ ] `environment/dec_pomdp.py` — `DecPomdpEnv` with `reset()`, `step(joint_action)`, `global_state()`, **zero gym imports**
- [ ] Tests: Manhattan mask correctness; reward at capture/timeout; env smoke 50-step random rollout

**DoD:** env is strict spec match (no gym); same-seed → identical first observation; `global_state()` accessible during training only (warned-on-call from execution path).

---

## Layer 4 — Per-agent recurrent Q-network + soft Polyak update

Commit: `Layer 4: per-agent recurrent Q-net + soft Polyak update`

- [ ] `model/init.py` — orthogonal init helpers (carried-over)
- [ ] `model/recurrent_q.py` — `QPerAgent(obs_dim, action_dim, hidden_sizes, gru_hidden_size)` with GRU
- [ ] `model/soft_update.py` — `polyak_update(target, source, tau)` (carried-over)
- [ ] Tests: forward shape, hidden state propagation, gradient flow, save/load roundtrip; Polyak 4-test math battery

**DoD:** GRU hidden propagates across timesteps; Polyak math battery passes (τ=0 / 1 / 0.5 / convergence).

---

## Layer 5 — VDN mixer (∑ additive identity)

Commit: `Layer 5: VDN mixer (sum-decomposition)`

- [ ] `model/vdn_mixer.py` — `VDNMixer(n_agents)` — `Q_tot = ∑ Qᵢ`
- [ ] Tests:
  - Sum identity: `mixer([q1, q2]) == q1 + q2`
  - n-agents generalisation
  - Gradient flow

**DoD:** VDN is trivially the additive baseline; 100 % coverage.

---

## Layer 6 — QMIX mixer (monotonic hypernetwork)

Commit: `Layer 6: QMIX mixer (monotonic hypernet + abs-weight parametrisation)`

- [ ] `model/qmix_mixer.py` — `QMIXMixer(n_agents, state_dim, embed_dim, hidden_dim)`
- [ ] Hypernet: state s → mixer weights (with `|·|` to enforce non-negativity → monotonicity)
- [ ] Tests:
  - Monotonicity: ∂Q_tot/∂Qᵢ ≥ 0 for all i (verified via random Qᵢ + finite-difference)
  - State-dependence: same Q inputs, different s → different Q_tot
  - Reduces to VDN-style when weights are equal

**DoD:** Monotonicity test passes for 100 random inputs; QMIX paper Section 3.2 IGM constraint verified.

---

## Layer 7 — OLoRA (Orthonormal Low-Rank Adaptation)

Commit: `Layer 7: OLoRA — QR-decomposed orthonormal-init PEFT`

- [ ] `model/olora.py` — `OLoRAAdapter(base_layer, rank)` wrapping a `nn.Linear`
- [ ] QR decomposition of init weights → orthonormal columns of `A` factor
- [ ] Tests:
  - `A` matrix has orthonormal columns: `A.T @ A == I_rank`
  - Reconstruction preserves base layer output at init (zero perturbation)
  - Gradient flow through `A` + `B` factors

**DoD:** OLoRA paper Eq. (3) initialisation verified; PEFT plugs in via a `wrap_with_olora(model)` helper.

---

## Layer 8 — Centralised replay buffer (sequence-aware, masked)

Commit: `Layer 8: centralised replay buffer (variable-length sequences + masks)`

- [ ] `memory/centralised_buffer.py` — stores `(s_seq, ō_seq, ā_seq, r̄_seq, s'_seq, ō'_seq, done_seq)`
- [ ] Variable sequence length up to `max_seq_len`; pad with mask
- [ ] Tests: push wraps at capacity; sample yields correctly-shaped + masked batch

**DoD:** stores full episodes (not transitions); seq + mask shape contract verified.

---

## Layer 9 — Exploration: ε-greedy + schedule

Commit: `Layer 9: ε-greedy exploration + linear schedule`

- [ ] `noise/epsilon_greedy.py` — ε-greedy over discrete action space
- [ ] `noise/schedule.py` — `LinearEpsilonSchedule(initial, final, decay_steps)` (carried-over from A5's σ schedule)
- [ ] Tests: ε=0 → always argmax; ε=1 → uniform random; schedule clamps

**DoD:** discrete-equivalent of A5's noise; same schedule abstraction.

---

## Layer 10 — QMIX update step (the headline math)

Commit: `Layer 10: QMIX update — TD target + per-agent Q + mixer + Polyak`

- [ ] `services/qmix_update.py` — `apply_qmix_update(q_nets, mixer, batch, hp) → UpdateDiagnostic`
- [ ] Three-network handling: per-agent Qᵢ + Mixer + targets
- [ ] **TDD pair — write tests first**:
  - Gradient flows to live Qᵢ + Mixer, NOT to targets
  - One update changes weights
  - Target drift > 0 after one step with τ=0.005
  - Monotonicity preserved post-update

**DoD:** 5-test TDD battery; clean separation of "live" vs "target" networks; matches L10 § 4.2 expressions.

---

## Layer 11 — VDN update + IQL baseline update

Commit: `Layer 11: VDN update + IQL update (baselines)`

- [ ] `services/vdn_update.py` — same shape as QMIX but mixer is the sum
- [ ] `services/iql_update.py` — **no mixer**; each Qᵢ trains independently against ITS OWN reward; the centralised buffer is used but mixer is absent
- [ ] Tests: VDN reduces to per-agent + sum; IQL gradients touch ONLY individual Qᵢ params

**DoD:** all three updaters share the `Updater` protocol; swappable via `marl.algorithm` config key.

---

## Layer 12 — MARL trainer (CTDE end-to-end)

Commit: `Layer 12: MARL trainer — CTDE end-to-end fit loop`

- [ ] `services/marl_trainer.py` — `MarlTrainer(env, q_nets, mixer, buffer, schedule, hp).fit(total_episodes)`
- [ ] Episode loop: reset → unroll up to 25 steps → push trajectory → sample batch → apply updater → log
- [ ] Tests: smoke 200-episode run on 3x3 grid finishes with finite diagnostics

**DoD:** trainer runs end-to-end on a 3x3 grid; pluggable updater (QMIX / VDN / IQL).

---

## Layer 13 — Game runner (6 sub-games per Game)

Commit: `Layer 13: game runner — 6 sub-games per game + GameReport`

- [ ] `services/game_runner.py` — `GameRunner(agents, game_cfg).play_one_game() → GameReport`
- [ ] Build the JSON per spec § 3.5 (group, students, github_repo, timezone, sub_games[6], totals)
- [ ] Tests: deterministic at seed; produces correctly-shaped JSON

**DoD:** running play_one_game from end-to-end produces a JSON that schema-validates against spec § 3.5 example.

---

## Layer 14 — SDK + experiments

Commit: `Layer 14: sdk facade + env_builder + trainers + experiments`

- [ ] `sdk/sdk.py::MarlLab(config_path)` — `make_env`, `train`, `evaluate`, `play_one_game`, `run_sweep`, `graphify`
- [ ] `sdk/env_builder.py::build_env(cfg, grid_size, algorithm)`
- [ ] `sdk/trainers.py::build_trainer(cfg, env)` — picks QMIX / VDN / IQL
- [ ] `sdk/experiments.py::ExperimentService` (multi-seed sweeps; same shape as A5)
- [ ] Tests: SDK make_env returns a DecPomdpEnv; SDK train returns finite TrainResult

**DoD:** SDK works as the single consumer entry-point; experiments handles all 4 sweep kinds.

---

## Layer 15 — MCP servers (cop + thief) + auth (localhost phase)

Commit: `Layer 15: MCP cop + thief servers + token auth (localhost phase 1)`

- [ ] `mcp/protocol.py` — pydantic message schemas (MoveRequest, MoveResponse, HealthResponse)
- [ ] `mcp/cop_server.py` — FastMCP server; `@mcp.tool def cop_move(...)`
- [ ] `mcp/thief_server.py` — same shape
- [ ] `auth/token_registry.py` — load tokens from env; `verify(token)`, `revoke(token)`
- [ ] `auth/middleware.py` — `Authorization: Bearer` header check
- [ ] Tests: both servers start on localhost; reject without token (401); accept with valid token; revoked token fails

**DoD:** both servers run on localhost simultaneously (different ports); auth works.

---

## Layer 16 — MCP client + game adjudicator over MCP

Commit: `Layer 16: MCP client + game adjudicator drives game via HTTP`

- [ ] `mcp/client.py::McpClient(url, token).move(obs, hidden_token) → action`
- [ ] `services/game_runner` gets a new constructor variant that drives the game by POSTing to the two MCP servers
- [ ] Tests: end-to-end localhost game runs through HTTP; logs to `assets/logs/mcp_session.log`

**DoD:** `marl-lab play --mode mcp-localhost` runs a full 6-sub-game game with both servers up.

---

## Layer 17 — Gmail API + JSON formatter + idempotency

Commit: `Layer 17: Gmail API + JSON formatter + idempotency guard`

- [ ] `gmail/formatter.py::build_game_email(report: GameReport) → (subject, body_json)`
- [ ] `gmail/sender.py` — common interface; implementations: `AppPasswordSender`, `OAuthSender`, `McpToolSender`
- [ ] ADR-010 idempotency: `results/sent_games.json` ledger; same game_id → no-op + warn
- [ ] Tests: formatter output schema-validates; idempotency: send twice = one actual SMTP call

**DoD:** `marl-lab report --game results/game_001.json` sends one email (or dry-runs and prints the body).

---

## Layer 18 — CLI (8 subcommands)

Commit: `Layer 18: CLI — train · evaluate · sweep · graphify · gui · serve · play · report`

- [ ] `interface/cli/main.py` — Click group
- [ ] `interface/cli/commands.py` — bodies for each subcommand
- [ ] Tests: each subcommand exits 0 on smoke

**DoD:** `marl-lab --help` lists 8+ subcommands; smoke runs ≤ 200 episodes.

---

## Layer 19 — GUI (Tkinter real-time board)

Commit: `Layer 19: Tkinter GUI — real-time board + score table + replay`

- [ ] `interface/gui/main_window.py` — Tabbed window
- [ ] `interface/gui/board_tab.py` — live 5x5 board + cop + thief + barriers
- [ ] `interface/gui/score_tab.py` — running score table
- [ ] `interface/gui/replay_tab.py` — load a `assets/logs/*.log` and replay
- [ ] Tests: smoke construction under offscreen Qt-Tk shim (or pytest with `TK_SILENCE_DEPRECATION`)

**DoD:** `marl-lab gui` opens a window; board updates in real time during a game.

---

## Layer 20 — Mini-Graphify port + viz tools

Commit: `Layer 20: Mini-Graphify port + viz tools (plots, GUI capture, log replay)`

- [ ] Port `tools/graphify/{walker,emitter,runner}.py` from A5 (rename proximal_lab → marl_lab)
- [ ] `tools/viz/plots.py` — `plot_learning_curve`, `plot_loss_curve`, `plot_trajectory_overlay`, `plot_per_agent_q`
- [ ] Tests: synthetic 3-module fixture; PNG outputs > 1 KB

**DoD:** `marl-lab graphify` builds `docs/wiki/` with the project's module graph; viz scripts emit at least 6 plots.

---

## Layer 21 — Empirical sweeps

Commit: `Layer 21: sweeps — grid_size + algorithm + observation_radius + ablation_seeds`

- [ ] `scripts/run_grid_sweep.py` — 2×2, 3×3, 4×4, 5×5
- [ ] `scripts/run_algorithm_sweep.py` — IQL vs VDN vs QMIX
- [ ] `scripts/run_radius_sweep.py` — r ∈ {1, 2, 3}
- [ ] `scripts/plot_sweep.py` — bar charts with t-CIs (from A5)
- [ ] Results → `results/sweeps/*.json`; plots → `assets/plots/*.png`
- [ ] Tests: smoke sweeps with reduced timesteps + 1 seed

**DoD:** 3 sweep JSONs + 3 plots; reflection-Q answers grounded in the JSON values.

---

## Layer 22 — Cloud deployment (Prefect/FastMCP) — stub if no creds

Commit: `Layer 22: cloud deployment via Prefect Cloud — full path or documented stub`

- [ ] `cloud/prefect_deploy.py` — uses Prefect API key from env
- [ ] `cloud/local.py` — always-works localhost runner
- [ ] If `PREFECT_API_KEY` missing: print step-by-step guide + skip actual deploy
- [ ] README documents BOTH paths
- [ ] Tests: stub mode doesn't fail; real-deploy is mocked

**DoD:** can demonstrate the cloud step OR convincingly document why it was skipped (with screenshots / curl examples).

---

## Layer 23 — Reproducibility + drift-test (carried from A5)

Commit: `Layer 23: reproducibility tests + meta-consistency drift-test + extension points`

- [ ] `tests/integration/test_reproducibility.py` — same-seed identical diagnostics over a 200-step run
- [ ] `tests/unit/test_doc_drift.py` — README + EXEC_SUMMARY layer-count matches `docs/TODO.md` (from A5 v1.26)
- [ ] PLAN.md § 12 extension points already written (Layer 0)
- [ ] Tests pass

**DoD:** drift test green; reproducibility test confirms bit-for-bit at the same seed on a small grid.

---

## Layer 24 — Notebook walkthrough (executed)

Commit: `Layer 24: notebook walkthrough — 7-cell guided tour, executed end-to-end`

- [ ] `notebooks/marl_lab_walkthrough.ipynb` — 7 cells:
  1. Imports + config
  2. Build env on 3x3 grid
  3. Initialise QMIX agents
  4. Train smoke (200 episodes)
  5. Play one 6-sub-game game; show GameReport JSON
  6. Visualise learning curves for cop + thief
  7. (Optional) plot mixer monotonicity surface
- [ ] Execute via `nbconvert --execute`; commit with embedded outputs

**DoD:** notebook renders; outputs embedded; runtime < 5 min.

---

## Layer 25 — Audit + reflection answers + comparison table

Commit: `Layer 25: audit + reflection answers + comparison table + lessons`

- [ ] `docs/AUDIT.md` — multi-cycle adversarial-review history (from A5 pattern, starting at cycle 1 for this assignment)
- [ ] `docs/COMPARISON_TABLE.md` — MARL vs single-agent DDPG / DQN / PPO; IQL vs VDN vs QMIX; pros/cons + citations
- [ ] `docs/FAILURE_MODES.md` — known issues + honest disclosures
- [ ] `docs/LESSONS_LEARNED.md` — meta lessons
- [ ] README cross-references each
- [ ] All 3 reflection questions answered with empirical evidence + citations

**DoD:** every PRD § 9 reflection question backed by JSON + plot + paragraph.

---

## Layer 26 — Final docs + EXECUTIVE_SUMMARY + Promptbook + COSTS + V3 tag v1.00

Commit: `Layer 26: final README + EXECUTIVE_SUMMARY + Promptbook + COSTS + sign-off + v1.00 tag`

- [ ] Rewrite top-level `README.md` (full version)
- [ ] `docs/EXECUTIVE_SUMMARY.md` — 1-pager for grader
- [ ] `docs/REPRODUCIBILITY.md` — exact replay commands
- [ ] `docs/PROMPTBOOK.md` — methodology log (carries over A5 § 15 iterative-adversarial-review pattern)
- [ ] `docs/COSTS.md` — token cost analysis
- [ ] `docs/SLIDE_MAP.md` — L10 slide → file:line citations
- [ ] `.github/workflows/assignment6-ci.yml` — green badge
- [ ] Verify shared with `rmisegal@gmail.com` (manual user step)
- [ ] Tag `assignment6-v1.00` + push

**DoD:** every V3 § 20.9 final-checklist item satisfied; CI badge green; tag pushed.
