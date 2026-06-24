# PLAN — Layered architecture, ADRs, pseudocode

> Companion to [PRD.md](PRD.md). Defines **how** we will build `marl_lab` end-to-end. V3 § 2.4 layered design + the prior assignments' `interface → sdk → services → {…} → shared` convention, extended for the new cloud/MCP pillars.

## 1. Architecture diagram (text)

```
                       ┌──────────────────────────────────────────────────┐
                       │                  interface/                       │
                       │  cli/  (Click)   gui/ (Tkinter — real-time board) │
                       └──────────────────────┬───────────────────────────┘
                                              │
                       ┌──────────────────────▼───────────────────────────┐
                       │                    sdk/                           │
                       │  MarlLab facade · env_builder · trainers · exps   │
                       └──────────────────────┬───────────────────────────┘
                                              │
       ┌──────────────────────────────────────▼────────────────────────────────────┐
       │                                services/                                   │
       │  marl_trainer · qmix_update · vdn_update · iql_update · ctde_orchestrator  │
       │  evaluation · comparison · game_runner (runs 6 sub-games per "game")       │
       └────┬──────────────────┬──────────────────┬──────────────────┬─────────────┘
            │                  │                  │                  │
   ┌────────▼────────┐  ┌──────▼─────┐    ┌──────▼─────┐    ┌────────▼────────┐
   │   environment/  │  │   model/   │    │   memory/  │    │    noise/       │
   │  dec_pomdp +    │  │ q_per_agent│    │ centralised│    │  ε-greedy +     │
   │  reward + grid  │  │ vdn_mixer  │    │ replay buf │    │  schedule       │
   │  obs masking    │  │ qmix_mixer │    │ (CTDE)     │    │  (boltzmann?)   │
   └────────┬────────┘  │ recurrent_q│    └────────────┘    └─────────────────┘
            │           │ olora      │
   ┌────────▼────────┐  │ soft_update│
   │   sensor/       │  └────────────┘
   │ partial-obs +   │
   │ manhattan radius│
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │      game/      │  game state · move dynamics · win conditions
   │  board · moves  │  barriers · scoring (Table 1) · sub-game accounting
   │  barriers       │  game = 6 sub-games (one Gmail report per game)
   └─────────────────┘
                                              │
                       ┌──────────────────────▼───────────────────────────┐
                       │     mcp/     │  gmail/  │  auth/  │  cloud/      │
                       │  cop server  │ Gmail API│ revocab │ Prefect      │
                       │  thief server│ + JSON   │ tokens  │ deploy       │
                       │  client/HTTP │ formatter│         │ helpers      │
                       └──────────────┴──────────┴─────────┴──────────────┘
                                              │
                       ┌──────────────────────▼───────────────────────────┐
                       │   data/   │   shared/ (config, logger, seed, types, version)   │
                       │ heuristic │   tools/graphify · tools/viz                       │
                       │ policies  │                                                    │
                       └───────────┴────────────────────────────────────────────────────┘
```

## 2. Package map (one paragraph each)

- **`shared/`** — config (YAML loader + version check), logger, seed (Python + NumPy + PyTorch + RNG for MCP), types (`Obs`, `JointAction`, `Transition`, `GameReport`, `SubGameResult`), version.
- **`data/`** — heuristic baseline policies (e.g., greedy-cop, evasive-thief) for warm-start dataset construction; pre-trained backbone bookkeeping.
- **`game/`** — pure game logic: `Board`, `MoveDynamics` (UP/DOWN/LEFT/RIGHT/STAY, optional PLACE_BARRIER for cop), win/loss adjudication, `SubGame` runner, `Game` (= 6 sub-games), `BarrierPlacement` (max 5).
- **`sensor/`** — `partial_observation(state, agent_id, radius) → local_view` (Manhattan-radius mask of the global state).
- **`environment/`** — `DecPomdpEnv` Gym-shape (NOT gym-subclass) wrapping `game/` + `sensor/`; `reward.py` (cop +20/-5; thief +10/-5; step penalties for shaping).
- **`model/`** — per-agent recurrent Q-network (GRU on observation history), VDN sum-mixer, QMIX monotonic mixer with hypernetwork, OLoRA QR-decomposition layer, soft Polyak update, init helpers.
- **`memory/`** — centralised replay buffer of full episode tuples (variable-length sequences, handled with masks).
- **`noise/`** — ε-greedy schedule + Boltzmann softmax alternative.
- **`services/`** — `marl_trainer` (CTDE training loop), `qmix_update` + `vdn_update` + `iql_update` (the three swappable updaters), `ctde_orchestrator` (manages cop + thief networks during training), `evaluation_service`, `comparison_service`, `game_runner` (executes one full 6-sub-game game + builds the report).
- **`sdk/`** — `MarlLab` facade, `env_builder`, `trainers`, `experiments` (multi-seed sweeps over grid_size / algorithm / observation_radius).
- **`mcp/`** — `cop_server` and `thief_server` (FastMCP); shared `protocol.py` (the message schemas); `client.py` (HTTP wrapper); `tools.py` (the `@mcp.tool` functions exposed by each server).
- **`gmail/`** — `formatter.py` (builds the JSON per § 3.5 of the spec); `sender.py` (App-Password SMTP, OAuth, or MCP-tool implementations behind one interface).
- **`auth/`** — token registry + revoke + middleware for the MCP servers.
- **`cloud/`** — Prefect deployment + workspace helpers (stub if no creds available).
- **`interface/cli/`** — Click group: `train`, `evaluate`, `sweep`, `graphify`, `gui`, `serve` (run an MCP server), `play` (orchestrate one game), `report` (email the JSON), `download-data`.
- **`interface/gui/`** — Tkinter: real-time board, sub-game counter, score table, replay-from-log.
- **`tools/graphify/`** — port from A5: AST → Obsidian Vault.
- **`tools/viz/`** — plots + GUI capture + log replay.

## 3. Class diagram (text)

```
ConfigManager(path)                          → frozen dict + dotted access
SubGameResult (dataclass) = (id, start, end, moves, winner, scores)
GameReport (dataclass) = (group, students, repo, tz, sub_games, totals)

Board(grid_size, barriers, cop_pos, thief_pos, step)
MoveDynamics.apply(board, joint_action) → new_board, info
WinAdjudicator.check(board) → "cop" | "thief" | None
BarrierPlacement.try_place(board, pos) → bool

Sensor.observe(state, agent_id, radius) → np.ndarray (Manhattan mask)

DecPomdpEnv(game_cfg, sensor_cfg, reward_cfg)
    .reset(seed) → joint_obs
    .step(joint_action) → (joint_obs, joint_reward, done, info)
    .global_state() → s   # only available during TRAINING

QPerAgent(obs_dim, action_dim, hidden, gru_hidden)  nn.Module
    forward(obs_seq, hidden_state) → q_values, new_hidden_state

VDNMixer(n_agents)            Q_tot = Σ Qᵢ
QMIXMixer(n_agents, state_dim, embed_dim)
    Hypernetwork weights conditioned on global state s
    Monotonicity enforced via |W| (abs-value) parametrisation

OLoRAAdapter(base_layer, rank)  Orthonormal Low-Rank Adaptation (QR-init)

CentralisedReplayBuffer(capacity, max_seq_len, obs_dim, n_agents)
    push(episode) → None
    sample(batch_size) → padded sequence batch + masks

CtdeTrainer(env, q_nets, mixer, buffer, hp)
    .train_episode(seed) → metrics
    .fit(total_episodes) → TrainResult

IqlTrainer(env, q_nets, buffer, hp)         # baseline (NO mixer)

GameRunner(env, agents, n_sub_games, max_moves)
    .play_one_game() → GameReport

CopMcpServer(host, port, auth, env_handle)   # FastMCP
ThiefMcpServer(host, port, auth, env_handle)
McpClient(server_url, token).move(state) → action

GmailSender(from_addr, app_password).send(report: GameReport) → message_id

MarlLab(config_path)
    .make_env(grid_size?, algorithm?) → DecPomdpEnv
    .train(algorithm) → TrainResult
    .evaluate(net) → EvalReport
    .run_sweep(kind) → SweepReport
    .play_one_game() → GameReport
    .graphify() → docs/wiki/
```

## 4. Training pseudocode (CTDE — QMIX path)

```text
Initialise:
    For each agent i:
        Qᵢ_net  ← QPerAgent(obs_dim_i, action_dim_i, hidden, gru_hidden)
        Qᵢ_tgt  ← deepcopy(Qᵢ_net)   ; freeze grads
    Mixer        ← QMIXMixer(n=2, state_dim=|S|)
    Mixer_tgt    ← deepcopy(Mixer)   ; freeze grads
    Buffer B     ← CentralisedReplayBuffer(capacity, max_seq_len)
    epsilon      ← 1.0   ; schedule = LinearEpsilonSchedule(1.0 → 0.05)

For episode = 1..E:
    s = env.reset(seed_e)
    For agent i: hᵢ = zero hidden
    trajectory = []
    For t = 1..max_moves (per sub-game):
        oᵢ_t = env.partial_observation(s, agent=i)
        with ε(t)-greedy:
            aᵢ_t = argmax_a Qᵢ_net(oᵢ_t, hᵢ) (or random with prob ε)
        joint_action = (a_cop, a_thief)
        s', r̄, done = env.step(joint_action)
        trajectory.append((s, ō_t, ā_t, r̄_t, s', ō'_t, done))
        s = s'
        if done: break
    B.push(trajectory)
    
    if B.size > warmup:
        seq_batch = B.sample(batch_size)
        # Per-agent forward over sequence:
        for i in agents:
            qᵢ_chosen[i]    = Qᵢ_net(o_seq_i, h_init_i)[ā_seq_i]
            qᵢ_max_next[i]  = max_a' Qᵢ_tgt(o'_seq_i, h_init_i)
        # Mix with global state:
        Q_tot_chosen  = QMIXMixer(qᵢ_chosen, s_seq)
        Q_tot_max_next = QMIXMixer_tgt(qᵢ_max_next, s'_seq)
        y = r̄_seq + γ · (1 - done_seq) · Q_tot_max_next
        loss = MSE(Q_tot_chosen, y) * mask
        loss.backward(); clip; step

        Polyak: Qᵢ_tgt ← τ·Qᵢ_net + (1-τ)·Qᵢ_tgt
                Mixer_tgt ← τ·Mixer + (1-τ)·Mixer_tgt

    epsilon = schedule.at(step)
    if episode % log_interval == 0: log diagnostics
```

## 5. Decentralised-execution pseudocode (the cloud-MCP path)

```text
# COP server (running at https://...prefect.cloud/cop) exposes:
@mcp.tool
def cop_move(global_or_local_obs, history_state_token) → {action, next_state_token}:
    """The COP agent's policy. Only sees its own local observation."""
    obs = obs_from_request_safely(global_or_local_obs)  # rejects full-state queries
    h = retrieve_or_init_hidden(history_state_token)
    with torch.no_grad():
        q = cop_q_net(obs, h)
    action = q.argmax().item()
    new_token = store(new_hidden)
    return {"action": ACTIONS[action], "next_state_token": new_token}

# Same shape for THIEF server.

# A GAME ADJUDICATOR (running locally) drives the round:
game = Game(grid_size=5, ...)
for sub_game in 1..6:
    state = game.reset()
    cop_token, thief_token = None, None
    for t in 1..25:
        cop_obs = game.partial_observation(state, agent="cop")
        cop_resp = http_post(COP_URL, {obs: cop_obs, token: cop_token}, headers={Authorization: BEARER ...})
        thief_obs = game.partial_observation(state, agent="thief")
        thief_resp = http_post(THIEF_URL, {obs: thief_obs, token: thief_token}, ...)
        state, _, done = game.step({cop: cop_resp.action, thief: thief_resp.action})
        cop_token, thief_token = cop_resp.next_state_token, thief_resp.next_state_token
        if done: break
    record_sub_game_result(...)

report = GameReport(...)
gmail_sender.send(report)
```

## 6. Configuration schema (`configs/setup.yaml`)

```yaml
version: "1.00"                 # checked against shared/version.py
seed: <int>
device: cpu | cuda

game:                            # § 3 spec
  grid_size: [H, W]
  max_moves: int                 # default 25
  num_games: int                 # default 6
  max_barriers: int              # default 5
  enable_barriers: bool
  observation_radius: int        # Manhattan radius

scoring:                          # § 3.4 spec Table 1
  cop_win, thief_win, cop_loss, thief_loss

marl:
  algorithm: "qmix" | "vdn" | "iql"
  gamma, tau, actor_lr, critic_lr, mixer_lr
  batch_size, replay_capacity, warmup_steps
  max_grad_norm, hidden_sizes, use_rnn, rnn_hidden_size
  use_olora, olora_rank

exploration:
  kind: "epsilon_greedy"
  epsilon_initial, epsilon_final, decay_steps

training:
  total_episodes, log_interval, eval_interval, n_eval_episodes

experiments:
  grid_size_sweep, algorithm_sweep, observation_radius_sweep, ablation_seeds

mcp:
  cop_host, cop_port, thief_host, thief_port
  framework, cloud_provider, auth_required, request_timeout_s

gmail:
  report_to, subject_prefix, send_mode

submission:
  group_code, group_name, students, github_repo, timezone

paths:
  results_dir, assets_dir, checkpoints_dir, data_dir, logs_dir, wiki_dir
```

## 7. Architectural Decision Records (ADRs)

### ADR-001 — Custom Dec-POMDP env, NO Gymnasium

Spec § 5.1 demands a custom env. We mirror the Gym API shape but import zero `gym`/`gymnasium` packages. (Same rule as A5 — verified by a CI grep.)

### ADR-002 — POSG framing in the analysis chapter, Dec-POMDP framing in implementation

L10 § 2.1 explicitly notes that adversarial reward = POSG, not pure Dec-POMDP. We implement the CTDE/VDN/QMIX machinery (cooperative-Dec-POMDP tooling) as a *practical approximation* and disclose the gap honestly in `docs/FAILURE_MODES.md`. The PRD § 3 mathematical tuple is correct **per agent** (each has its own R, so it's the POSG ⟨I, S, {Aᵢ}, {Oᵢ}, P, Ω, {Rᵢ}, γ⟩ from L10 equation 3).

### ADR-003 — YAML config, not JSON

Spec § 3.6 allows either; YAML matches the spec's example exactly. We commit BOTH a `setup.yaml` (canonical) and a `setup.json` (auto-generated by a small script for callers who prefer JSON).

### ADR-004 — GRU not LSTM for the recurrent Q

L10 § 5 mentions both. GRU has fewer params and trains faster on the short (≤ 25-step) horizons here. LSTM left as an ablation in the experiments sweep.

### ADR-005 — Two separate MCP servers, NOT one with two endpoints

Spec § 5.3 is explicit. Each agent is "autonomous" — separate process, separate token. Means the game adjudicator coordinates **two** HTTP clients in parallel (uses `asyncio.gather` for speed).

### ADR-006 — FastMCP as the MCP framework

Spec § 5.3 recommends it; the Anthropic spec [11] matches; runs naturally on Prefect Cloud per the spec's appendix § 8.

### ADR-007 — Token authentication via a simple `Authorization: Bearer <tok>` header

V3 § 5.3 mentions API gateway / rate limiting; for this assignment we implement the simple version (token list in env var) + a `revoke(token)` admin endpoint guarded by an admin-only token. Production would use OAuth + JWT — documented as extension.

### ADR-008 — Centralised replay buffer stores FULL episode tuples, not per-step transitions

QMIX target computation needs the global state at every timestep of the sequence + the hidden states of the per-agent GRUs. Per-step transition buffer (as in A5's DDPG) would lose the recurrent context. ADR is to store `(s_seq, ō_seq, ā_seq, r̄_seq, s'_seq, ō'_seq, done_seq)` as variable-length arrays with masks.

### ADR-009 — `experiments.py` lives under `sdk/`, not `services/`

Carried-over lesson from A4 Layer 17 / A5 ADR-007: experiments imports SDK, so it lives above it in the layering arrow.

### ADR-010 — Gmail email is sent at most ONCE per game (the spec is explicit)

Spec § 3.5 explicit. We add an idempotency guard: `gmail/sender.py` checks a local `results/sent_games.json` ledger before sending. Re-sending the same game's report (same `game_id`) is a no-op with a warning. Prevents accidental duplicate submissions during testing.

## 8. Test plan (high level)

| Module | Tests |
|---|---|
| `shared/config` | YAML round-trip, version-mismatch raises |
| `game/board` | grid init, move validity, win adjudication, barrier placement (≤ max) |
| `game/sub_game` | 25-move cap, capture-on-overlap, scoring |
| `game/game` | exactly 6 sub-games per game; report JSON shape |
| `environment/dec_pomdp` | reset → obs shape; step contract |
| `sensor/partial_observation` | Manhattan radius mask correctness on a 5x5 + 8 cells of each side |
| `model/qmix_mixer` | monotonicity assertion: ∂Q_tot/∂Qᵢ ≥ 0 (sign of mixer weights ≥ 0) |
| `model/vdn_mixer` | Σ identity sanity |
| `model/olora` | QR orthonormality of init A; reconstruction preservation |
| `memory/centralised_buffer` | sequence padding + mask correctness |
| `services/qmix_update` | gradient flows; one update changes weights; target_drift > 0 |
| `services/iql_update` | baseline contrast; no mixer |
| `services/game_runner` | full 6-sub-game integration, deterministic at seed |
| `mcp/protocol` | message schema validation (pydantic models) |
| `mcp/cop_server` + `thief_server` | localhost smoke (start, GET /healthz, POST /move with valid + invalid tokens) |
| `mcp/client` | timeout + 401 handling |
| `gmail/formatter` | JSON exactly matches spec § 3.5 schema |
| `gmail/sender` | dry-run mode + idempotency guard (ADR-010) |
| `auth/token_registry` | revoke flow + middleware integration |
| `interface/cli` | each subcommand exits 0 on smoke run |
| `interface/gui` | smoke under offscreen Qt (window + tabs render) |
| `tests/integration/test_drift.py` | layer count consistency (carried from A5 v1.26) |

## 9. Build order summary (one commit per layer)

Full DoD per layer in [TODO.md](TODO.md).

| Layer | Headline |
|---|---|
| 0 | Scaffold + docs (this layer) |
| 1 | shared/* + types + ConfigManager (YAML) |
| 2 | game/* (Board, MoveDynamics, win adjudication, barriers) |
| 3 | sensor/partial_observation + environment/dec_pomdp + reward |
| 4 | model/recurrent_q + soft_update + init |
| 5 | model/vdn_mixer (∑) + 4-test additive identity battery |
| 6 | model/qmix_mixer (monotonic hypernet) + monotonicity test |
| 7 | model/olora (QR orthonormal-init PEFT) + math test |
| 8 | memory/centralised_buffer (sequence-aware, masked) |
| 9 | noise/epsilon_greedy + schedule |
| 10 | services/qmix_update + actor + critic + Polyak (the headline math) |
| 11 | services/vdn_update + services/iql_update (baselines) |
| 12 | services/marl_trainer (CTDE end-to-end fit loop) |
| 13 | services/game_runner (6 sub-games + GameReport) |
| 14 | sdk/* (MarlLab facade + env_builder + trainers + experiments) |
| 15 | mcp/protocol + mcp/cop_server + mcp/thief_server + auth/* (localhost) |
| 16 | mcp/client + game adjudicator over MCP |
| 17 | gmail/formatter + gmail/sender (App-Password + OAuth + idempotency) |
| 18 | interface/cli + 8 subcommands |
| 19 | interface/gui (Tkinter real-time board) |
| 20 | tools/graphify port + tools/viz (plots) |
| 21 | sweeps: grid_size_sweep + algorithm_sweep + observation_radius_sweep |
| 22 | cloud/* — Prefect deployment helpers (stub if creds missing) |
| 23 | Reproducibility + drift-test (carried from A5 v1.26) + meta-consistency lock |
| 24 | Notebook walkthrough (executed) |
| 25 | Audit + reflection answers + comparison table + lessons learned |
| 26 | Final README + EXECUTIVE_SUMMARY + Promptbook + COSTS + V3 tag v1.00 |

## 10. Out of scope

- Real-time cloud deployment that requires the grader to pay for Prefect Cloud — we deploy on the **free tier** if possible, otherwise document the cloud step as a script + screenshot.
- TD3 / SAC — they're single-agent algos; out of scope for MARL.
- LLM-driven agents (Vibe Coding refers to *us* using LLMs to build the project, not the trained agent being an LLM).
- Real-robot deployment — pure simulation.

## 11. Risks + mitigations

| Risk | Mitigation |
|---|---|
| MCP cloud deployment needs creds the user doesn't have | Cloud module is split: `cloud/local.py` (always works) + `cloud/prefect.py` (skippable). README documents both paths. |
| Gmail API needs OAuth setup | Default to App-Password (much simpler); OAuth path is implemented but optional. |
| 5x5 with 2 agents under partial-obs may need long training | Curriculum: train at 2x2 → transfer to 3x3 → ... → 5x5. Layer 21 sweep validates. |
| QMIX monotonicity assumption violated by adversarial reward | Acknowledge in `docs/FAILURE_MODES.md`; report IQL + VDN + QMIX side-by-side. |
| Lecturer needs to verify cloud agents work | Token-authenticated public URL + README has a `curl` example that works against the running server. |

## 12. Extension points (V3 § 12.1)

| # | Surface | Future-work hook |
|---|---|---|
| 1 | `configs/setup.yaml` | All hyperparameters live here |
| 2 | `game.MoveDynamics` | Swap to continuous (x, y) action by writing a new same-API class |
| 3 | `sensor.partial_observation` | Replace Manhattan with line-of-sight ray-cast |
| 4 | `model.{vdn,qmix}_mixer` | Add QPLEX / Weighted-QMIX as a sibling mixer class |
| 5 | `services.marl_trainer` | Curriculum learning hook (start at 2x2, scale up) |
| 6 | `mcp/cop_server` | Add new `@mcp.tool` (e.g., "explain_decision") |
| 7 | `gmail.sender` | Switch send_mode from app_password → oauth → mcp_tool by config |
| 8 | `auth.token_registry` | Add OAuth / JWT validators behind the same `verify(token)` interface |
| 9 | `cloud.prefect` | Swap to Modal / Replicate by writing a same-shape deploy module |
| 10 | `interface.gui` | Add multiplayer / log-replay view via new Tab |

### Why we did NOT add lifecycle hooks

V3 § 12.1 example shows `beforeCreate` / `afterUpdate`. `marl_lab` has a *training loop* but not an event bus. Lifecycle hooks add machinery for hypothetical consumers and don't pay back in the V3 § 7 sense. The 10 surfaces above are concrete future requests, not infrastructure.
