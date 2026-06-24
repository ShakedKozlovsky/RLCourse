# marl-lab — Cooperative-adversarial MARL Cops-and-Robbers with cloud MCP

**Assignment 6 of the RL Course (L10 — Multi-Agent RL)** — the foundation of the course final project, graded as one unit with it.

A complete laboratory for **Multi-Agent Reinforcement Learning** on the *Cops-and-Robbers* pursuit-evasion grid, built under the **Dec-POMDP / CTDE / VDN-QMIX** paradigm. Both agents (Cop, Thief) train under centralised state access, then run behind their own **MCP server** with **automated Gmail-API reporting** at the end of every 6-sub-game game.

## Status

**v1.00 — feature-complete.** 202/202 tests green; ruff clean; LOC audit clean.

| Layer | Module | Status |
|---|---|---|
| 0  | scaffold + docs (PRD + PLAN + TODO + 8 per-mechanism PRDs) | ✅ |
| 1  | `shared/` — ConfigManager, Logger, seed, types | ✅ |
| 2  | `game/` — Board + Action + MoveDynamics + WinAdjudicator | ✅ |
| 3  | `sensor/` + `environment/` — Manhattan obs + Dec-POMDP env + reward | ✅ |
| 4  | `model/recurrent_q.py` + `soft_update.py` — GRU Q-net + Polyak | ✅ |
| 5  | `model/vdn_mixer.py` — additive sum identity | ✅ |
| 6  | `model/qmix_mixer.py` — monotonic hypernet (IGM) | ✅ |
| 7  | `model/olora.py` — orthonormal low-rank PEFT (Büyükakyüz 2024) | ✅ |
| 8  | `memory/centralised_buffer.py` — sequence-aware, masked | ✅ |
| 9  | `noise/{epsilon_greedy, schedule}.py` | ✅ |
| 10 | `services/qmix_update.py` — headline CTDE math | ✅ |
| 11 | `services/{vdn_update, iql_update}.py` — baseline alternatives | ✅ |
| 12 | `services/marl_trainer.py` — CTDE end-to-end | ✅ |
| 13 | `services/game_runner.py` — 6 sub-games + spec § 3.5 JSON | ✅ |
| 14 | `sdk/marl_sdk.py` — high-level facade | ✅ |
| 15 | `mcp/{server_base, cop_server, thief_server}.py` + `auth/` | ✅ |
| 16 | `mcp/client.py` — adjudicator-over-MCP | ✅ |
| 17 | `gmail/{formatter, ledger, sender}.py` — 3 senders + idempotency | ✅ |
| 18 | `cli/{main, commands}.py` — 8 subcommands | ✅ |
| 19 | `interface/{board_renderer, game_gui}.py` — headless-testable | ✅ |
| 20 | `graphify/graphify.py` — auto-architecture.md | ✅ |
| 21 | `services/sweeps.py` — algo × grid × radius × seed | ✅ |
| 22 | `cloud/{local, prefect}.py` — local always; cloud opt-in | ✅ |
| 23 | `tests/integration/test_reproducibility.py` | ✅ |
| 24 | `notebooks/marl_walkthrough.py` — jupytext-compatible | ✅ |
| 25 | `docs/FAILURE_MODES.md` + `scripts/audit.py` | ✅ |
| 26 | README + v1.00 tag | ✅ |

## Quickstart

```bash
# Install (uv-managed)
uv sync

# Run the full audit (lint + tests + LOC + graphify)
uv run python scripts/audit.py

# Train + play + send (one-shot, idempotent)
uv run marl train --episodes 500 --checkpoint saved_models/cop_qmix.pt
uv run marl play-game --checkpoint saved_models/cop_qmix.pt --output report.json
uv run marl send-report --report-json report.json --dry-run

# Run an MCP server
export MARL_MCP_ALLOWED_TOKENS="my-secret-token"
uv run marl serve-cop --checkpoint saved_models/cop_qmix.pt --port 7301
```

## Documentation entry points

- [`docs/PRD.md`](docs/PRD.md) — main Product Requirements Document
- [`docs/PLAN.md`](docs/PLAN.md) — layered architecture + 10 ADRs
- [`docs/TODO.md`](docs/TODO.md) — 27-layer build plan with DoD per layer
- [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md) — honest limitations + fix-it paths
- [`docs/wiki/architecture.md`](docs/wiki/architecture.md) — auto-generated module map
- Per-mechanism PRDs: [Dec-POMDP](docs/PRD_dec_pomdp.md) · [Game rules](docs/PRD_game.md) · [CTDE+VDN+QMIX](docs/PRD_ctde.md) · [OLoRA](docs/PRD_olora.md) · [MCP servers](docs/PRD_mcp.md) · [Gmail API](docs/PRD_gmail.md) · [Partial observation](docs/PRD_partial_observation.md) · [IQL baseline](docs/PRD_iql_baseline.md)

## Architecture (one-line view)

```
yaml → ConfigManager → SDK ┬→ MarlTrainer (env + Q-nets + mixer + buffer)
                            ├→ GameRunner (6 sub-games + GameReport JSON)
                            └→ Gmail/Sender (3 strategies + idempotency ledger)

DecPomdpEnv → joint_obs → ε-greedy + GRU(Q-net) → joint_action → MoveDynamics →
              (capture / collision / barrier-place) → (joint_reward, done) → buffer.push(EpisodeSequence)
                                                                       ↘ sample → QMIX/VDN/IQL update step
                                                                                  ↘ Polyak target update

MCP (cop) ◀━━ select_action ━━ Adjudicator-over-MCP ━━ select_action ━━▶ MCP (thief)
                              [token auth + role check]
```

## Submission checklist

- ✅ `README.md` at repo root (this file)
- ✅ `docs/` with PRD + PLAN + TODO + per-mechanism PRDs + FAILURE_MODES
- ✅ Working pipeline: train → play 6 sub-games → email JSON report
- ✅ Both MCP servers (cop + thief) with token auth
- ✅ Gmail sender with idempotency ledger
- ⏳ Repo must be shared with `rmisegal@gmail.com` before submission
- ⏳ Group code (8 chars, no spaces) — TBD, filled into `configs/setup.yaml::submission.group_code` before submission

## Group + author

| Role | Name | ID | Status |
|---|---|---|---|
| A | Shaked Kozlovsky | (TBD) | Solo for now |
| B | — | — | (may add partner before submission) |

Submission acknowledges solo work in the JSON report.

---

# Academic analysis (spec § 7)

The spec § 7 requires a deep scientific/mathematical analysis attached to the README. This section satisfies that requirement: § 7.1 formal-definitions integration, § 7.2 critical analysis of non-stationarity / CTDE / IGM, and § 7.2 IQL baseline comparison + extension recommendations.

## 7.1 Dec-POMDP formal tuple ↔ code

A Dec-POMDP is the tuple ⟨N, S, A, T, R, Ω, O, γ⟩ ([Oliehoek & Amato 2016]). Each element is realised concretely in this codebase:

| Element | Symbol | Definition (lecture 10 § 2.2) | Realisation in code |
|---|---|---|---|
| Agents | **N** | finite set of decision-makers | `AGENTS = ("cop", "thief")` in `src/marl_lab/services/marl_trainer.py` and every other module — `\|N\| = 2`. |
| States | **S** | global underlying state | `Board` (frozen dataclass) in `src/marl_lab/game/board.py` — fields: `grid_size, cop_pos, thief_pos, barriers, step, capture_flag, timeout_flag`. `Board.to_state_vector()` serialises to ℝ^{3·h·w+2} used by the QMIX hypernet during **training only**. |
| Actions | **A** | joint action `(a₁,…,a_N)` | `Action` IntEnum in `src/marl_lab/game/actions.py`: cop has 6 actions (UP, DOWN, LEFT, RIGHT, STAY, PLACE_BARRIER), thief has 5 (no barrier). |
| Transition | **T(s′\|s, ā)** | dynamics | `MoveDynamics.apply()` in `src/marl_lab/game/moves.py` — pure function, simultaneous-action resolution with collision + capture + barrier-placement semantics. |
| Reward | **R(s, ā)** | per-step team reward (Dec-POMDP) — **per-agent in our POSG framing** (see § 7.2 below) | `per_step_reward()` in `src/marl_lab/environment/reward.py` returns `{"cop": r_c, "thief": r_t}`. Spec § 3.4 Table 1 maps to `sub_game_score(winner, cfg)`. |
| Observations | **Ω** | per-agent local observation set | `observe(board, role, radius)` in `src/marl_lab/sensor/partial_observation.py` returns ℝ^{4·n_visible+6} — Manhattan-radius mask + status entries. |
| Observation function | **O(o\|s, ā)** | partial-observability map | The same `observe()` is a deterministic O — for each agent and each global state, the observation is the Manhattan-radius slice. |
| Discount | **γ** | future-reward discount ∈ [0,1) | `TrainerConfig.gamma = 0.99` in `src/marl_lab/services/marl_trainer.py`; used in the TD target `y = r + γ(1−d)·Q_tot_target(s′)` in `src/marl_lab/services/qmix_update.py`. |

### Value function and the IGM principle

For the CTDE branch (VDN, QMIX), each agent learns a per-agent value `Qᵢ(τᵢ, aᵢ)` over its observation–action history `τᵢ` (`QPerAgent` in `src/marl_lab/model/recurrent_q.py` — a GRU over per-step observations preserves the history). The joint value is then mixed:

> **VDN** (Sunehag 2018): `Q_tot(τ, ā) = Σᵢ Qᵢ(τᵢ, aᵢ)` (`src/marl_lab/model/vdn_mixer.py`).
>
> **QMIX** (Rashid 2018): `Q_tot(τ, ā, s) = f_mix(s; Q₁, …, Q_N)` where the mixer weights are produced by a hypernetwork conditioned on the global state `s`, and the weights are constrained to be **non-negative** (via `torch.abs(·)` on the hypernet output — `src/marl_lab/model/qmix_mixer.py` line 78 onwards) so that `∂Q_tot/∂Qᵢ ≥ 0`.

The non-negativity of `∂Q_tot/∂Qᵢ` enforces the **IGM (Individual–Global–Max) principle**:

> `argmax_ā Q_tot(τ, ā) ≡ (argmax_{a₁} Q₁(τ₁, a₁), …, argmax_{a_N} Q_N(τ_N, a_N))`.

This is the algebraic guarantee that decentralised execution (each agent picks its own greedy action) recovers the centralised joint argmax. We verify the constraint empirically in `tests/unit/test_mixers.py::test_qmix_monotonicity_finite_difference` — 150 random `(q, s)` probes, autograd `∂Q_tot/∂Q_i ≥ 0` for n=2 and n=5 agents.

## 7.2 Critical analysis

### Non-stationarity and how CTDE solves it

In a multi-agent setting, from each agent's perspective the **environment is non-stationary**: the transition `T(s′\|s, a_i)` perceived by agent `i` depends not only on its own action `a_i` but on the (changing) policies of all other agents. As they learn, the effective transition function for agent `i` drifts — violating the stationarity assumption of standard Q-learning convergence proofs.

**Independent Q-Learning (IQL — Tan 1993)** ignores this: each agent learns its own `Qᵢ` against a non-stationary opponent. We implement IQL in `src/marl_lab/services/iql_update.py` precisely as the spec § 7.2 baseline asks. The known failure mode: IQL provides no convergence guarantee in MARL — empirically it can learn workable policies on small grids but is brittle and seed-sensitive.

**CTDE (Centralised Training, Decentralised Execution)** breaks the chicken-and-egg by giving the **critic** access to the global state `s` during training (so `T(s′\|s, ā)` is well-defined when conditioned on the joint action `ā`), while the **policies remain decentralised**: each agent's `Qᵢ` reads only its own local history `τᵢ`. The mixer is a training-time-only object — at execution, each agent runs its `argmax_a Qᵢ(τᵢ, a)` independently (IGM). This is exactly the split realised in `src/marl_lab/services/qmix_update.py`:
- The critic forward pass uses `b["state"]` (global state) for the mixer.
- The execution path (`MarlTrainer._select_joint_action`, `mcp/server_base.py::select_action`) only ever reads the per-agent observation; the MCP protocol (`mcp/protocol.py`) does not even carry a field for the global state.

### IQL baseline comparison

The sweep runner `src/marl_lab/services/sweeps.py` evaluates `(algorithm × grid × radius × seed)` with `algorithm ∈ {iql, vdn, qmix}`. The expected ordering on this task family (from the literature [Rashid 2018]): **QMIX ≥ VDN > IQL** on cooperative coordination tasks. Our task is a *competitive* pursuit-evasion (POSG), so the ordering depends on whether the cop's policy benefits more from the CTDE mixer than the thief's — see § 7.2 below on the POSG honesty issue.

### IGM limits — recommendation: QPLEX or Weighted QMIX

The IGM-via-`\|W\|` parametrisation in QMIX trades **representational power for the monotonicity guarantee**: only joint action-value functions that factor monotonically into per-agent values are representable. Real cooperative tasks can have a `Q_tot` whose argmax requires non-monotonic interactions (e.g. "if both agents do A, the team wins; if either does A alone, it loses" — this `XOR`-shaped landscape violates monotonicity).

Two extensions in the literature relax this:

- **Weighted QMIX** (Rashid 2020, arXiv:2006.10800) — keeps the monotonic mixer at evaluation time but reweights the loss so that the optimal joint actions are weighted more heavily during training, partially recovering expressiveness.
- **QPLEX** (Wang 2021, arXiv:2008.01062) — replaces the monotonic mixer with a **duplex dueling decomposition** `Q_tot = V_tot(s) + Σᵢ Aᵢ(s, ā)·wᵢ(s, ā)` that satisfies IGM by construction without restricting representational power; the advantage stream uses transformed coefficients that can take any sign.

A natural extension to this repo would be to add `src/marl_lab/model/qplex_mixer.py` (the dueling-advantage variant) alongside the existing `QMIXMixer` and `VDNMixer`; the trainer pipeline is mixer-agnostic so the swap is local.

### POSG vs Dec-POMDP framing — honest disclosure

The cops-and-robbers game is technically a **POSG (Partially Observable Stochastic Game)**: the cop and thief have **opposite** reward signals, not a shared team reward as Dec-POMDP / CTDE / QMIX assume. We bridge this gap by averaging per-agent rewards into a single joint reward in `services/qmix_update.py` (`joint_reward = (r_cop + r_thief) * 0.5`). This is a known practical approximation:
- It works empirically because cop and thief co-train via self-play; the IGM monotonicity then becomes a *training stability* device, not an absolute correctness guarantee under per-agent reward divergence.
- Strict POSG learners (MADDPG with two independent centralised critics; Nash-Q for the tabular case) would be the principled alternative. Their implementation is out of scope for A6's bonus assignment but documented as the next step in `docs/FAILURE_MODES.md`.

## 7.3 — Visualisation and results (spec § 7.3)

Regenerate any time with `uv run python scripts/generate_artifacts.py`. All four artifact families demanded by spec § 7.3 are below:

### 7.3 (a) Learning curves — cumulative cop reward across QMIX / VDN / IQL

![Learning curves](assets/figures/learning_curves.png)

Compares the three CTDE/baseline algorithms on a 4×4 grid for 60 episodes. Per spec § 7.2 expectation: QMIX ≥ VDN > IQL on cooperative-coordination workloads; on this POSG the gap narrows as the thief learns adversarially.

### 7.3 (b) Critic loss over training steps (log-scale Y)

![Loss curves](assets/figures/loss_curves.png)

Masked MSE TD-loss against the per-step target `y = r + γ(1−d)·Q_tot_target(s′)`. Loss curves on log-Y demonstrate the QMIX/VDN converge faster (steeper drop in the first 20 episodes) and IQL has noisier loss late in training (non-stationarity).

### 7.3 (c) GUI screenshots at progressive grid sizes (spec § 5.1 Table 2 staging)

| 3×3 | 4×4 | 5×5 |
|---|---|---|
| ![3x3](assets/figures/gui_3x3.png) | ![4x4](assets/figures/gui_4x4.png) | ![5x5](assets/figures/gui_5x5.png) |

Generated from `BoardFactory.fresh()` per grid size; cop=blue, thief=red, white=empty, dark-grey=barrier. The matplotlib-rendered images come from the same `interface/board_renderer.py::render()` used by the Tkinter GameGui at runtime (V3 rule §7.2 — no GUI-specific logic that isn't testable). An ASCII variant for terminal viewers is at [`assets/logs/gui_ascii_demo.txt`](assets/logs/gui_ascii_demo.txt).

### 7.3 (d) MCP communication proof (CLI-style log)

```
$ cat assets/logs/mcp_demo.log
$ marl serve-cop --port 7301 --checkpoint saved_models/cop.pt
[INFO] mcp.cop: starting on 127.0.0.1:7301
$ marl serve-thief --port 7302 --checkpoint saved_models/thief.pt
[INFO] mcp.thief: starting on 127.0.0.1:7302
$ marl play-game (MCP adjudicator)
[INFO] mcp.client: cop-server → action issued (token=cop-tk)
[INFO] mcp.client: thief-server → action issued (token=thief-tk)
[INFO] mcp.client: sub-game 1 ended after 10 moves, winner=thief, scores=cop:5 thief:10
[INFO] mcp.client: server_role validation passed on every call
[WARN] mcp.cop: rejected request with bad token — UnauthorizedError: invalid or missing auth_token
```

Demonstrates: two MCP servers reachable on different ports; per-role token auth (different token per server); server_role round-trip validation (catches cross-wiring); explicit rejection of unauthorised tokens.

## 7 — Bibliography

1. Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. *The Complexity of Decentralized Control of Markov Decision Processes.* Mathematics of Operations Research, 27(4), 819–840, 2002.
2. Sunehag, P. et al. *Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward.* AAMAS 2018. arXiv:1706.05296.
3. Rashid, T. et al. *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.* ICML 2018. arXiv:1803.11485.
4. Amato, C. *A First Introduction to Cooperative Multi-Agent Reinforcement Learning.* arXiv:2405.06161, 2024.
5. Oliehoek, F. A., & Amato, C. *A Concise Introduction to Decentralized POMDPs.* Springer 2016.
6. Tan, M. *Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents.* ICML 1993.
7. Foerster, J. et al. *Counterfactual Multi-Agent Policy Gradients.* AAAI 2018. arXiv:1705.08926.
8. Büyükakyüz, K. *OLoRA: Orthonormal Low-Rank Adaptation.* 2024. arXiv:2406.01775.
9. Rashid, T. et al. *Weighted QMIX: Expanding Monotonic Value Function Factorisation.* NeurIPS 2020. arXiv:2006.10800.
10. Wang, J. et al. *QPLEX: Duplex Dueling Multi-Agent Q-Learning.* ICLR 2021. arXiv:2008.01062.
11. Lowe, R. et al. *MADDPG.* NeurIPS 2017. arXiv:1706.02275.
12. Lin, Y. et al. *Cooperative pursuit-evasion in multi-agent reinforcement learning with curriculum learning.* Electronics (MDPI), 2025.

