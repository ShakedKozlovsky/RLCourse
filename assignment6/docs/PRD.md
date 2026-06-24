# Product Requirements Document — marl-lab

> **Assignment 6** of the RL Course (L10 — MARL). **Final-project foundation.** Dr. Yoram Segal, 2026.

## 1. Project goal

Build a complete laboratory for **Multi-Agent Reinforcement Learning** on the *Cops-and-Robbers* pursuit-evasion grid, with:

1. A **mathematically-rigorous Dec-POMDP** formulation of the problem (proof + code traceability).
2. A **CTDE** training pipeline implementing **VDN** and **QMIX** value-decomposition, with **Independent Q-Learning (IQL)** as the documented baseline.
3. **Two independent MCP servers** (one per agent — Cop, Thief) that communicate over HTTP; runnable on localhost AND deployed to the cloud with **revokable token authentication**.
4. **Automated Gmail-API reporting**: one email per 6-sub-game game, JSON body, sent to `rmisegal+marl@gmail.com`.
5. A **GUI** (Tkinter) showing real-time board state, agent positions, moves, and progression.
6. Optional **OLoRA** (Orthonormal Low-Rank Adaptation) for parameter-efficient fine-tuning of a pre-trained backbone.

Pedagogical claim under test:

> *L10 § 4: **"CTDE with value decomposition (VDN/QMIX) solves the Dec-POMDP non-stationarity problem that Independent Q-Learning (IQL) cannot. The Individual-Global-Max (IGM) principle guarantees the local maximisers are also the global optimum — but only under monotonicity constraints (QMIX) or strict additive structure (VDN)."***

## 2. Mapping from L10 lecture → code locations

| L10 section / concept | Where it lives in `marl_lab` |
|---|---|
| § 2 — Dec-POMDP ⟨N, S, A, T, R, Ω, O, γ⟩ | [`PRD_dec_pomdp.md`](PRD_dec_pomdp.md) + [`environment/dec_pomdp.py`](../src/marl_lab/environment/dec_pomdp.py) |
| § 2.1 — Cooperative vs adversarial / POSG | [`environment/reward.py`](../src/marl_lab/environment/reward.py) (zero-sum rewards: cop_win = +20, thief_win = +10) |
| § 2.2 — Pursuit-evasion swarm cooperative framing | [`PRD_game.md`](PRD_game.md) § 1 |
| § 3 — Non-stationarity + IQL failure | [`PRD_iql_baseline.md`](PRD_iql_baseline.md) + [`services/iql_baseline.py`](../src/marl_lab/services/iql_baseline.py) |
| § 4 — CTDE + VDN + QMIX + IGM | [`PRD_ctde.md`](PRD_ctde.md) + [`model/vdn_mixer.py`](../src/marl_lab/model/vdn_mixer.py) + [`model/qmix_mixer.py`](../src/marl_lab/model/qmix_mixer.py) |
| § 5 — LSTM/GRU recurrent obs | [`model/recurrent_q.py`](../src/marl_lab/model/recurrent_q.py) |
| § 6 — OLoRA QR-decomp PEFT | [`PRD_olora.md`](PRD_olora.md) + [`model/olora.py`](../src/marl_lab/model/olora.py) |
| § 7 — Staged validation 2x2 → 5x5 | [`configs/setup.yaml::experiments.grid_size_sweep`](../configs/setup.yaml) |
| § 8 — Cloud MCP architecture | [`PRD_mcp.md`](PRD_mcp.md) + [`mcp/`](../src/marl_lab/mcp/) |
| § 9 — GUI + visualisation + auto-reporting | [`interface/gui/`](../src/marl_lab/interface/gui/) + [`PRD_gmail.md`](PRD_gmail.md) |

## 3. The Dec-POMDP tuple (verbatim — L10 equation 1)

$$\mathrm{Dec\text{-}POMDP} = \langle N,\, S,\, A,\, T,\, R,\, \Omega,\, O,\, \gamma \rangle$$

| Symbol | Meaning | This project |
|---|---|---|
| `N` | Agent set | {cop, thief} (2 agents — extensible to swarm) |
| `S` | Global state space | (cop_pos, thief_pos, barriers, step, capture_flag) |
| `A = A₁ × A₂` | Joint action space | each Aᵢ = {UP, DOWN, LEFT, RIGHT, STAY, PLACE_BARRIER*} (*cop only) |
| `T(s' \| s, ā)` | Transition kernel | Deterministic — collisions cap at walls/barriers |
| `R(s, ā)` | Reward (shared scalar for fully-cooperative; we use **competing scalars** per agent → POSG) | `+20/-5` for cop on capture / loss; `+10/-5` for thief |
| `Ω` | Joint observation space | Manhattan-radius local view per agent |
| `O(ō \| s', ā)` | Observation function | Deterministic mask of `S` to radius-r neighbourhood |
| `γ` | Discount | 0.99 |

**Honest model framing**: because cop and thief have **distinct reward functions**, the strict-Dec-POMDP cooperative assumption is violated; the rigorous model is a **POSG** (Partially Observable Stochastic Game; L10 equation 3). The CTDE/VDN/QMIX machinery is documented as a *practical approximation* that works empirically in the literature on pursuit-evasion MARL [Lin 2025], with explicit acknowledgement in the analysis section.

## 4. Game environment (the engineering spine)

| Item | Value | Source |
|---|---|---|
| Grid | 5×5 default, configurable 2×2 → 5×5 | Spec § 3.1 |
| Players | 2 AI agents — Cop, Thief | Spec § 2 |
| Game = 6 sub-games | One email per game | Spec § 3.5 |
| Sub-game ≤ 25 moves | Cop wins on capture; thief wins on survival | Spec § 3.2 |
| Cop's bonus action | Place barrier on adjacent cell (max 5 per game) | Spec § 3.3 |
| Scoring per sub-game | cop_win 20 · thief_win 10 · cop_loss 5 · thief_loss 5 | Spec § 3.4 |
| Observation radius | Manhattan r = 2 default | Implementation choice (§ 1.5 of L10's partial-observation diagram) |
| Config-driven | EVERY numeric in `configs/setup.yaml` | Spec § 3.6 + V3 § 7.2 |

## 5. CTDE training pipeline

| Stage | What happens |
|---|---|
| Centralised collection | Run the env. The **central learner** sees full state `s`, each agent's observation `oᵢ`, joint action `ā`, joint reward `r̄`. |
| Replay buffer | Centralised buffer of full episode tuples `(s, ō, ā, r̄, s', ō')`. |
| Per-agent Q-net | `Qᵢ(oᵢ_t, hᵢ_t)` — GRU recurrent on observation history (slide 5). |
| Mixing | VDN: `Q_tot = Σ Qᵢ`. QMIX: `Q_tot = f_θ(Q₁, …, Qₙ ; s)` with f monotonic in each Qᵢ. |
| TD target | `y = r + γ · max_ā' Q_tot(s', ā')` — bootstrapped via target mixer (Polyak τ = 0.005). |
| Decentralised execution | At play time each agent receives ONLY its local observation; runs its own Qᵢ; communicates moves over MCP to the env adjudicator. |

## 6. Hyperparameters

All in `configs/setup.yaml`. Justifications + sources:

| Param | Value | Why |
|---|---|---|
| γ | 0.99 | Standard discount; horizon ≈ 25 steps (= 1 sub-game) |
| τ | 0.005 | Slow Polyak target update — same as Lillicrap 2016 in A5 |
| Actor LR | 1e-4 | Conservative — actor's gradient amplified by mixer derivative |
| Critic / mixer LR | 1e-3 | Tracks moving target faster |
| Batch | 64 | RNN sequence batch; smaller than DDPG because each batch has T-step BPTT |
| Replay capacity | 100 k | Enough for 5 000 episodes × 20 steps |
| Hidden | [128, 128] | Mid-size; OLoRA can swap with a larger pre-trained backbone |
| GRU hidden | 64 | Memory width per agent — captures last few moves of opponent |
| ε-initial | 1.0 | Pure random warmup |
| ε-final | 0.05 | Always some exploration for non-stationarity |
| ε-decay | 50 k steps | Tapers exactly at end of training |

## 7. Empirical study plan (§ 7 of the spec)

| Study | Cells | Seeds | Headline question |
|---|---|---|---|
| **Staged grid-size validation** | 2×2, 3×3, 4×4, 5×5 | 3 | § 5.1 Table 2 — does the pipeline survive scaling? |
| **Algorithm comparison** | IQL, VDN, QMIX | 3 | Does CTDE beat IQL non-stationarity? |
| **Observation-radius sweep** | r ∈ {1, 2, 3} | 3 | How much does partial-observation hurt? |
| **OLoRA vs full-finetune** | (off, on) | 3 | Does PEFT help in unstable RL? |

## 8. Visualisations (mandatory per spec § 7.3)

| Required | Where |
|---|---|
| Learning curves per agent | `assets/plots/learning_curve_cop.png`, `learning_curve_thief.png` |
| Loss curves over training stages | `assets/plots/loss_curves.png` |
| GUI screenshots at multiple grid sizes | `assets/gui/grid_{2x2,3x3,4x4,5x5}.png` |
| MCP communication logs / CLI screenshots | `assets/logs/mcp_session.log` |
| Trajectory overlay on the board | `assets/plots/trajectory_overlay.png` |

## 9. Three reflection questions (§ 7.2 — analysis)

1. **CTDE non-stationarity solution**: How does centralised training with decomposed values escape the IQL non-stationarity trap?
2. **IGM limits**: When does QMIX's monotonicity assumption break? What does QPLEX / Weighted QMIX add?
3. **Cooperative pursuit-evasion**: How does the *swarm vs single agent* framing change the optimal-policy story?

## 10. Originality hooks (V3 § 1.4 + spec § 5.2 invites)

| Hook | Why beyond spec |
|---|---|
| **Mini-Graphify port** (carried from A4 + A5) | AST walker → Obsidian Vault for `src/marl_lab` |
| **Full LSTM ablation** vs GRU | L10 § 5 mentions both; we report which works on partial-obs |
| **Curriculum learning** 2×2 → 5×5 with transfer | Lin 2025 paper directly cited; transfer the smaller-grid policy as the bigger-grid warmstart |
| **Counterfactual MAPG** (Foerster 2018) as alt to QMIX | Mentioned in spec § 7.2 — implementing it as a 100-LOC alternative |
| **GUI replay-from-log** | Load a session log and visually replay any past game |
| **Drift-test for layer count** | Carried from A5 v1.26 — prevents recurring intro-count drift |
| **Iterative-adversarial-review pattern** | The methodology that drove A5 from 82 → 95 — applied here from day 1 |

## 11. KPIs (Definition of Project Done)

| KPI | Target |
|---|---|
| L10 coverage | 100 % of L10 sections § 2–8 have a corresponding code module |
| Spec compliance | Every Mandatory item from `EX06.pdf` § 5 satisfied |
| All `configs/setup.yaml` keys consumed | Yes (config-coverage test) |
| Tests | ≥ 100 unit + integration; coverage ≥ 85 % |
| Files ≤ 150 LOC each | Hard rule |
| ruff clean | 0 warnings |
| 3 reflection questions | Answered with empirical evidence + citations |
| MCP localhost demo | End-to-end game runs |
| MCP cloud demo | Both agents reachable behind tokens (or fully documented stub if cloud creds missing) |
| Gmail email | Real send works (or documented stub with screenshot) |
| GUI screenshots at all 4 grid sizes | Yes |
| Originality hook | Mini-Graphify + curriculum + counterfactual MAPG |

## 12. Deliverables

1. `src/marl_lab/` — modular Python under V3 rules
2. `tests/` — math batteries + integration tests
3. `configs/setup.yaml` — single config (versioned)
4. `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` + 8 per-mechanism PRDs
5. `notebooks/marl_lab_walkthrough.ipynb` — executed end-to-end
6. `assets/plots/*.png` + `assets/gui/*.png` + `assets/logs/*`
7. `docs/PROMPTBOOK.md`, `docs/COSTS.md`, `docs/EXECUTIVE_SUMMARY.md`, `docs/AUDIT.md`, `docs/FAILURE_MODES.md`, `docs/LESSONS_LEARNED.md`
8. `.github/workflows/assignment6-ci.yml` — green badge
9. Tag `assignment6-v1.00` after the final layer

## 13. Sources

1. Bernstein et al., *Complexity of decentralised control of MDPs*, MoOR 2002.
2. Sunehag et al., *Value-Decomposition Networks for Cooperative MARL*, AAMAS 2018.
3. Rashid et al., *QMIX: Monotonic Value Function Factorisation*, ICML 2018.
4. Amato, *A First Introduction to Cooperative MARL*, arXiv:2405.06161, 2024.
5. Lin et al., *Cooperative Pursuit-Evasion with Curriculum Learning*, Electronics 2025.
6. Segal Y., *Lecture 10 — MARL: theoretical and engineering analysis*, BIU 2026.
7. Büyükakyüz, *OLoRA — Orthonormal Low-Rank Adaptation*, arXiv:2406.01775, 2024.
8. Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018.
9. Rashid et al., *Weighted QMIX*, NeurIPS 2020.
10. Wang et al., *QPLEX — Duplex Dueling MA-Q-Learning*, ICLR 2021.
11. Anthropic, *MCP specification*, modelcontextprotocol.io, 2024.
12. Course material: V3 PDF `software_submission_guidelines-V3.pdf` (Dr. Y. Segal 2026).
13. Assignment spec: `EX06.pdf` (Dr. Y. Segal 2026).
