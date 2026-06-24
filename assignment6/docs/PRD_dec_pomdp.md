# PRD — Dec-POMDP formalisation

> Per-mechanism PRD. The mathematical contract the rest of the project rests on. L10 § 2 + spec § 5 demand a formal tuple proof.

## 1. The tuple (L10 equation 1)

$$\mathrm{Dec\text{-}POMDP} = \langle N, S, A, T, R, \Omega, O, \gamma \rangle$$

| Symbol | Meaning | `marl_lab` instantiation |
|---|---|---|
| `N` | Agent set | `{cop, thief}` (n=2; extensible) |
| `S` | Global state | `(cop_xy, thief_xy, barriers: frozenset, step: int, capture_flag: bool)` — `\|S\|` is finite and bounded by `(H·W)² · 2^(H·W) · max_moves · 2` |
| `A = A₁ × A₂` | Joint action | `Aᵢ = {UP, DOWN, LEFT, RIGHT, STAY}`; cop also has `PLACE_BARRIER` per cell-adjacent target |
| `T(s' \| s, ā)` | Transition kernel | Deterministic — moves resolve simultaneously; collisions cap at walls and barriers; PLACE_BARRIER permits no move that turn |
| `R(s, ā)` | Reward | **Per-agent** scalars (POSG framing — see § 3): `R_cop`, `R_thief`. Sub-game-end events emit Table-1 scores (cop_win=+20, thief_win=+10, etc.); per-step shaping is configurable in `configs/setup.yaml::reward` |
| `Ω` | Joint observation space | `Ω₁ × Ω₂` where each `Ωᵢ` = local Manhattan-radius view |
| `O(ō \| s', ā)` | Observation function | Deterministic mask: `oᵢ` = subset of `s'` cells whose Manhattan distance from agent `i` ≤ `observation_radius`. If thief is OUTSIDE cop's view → cop sees `(unseen)` marker |
| `γ` | Discount | 0.99 (per spec convention) |

## 2. Code → tuple traceability

| Tuple element | Source file |
|---|---|
| `N` | `game/board.py::Board.AGENTS` |
| `S` | `game/board.py::Board` |
| `A` | `game/actions.py::Action` (Enum) |
| `T` | `game/moves.py::MoveDynamics.apply` |
| `R` | `environment/reward.py::compute_reward` |
| `Ω, O` | `sensor/partial_observation.py::observe` |
| `γ` | `configs/setup.yaml::marl.gamma` |

## 3. Honest framing: POSG vs Dec-POMDP

L10 § 2.1 makes the point explicit: pure cooperative Dec-POMDP requires **one shared reward**. Cops-and-Robbers is **adversarial** (cop maximises capture; thief maximises evasion). The correct strict model is a **Partially Observable Stochastic Game (POSG)** with per-agent reward functions:

$$G = \langle I, S, \{A_i\}, \{O_i\}, P, \Omega, \{R_i\}, \gamma \rangle$$

with NEXP-NP complexity for optimal cooperative-best-response equilibrium (vs NEXP-Complete for Dec-POMDP).

We **implement** CTDE/VDN/QMIX (which were designed for cooperative Dec-POMDP) as a *practical approximation* — well-documented in the pursuit-evasion MARL literature ([Lin 2025] uses curriculum-MAPPO on the same problem). The disclosure lives in `docs/FAILURE_MODES.md`.

## 4. The IGM (Individual-Global-Max) principle

The Dec-POMDP / value-decomposition core principle:

$$\arg\max_{\bar{a}} Q_{tot}(s, \bar{a}) = \left( \arg\max_{a_1} Q_1(o_1, a_1),\, \dots,\, \arg\max_{a_n} Q_n(o_n, a_n) \right)$$

i.e. the centralised optimum factorises into independent per-agent argmaxes. **VDN** (sum) and **QMIX** (monotonic) are two functional forms guaranteed to satisfy IGM.

| Mixer | IGM guarantee |
|---|---|
| VDN: `Q_tot = ∑ Qᵢ` | Automatic (every term is monotone in itself) |
| QMIX: `Q_tot = f(Q₁,…,Qₙ ; s)` with `f` monotonic in each Qᵢ | Enforced by `\|W\|` parametrisation in the hypernet |
| QPLEX (Wang 2021) | Broader functional class via duplex dueling |

## 5. Test plan

| Test | Pass criterion |
|---|---|
| State tuple ⟨N, S, A, T, R, Ω, O, γ⟩ — each element present in `Board` + config | Static fields all exist |
| `O` deterministic | Same `s` + agent → same `oᵢ` |
| `T` deterministic + bounded | Same `(s, ā)` → same `s'`; `s'.step == s.step + 1` |
| `R_i` separately accessible | `env.reward_for("cop") != env.reward_for("thief")` after a capture |
| IGM-violation alarm | If we ever swap in QPLEX, a test that asserts QMIX-monotone fails for that mixer fires a warning |

## 6. Acceptance criteria

1. The tuple is fully constructible from `Board` + config — pickle round-trips it.
2. The per-agent reward functions are pure (no env mutation).
3. The transition is deterministic at fixed seed (Layer 23 reproducibility test).
4. `docs/FAILURE_MODES.md` discloses the POSG / Dec-POMDP gap with a citation.

## 7. Non-goals

- Continuous-action MARL (out of scope; spec is discrete).
- Communication channels between agents *at execution time* (the spec's MCP protocol IS the communication — but only via the env adjudicator, not direct agent-to-agent during execution).
- LLM-driven agents (Vibe Coding ≠ LLM agent).
