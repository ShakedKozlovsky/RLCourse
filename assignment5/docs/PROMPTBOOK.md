# Promptbook — AI-assisted development log (Assignment 5)

> Required by V3 § 8.3 / § 17.1 / § 20.9 # 1. This captures the prompt-engineering
> methodology used to build `roomba-lab` end-to-end with Claude Opus 4.7 as the
> implementer under a single human Software Architect.

## 0. The meta-rule

Same as Assignment 4 — used at session start:

> *"Use the V3 coding-rules PDF as your reference. ≤ 150 LOC per file, ruff
> clean, ≥ 85 % coverage gate, uv only, no magic numbers, layered architecture,
> `Layer N: <summary>` commit messages, PRD + PLAN + TODO + per-mechanism PRDs
> before any code."*

## 1. Domain framing (the only A5-specific opening)

> *"Assignment 5 = DDPG + cleaning robot + HouseExpo. Read the L09 lecture PDF
> and EX05 spec. Identify the **hard constraints** (no Gym, no Gazebo, custom
> 2-D sim from scratch, HouseExpo maps required, 3 reflection questions,
> 2 mandatory graphs). Pick a project name; mine carries 'lab' from prior work."*

Outcome: `roomba_lab` chosen (the spec literally compares the task to "Roomba"
in Hebrew). Project structure inherited from Assignment 4 with one new package
(`simulator/`) and a renamed pair (`environment/` becomes
no-gym wrapper; `data/` is for HouseExpo I/O — disambiguated from `memory/`
which is the replay buffer per ADR-008).

## 2. PRD + PLAN + TODO authoring

> *"Write docs/PRD.md with the slide → code mapping, the two core equations
> verbatim, the empirical study plan, the 3 reflection questions as KPIs."*

> *"Write docs/PLAN.md with the layered architecture diagram, package paragraph
> summaries, full DDPG pseudocode (annotated to slide numbers), config schema,
> and **9 ADRs** documenting design decisions with trade-offs (custom env per
> ADR-001 → no gym; shapely + numpy split per ADR-002; soft updates default per
> ADR-004; etc.)."*

> *"Write docs/TODO.md as a 17-layer task list. Each layer = one commit. Each
> layer has explicit DoD covering code + tests + docs."*

The plan-first round took ~25 minutes wall-clock and ~30 K input tokens. The
ratio held: every Layer-N implementation that worked on first try was one whose
DoD was unambiguous before code was written.

## 3. The 4-test math battery pattern (Layers 4 + 7)

The same TDD pattern that worked for Assignment 4's GAE and PPO-clip layers.
For Polyak in this project:

> *"Write `tests/unit/test_soft_update.py` FIRST with 4 cases: (1) τ=0 → target
> unchanged, (2) τ=1 → target = source (hard copy), (3) τ=0.5 → midpoint,
> (4) repeated calls converge target → source. Only after the tests are
> written, implement `model/soft_update.py::polyak_update`."*

For DDPG update step:

> *"Write `tests/unit/test_ddpg_update.py` with 6 gradient-flow + math cases:
> critic_loss finite + non-negative, critic gradient flows ONLY to critic (not
> actor, not target), actor gradient flows ONLY to actor, one step changes
> weights >1e-6, diagnostics returned, τ=0 freezes target. Then implement."*

These tests both passed on first implementation, with no follow-up
red-light cycles. That is the headline win of TDD on math layers.

## 4. The "no gym imports" constraint enforcement

> *"Verify the custom env wrapper imports zero gym / gymnasium packages. Add a
> rule to PRD § 4 stating this as a hard requirement. The env should mirror the
> Gym API SHAPE (reset, step → (obs, r, done, info)) but be a plain Python
> class."*

Used `grep "import gym" src/` after every commit until Layer 12; no false
positives. The constraint is in [`docs/PRD.md`](PRD.md) § 11 KPIs and in
[`docs/PLAN.md`](PLAN.md) ADR-001.

## 5. Live-data integration

> *"HouseExpo is upstream at github.com/TeaganLi/HouseExpo. Download the
> json.tar.gz (one-time, ~25 MB), extract the 10 maps listed in their
> map_id_10.txt, and commit them at data/raw/sample_maps/. Don't commit the
> full tar.gz — keep it gitignored."*

This kept the repo small (~80 KB of sample data) while still proving the
loader works on real upstream data.

## 6. The reflection-question-evidence loop

> *"Layer 11 should produce a 4-cell × 3-seed noise-σ sweep on the primary
> apartment. The σ=0.0 cell IS the reflection-Q2 evidence. Run it. Wire the
> result into the README so the answer points at a JSON, not at a hand-wave."*

Cost ~9 minutes CPU and produced [`results/sweeps/noise_sigma.json`](../results/sweeps/noise_sigma.json) +
[`assets/plots/noise_sigma_sweep.png`](../assets/plots/noise_sigma_sweep.png) — both embedded in README.

## 7. Audit prompt (Layer 13)

> *"Play the role of a critical RL professor reviewing this project against
> EX05's spec + V3 § 19.1. Find 10–20 weaknesses categorised as
> Critical / Important / Nice-to-have. For each: file, line, fix."*

14 findings; 2 Critical fixed in Layer 13 (reproducibility test + CI), 3
Important closed, 9 Nice-to-have moved to PLAN.md § 14 extension points.

## 8. Patterns that worked

| Pattern | Why it worked |
|---|---|
| PRD/PLAN/TODO before code | Same ~50% iteration savings as A4 |
| TDD for math layers | Polyak and DDPG-update both green on first run |
| Lazy CLI imports | `roomba-lab --help` < 200 ms despite torch + PyQt6 deps |
| `Layer N: <summary>` commits | Git log reads as build story; clean reverts |
| ADR-007 (experiments under sdk/) | Avoided the A4 Layer 17 layering-violation refactor |
| Real upstream data loader from day 1 | No "fake apartment" placeholder to throw away later |

## 9. Patterns to avoid (carried lessons)

| Anti-pattern | Why we didn't fall into it |
|---|---|
| Putting experiments under services/ | ADR-007 explicitly forbids it |
| Mocking the simulator in tests | Integration tests use a real HouseExpo apartment |
| Hard-coding LIDAR beams or robot radius | Every value in `configs/setup.json` |
| Single-seed sweep | Layer 11 multi-seeded from the start |
| Trying to ship before reproducibility test | Layer 13 added it before final docs |

## 10. Estimated effort

| Phase | Layers | Wall-clock | Human review % |
|---|---|---|---|
| Planning (PRD/PLAN/TODO/per-mechanism PRDs) | 0 | ~30 min | 100 % |
| Scaffolding | 0 cont. | ~15 min | 80 % |
| Shared + data + simulator | 1 + 2 | ~40 min | 50 % |
| Env + LIDAR + reward | 3 | ~30 min | 60 % |
| Networks + soft update | 4 | ~25 min | 50 % |
| Buffer + noise | 5 + 6 | ~20 min | 40 % |
| DDPG update + service | 7 + 8 | ~40 min | 50 % |
| SDK + CLI | 9 | ~30 min | 30 % |
| Graphify port + sweeps | 10 + 11 | ~40 min | 30 % |
| Visualisations + audit | 12 + 13 | ~35 min | 60 % |
| GUI + notebook | 14 + 15 | ~30 min | 30 % |
| Final docs + reflection answers | 16 | ~45 min | 80 % |
| V3 polish | 17 | ~20 min | 90 % |
| **v1.10 polish (TD3 + lessons + cross-apt)** | 18-20 | ~90 min | 70 % |
| **v1.20 TA-audit cycle** | 21-26 | ~120 min | 80 % |
| **v1.21 TA re-grade follow-up** | 27 | ~40 min | 90 % |
| **v1.22 substantive M1/M5 closure** | 28 | ~60 min | 80 % |

**Total wall-clock: ~10 hours**. The v1.20 cycle was driven by an explicit
adversarial-review pass (role-playing the grader) — see
[`docs/LESSONS_LEARNED.md`](LESSONS_LEARNED.md) § 9 for why this is a
high-leverage exercise.

## 15. Iterative adversarial review (v1.20 → v1.21 → v1.22)

The TA-role-play pattern from § 14 was applied **3 times in sequence** for the
post-v1.00 polish cycles:

1. **First review** (after v1.10) → identified 14 findings (5 Major + 8 Mod + 9 Minor)
   → drove the **v1.20** polish layer (Layers 21-26)
2. **Second review** (after v1.20) → caught 6 partial-closure issues + 1 NEW1
   finding (README CIs hand-edited) → drove **v1.21** (Layer 27)
3. **Third review** (after v1.21) → caught 3 NEW doc-drift items + held 3
   prior partials → drove **v1.22** (Layer 28)

**Convergence behaviour**: each review found ~70 % fewer items than the
previous. The grade trajectory: 82 → 91 → 92.5 → 95+ (targeted).

**Lesson**: adversarial review is not single-pass. Each fix introduces new
opportunities for drift (intro lines stale, cost docs incomplete) and reveals
fixes that were partial (TA's "Mod6 GIF unwatched" got "fixed" with a
re-recorded GIF, but the v1.21 review caught that we hadn't extracted frames
for visual verification). **Run the TA review after every major version bump**;
expect 2-3 iterations to converge.

## 14. The adversarial-review prompt (v1.20 cycle)

For the v1.20 polish I gave the agent a different role:

> *"Roleplay as my professor's grading agent. Read the submission and find
> things to REDUCE GRADE about. Don't be polite. Be specific — file:line,
> what's wrong, what would lose points."*

The agent produced a 14-finding report categorised Major / Moderate / Minor,
with a 82/100 provisional grade and clear remediation steps. I then switched
the agent back to implementer mode with the explicit goal of pushing toward
100/100, layer-by-layer, keeping all docs aligned.

Pattern: **before submitting, role-play the grader explicitly.** Save the
findings document; use it as the v1.x → v1.(x+1) target list. Re-run on every
major version bump.
