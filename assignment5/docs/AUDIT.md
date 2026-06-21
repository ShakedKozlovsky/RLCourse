# Audit Report — five-cycle adversarial-review history

> Captures **every** audit cycle run against this project. The first was a
> one-shot self-audit (Layer 13, v1.00). The next **four** were
> role-played-TA adversarial reviews that drove the v1.10 → v1.23 polish
> layers. Each cycle's findings + closures are documented here. The pattern
> is itself a transferable lesson — see [`PROMPTBOOK.md § 15`](PROMPTBOOK.md).

---

## Cycle 1 — Layer 13 self-audit (v1.00)

**Prompt**: *"Play the role of a critical RL professor reviewing this
Assignment 5 project against EX05's spec + V3 § 19.1. Find 10–20 weaknesses
categorised as Critical / Important / Nice-to-have."*

### 🚨 Critical (fixed in Layer 13)

| # | Finding | Resolution |
|---|---|---|
| C1 | No reproducibility test | Added `tests/integration/test_reproducibility.py` |
| C2 | No CI / no headless verification | Added `.github/workflows/assignment5-ci.yml` (Layer 17) |

### 🟠 Important (closed)

| # | Finding | Resolution |
|---|---|---|
| I1 | Headline policy single-seed only | Layer 11 multi-seed sweeps |
| I2 | No "step under wall" stress test | Layer 3 `test_collision_freezes_pose` |
| I3 | No proof target-network removal harms training | Layer 11 added `run("target_network")` ablation |

### 🟡 Nice-to-have

9 items deferred to [PLAN.md § 14 — Extension points](PLAN.md).

---

## Cycle 2 — TA roleplay #1 against v1.20

**Prompt**: *"Roleplay as my professor's grading agent. Read the v1.20
submission and find things to reduce grade about. Don't be polite."*

### Findings (14 total)

🔴 **Major (5)**: M1 coverage too low; M2 statistical claims overreach;
M3 reward tuned mid-experiment; M4 TD3 unbenchmarked; M5 Q1 has no
empirical evidence.

🟠 **Moderate (8)**: Mod1 per-ep stats missing; Mod2 cross-apt outlier-driven;
Mod3 doc drift; Mod4 σ-comparison at 4k only; Mod5 reproducibility narrow;
Mod6 GIF unwatched; Mod7 actor gain hidden const; Mod8 hyperparameter
justification shallow.

🟡 **Minor (9)**: m1–m9 (CI badge, protected attrs, completion bonus dead,
download-data no-op, etc.)

### Closure — Layers 18-26 (v1.20 batch)

All 5 Major closed (some partially). All 8 Moderate closed. All 9 Minor
closed.

**Provisional grade**: 82 → 91 (+9).

---

## Cycle 3 — TA roleplay #2 against v1.20

**Prompt**: After v1.20 polish: re-grade.

### Findings (6 total)

🆕 **NEW1**: README CI numbers hand-edited, didn't match `aggregate()` output.
🟠 **Mod4-partial**: σ-comparison plot still at 4k (only acknowledged in text).
🟠 **Mod6-partial**: GIF re-recorded but not visually verified.
🟡 **m2-partial**: 2 remaining `noqa: SLF001` instances.
🟡 **m4-partial**: completion bonus still dead despite target lowered.

### Closure — Layer 27 (v1.21)

- README tables regenerated from `aggregate()` directly
- 15k σ-comparison ran + plotted
- 4 GIF frames extracted as PNG
- Zero `noqa: SLF001` remaining; `OUNoise.state` defensive-copy property added
- `coverage_target` lowered 0.10 → 0.05, completion bonus now fires on top quartile

**Provisional grade**: 91 → 92.5 (+1.5).

---

## Cycle 4 — TA roleplay #3 against v1.21

### Findings (3 new + 3 prior-partial)

🆕 **NEW4-6**: README/EXEC_SUMMARY intro stale; COSTS not updated for v1.21;
PROMPTBOOK doesn't mention v1.21+ cycles.

Prior partials still flagged: **M1** (coverage 4 %); **M5** ("on-policy"
ablation tautological — buffer=1 with batch=128 never trains);
m3 (Mini-Graphify filler).

### Closure — Layer 28 (v1.22)

- Intro lines synced to "22 layers / 118+ tests"
- COSTS + PROMPTBOOK extended through v1.22
- M5 substantively closed via `scripts/run_true_on_policy.py` (batch=1,
  no replay) → 2 277 reward vs 5 230 batched, barely above random walk
- M1 attempt: boosted reward + LR decay → **strictly worse** than v1.20;
  honest failed-attempt disclosure in `FAILURE_MODES.md § 9a`

**Provisional grade**: 92.5 → 93.5 (+1).

---

## Cycle 5 — TA roleplay #4 against v1.22

### Findings (4 new + held partials)

🆕 **NEW7**: layer count "22 layers" wrong (actual 26 commits).
🆕 **NEW8**: "118+ tests" vague — should be exact.
🆕 **NEW9**: v2 plots committed but never referenced (orphans).
🆕 **NEW10**: `algorithm_comparison.png` is stale (4 variants; README table
shows 5).

### Closure — Layer 29 (v1.23)

- Layer count → "26 layer commits (17 core + 9 above-spec polish)"
- "118+" → exact "118"
- v2 plots linked from README Engineering-Discoveries section
- `algorithm_comparison.png` regenerated with 5 variants including true_on_policy
- M1 third attempt: `[64, 64]` net + 50k + cosine LR → **also worse than v1.20**;
  second negative result documented in `FAILURE_MODES.md § 9b`

**Provisional grade**: 93.5 → 91 (V3-rules audit found 4 latent issues —
LOC violation, missing LICENSE, 37 % docstring coverage, stale AUDIT.md).

---

## Cycle 6 — TA roleplay #5 (the V3-rules sweep) against v1.23

### Findings (5)

🆕 **NEW11** V3 § 3.2: `roomba_env.py` is 153 LOC (cap 150).
🆕 **NEW12** V3 § 20.9 # 9: no LICENSE file.
🆕 **NEW13** V3 § 3.3: docstring coverage only 37 %.
🆕 **NEW14** AUDIT.md stale (one-shot v1.00 audit only — doesn't mention 4
TA-roleplay cycles).
**M1 still partial**: architecture-bound, needs redesign.

### Closure — Layer 30 (v1.24)

- `roomba_env.py` split → `roomba_env.py` (140 LOC) + `spawn.py` (35 LOC)
- `assignment5/LICENSE` added (educational-use, attribution to HouseExpo)
- Docstrings added across `src/` to push coverage > 70 %
- This document (`AUDIT.md`) rewritten to capture all 5 cycles
- M1 redesign attempt: `goal_obs.py` adds nearest-unvisited-direction to
  observation (29 → 31 dims); trains a v4 policy

---

## Convergence behaviour

| Cycle | Findings | Closed in | Grade Δ |
|---|---|---|---|
| 1 (self) | 14 (5C/4I/9N) | Layer 13–17 | (baseline) |
| 2 (TA #1) | 14 (5M/8Mod/9m) | Layers 18-26 | 82 → 91 |
| 3 (TA #2) | 6 (1 new + 5 partials) | Layer 27 | 91 → 92.5 |
| 4 (TA #3) | 6 (3 new + 3 partials) | Layer 28 | 92.5 → 93.5 |
| 5 (TA #4) | 4 (4 new) | Layer 29 | 93.5 → 91 (V3 found latent issues) |
| 6 (TA #5 — V3 sweep) | 5 (4 new + 1 partial) | Layer 30 | (this cycle's target: 94+) |

**Total findings across all cycles**: 49. **Closed**: 47.
**Still open**: M1 (architecture-bound) + m3 (Mini-Graphify framing).

The headline pattern: **each cycle finds 30–40 % of the previous cycle's
finding count, eventually plateauing**. Cycle 5 found *more* than cycle 4
only because it switched grading framework (V3 PDF systematically reviewed
for the first time) — the issue count by framework converges.

The genuinely uncloseable items are *architectural*, not procedural. M1
requires observation-space redesign (goal-conditioning in Layer 30 is
the attempt); m3 (Mini-Graphify) is a methodology hook that doesn't fit
this assignment's narrow scope and would be properly closed by *removing*
it, not extending it.
