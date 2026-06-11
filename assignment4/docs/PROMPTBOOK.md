# Promptbook — AI-assisted development log

> Required by V3 § 8.3 / § 17.1 / § 20.9 #1. This document captures the **prompt-engineering methodology** used to build the project with an AI agent (Claude Opus 4.7) acting as the implementer under a single human Software Architect.
>
> The methodology is the "Vibe Coding" approach from V3 § 1.4: define requirements clearly **before any line of code**, then delegate implementation while reviewing each layer.

## 0. Meta-rule (the most important prompt)

Every other prompt below is downstream of this one. Used at session start:

> *"Use the V3 coding-rules PDF as your reference. ≤ 150 LOC per file, ruff clean, ≥ 85 % branch coverage, uv only, no magic numbers, layered architecture (interface → sdk → services → ...), `Layer N: <summary>` commit messages. Before writing any code, produce `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, and per-mechanism PRDs. Only start coding after I approve the docs."*

This single sentence enforces ~80 % of the V3 rules automatically.

## 1. Planning phase prompts (Layer 0)

### 1.1 Domain framing

> *"Read the L08 slides (PPO + GAE) and the Active Knowledge Architecture methodology PDF. Propose 3 domain choices for the assignment, explain trade-offs for each, then recommend one."*

**Outcome**: chose MuJoCo continuous control over RLHF (too big), over domain-specific recommender (already done in Assignment 3).

### 1.2 PRD authoring

> *"Write `docs/PRD.md` covering: project goal, lecture-slide mapping (each slide → code location), the two core equations verbatim, environments, network architecture, hyperparameters, empirical study plan (λ/γ/ε sweeps), mini-Graphify originality hook, KPIs, deliverables."*

**Iteration**: revised three times — added the bias-variance table, the slide-by-slide mapping, the originality hook explanation.

### 1.3 PLAN with ADRs

> *"Write `docs/PLAN.md` with: layered architecture diagram, package map, class diagram, full PPO + GAE pseudocode, configuration schema, **8 ADRs** documenting design decisions with trade-offs."*

**Key insight**: ADRs are how we document *changes of mind* — when a Layer-N decision overrides a Layer-0 plan, the ADR explains why without rewriting history.

### 1.4 TODO with DoD per layer

> *"Write `docs/TODO.md` as a 16-layer task list. Each layer = one commit. Each layer must have explicit Definition of Done (DoD) covering: code + tests + docs."*

## 2. Scaffolding-phase prompts (Layer 0 continued)

### 2.1 Project skeleton

> *"Create the directory tree per V3 § 2.4. Add empty `__init__.py` files. Write `pyproject.toml` with the V3 ruff config (E, F, W, I, N, UP, B, C4, SIM), `fail_under = 85`, MuJoCo + PyTorch + PyQt6 + click + matplotlib deps. Verify `python -c 'import proximal_lab'` works."*

### 2.2 Configuration schema

> *"Design `configs/setup.json` schema. Every hyperparameter goes here — zero magic numbers in source. Include version field set to '1.00'. Make `ConfigManager` raise on version mismatch."*

## 3. TDD-phase prompts (Layers 4, 5 — the math layers)

### 3.1 Red-Green-Refactor for GAE

> *"Write `tests/unit/test_gae.py` FIRST with 4 cases: (1) λ=0 → GAE = TD error, (2) λ=1 → GAE = MC return − V, (3) closed-form 3-step trajectory with hand-computed expected values, (4) terminal handling: done zeroes bootstrap. Only after tests are written, implement `services/gae.py::compute_gae` to make them pass."*

This is the **headline TDD pattern**: math correctness lives in the test, not the doctring.

### 3.2 Red-Green-Refactor for PPO clip

> *"Write `tests/unit/test_ppo_clip.py` with 4 sign × clip-window cases (slides 11-12): Â>0 r∈window, Â>0 r>1+ε, Â<0 r>1+ε (the safety case — unclipped wins), Â<0 r<1-ε. Then implement `services/ppo_clip.py::ppo_clip_loss` to satisfy them."*

The "safety case" test (Â<0 r>1+ε ⇒ unclipped wins) is the central PPO intuition — if a test guards it, no future refactor can silently break it.

## 4. Layer-execution prompts (Layers 1–14)

### 4.1 Layer-N kickoff

> *"Layer N: [topic]. Reference the layer's TODO.md entry. Implement, write tests, run ruff + pytest + coverage. Commit with the documented `Layer N: <summary>` format + bullet body + Co-Authored-By footer. Push to main."*

### 4.2 LOC enforcement

> *"This file is at 168 LOC, over the 150 cap. Split by responsibility — what's the minimum-change refactor that keeps the public API stable?"*

Used 3 times during the build: SDK, PPOService, CLI.

### 4.3 Coverage gap closure

> *"Coverage at 73 %. Identify the uncovered branches and write targeted tests. Skip CUDA / pragma-no-cover paths."*

## 5. Empirical-study prompts (Layers 10, 13, 16d)

### 5.1 Sweep design

> *"Design a multi-seed λ-sweep. Cells = {0.0, 0.5, 0.9, 0.95, 0.99, 1.0}. 3 seeds per cell. Output JSON with `final_reward_mean` + `final_reward_ci_95`. Aggregate via normal-approx CI (z=1.96)."*

### 5.2 Statistical-significance framing

> *"The λ=0.95 cell beats both λ=0 and λ=1, but show it's not noise. Report mean differences vs CI sum and explicitly say which differences are outside the bands."*

## 6. Audit-phase prompts (Layers 13 + 17 + V3-PDF audit)

### 6.1 Self-audit (Layer 13)

> *"Play the role of a critical RL professor reviewing this project. Find 10–20 weaknesses categorized as Critical / Important / Nice-to-have. For each: specific file, specific line, specific fix."*

This yielded 20 findings; all closed.

### 6.2 V3-rules compliance audit

> *"Audit the project against the V3 PDF (file path provided). Report each rule with status. Flag any genuine violations honestly — do not gloss."*

Yielded the `services → sdk` layering violation, which became Layer 17's refactor.

## 7. Documentation-phase prompts (Layer 15)

### 7.1 README structure

> *"Write the final README with 15 sections: project goal, slide mapping, equations, environments, networks, training pipeline, headline empirical result, cross-env, audit response, GUI/CLI/SDK, Graphify section, quality bar, reflection answers, honest acknowledgements, sources."*

### 7.2 Executive summary

> *"Write a 1-page `docs/EXECUTIVE_SUMMARY.md` for the grader's first pass. Headline finding + KPI table + 'where to look first' list + honest acknowledgements."*

## 8. Originality-hook prompts (Layer 16)

> *"What 6 above-spec additions would push this from very good to memorable? Prioritize by impact-to-effort. I want: visual elements (GIF, diagram), an empirical proof, engineering signal, and a polished doc."*

Outcome: clipped-surrogate viz + GAE-as-advantage ablation + architecture diagram + CI badge + policy GIF + reproducibility statement.

## 9. Patterns that worked well

| Pattern | Why it worked |
|---|---|
| **PRD/PLAN/TODO before code** | The agent could read its own contract; saved ~50 % iteration time |
| **TDD with explicit math** | Math tests are unambiguous specs. No "but the function looks right" arguments |
| **`Layer N: <summary>` commits** | Git log reads as a build story. Easy to revert one layer cleanly. |
| **ADRs for changes of mind** | Layer-17 wasn't "fixing a bug" — it was "documenting an architectural shift" |
| **Periodic self-audit** | Forced finding gaps before the grader does |

## 10. Patterns to avoid

| Anti-pattern | What happens |
|---|---|
| *"just implement X"* without spec | Agent guesses; iteration cost balloons |
| Single-mega-prompt | Loses focus; partial implementations |
| No coverage gate | Easy to ship 60 %-covered code without noticing |
| Skipping CI early | Layer 16's CI surfaced 3 bugs the local dev machine missed (libEGL, PyOpenGL, flaky KL assertion) |

## 11. Tooling layered on top

- **Claude Code** as the implementing agent
- **uv** for env + lockfile
- **pytest + ruff** as quality gates
- **GitHub Actions** for headless CI verification
- **NotebookLM** (referenced by the lecturer) for methodology framing

## 12. Estimated effort

| Phase | Layers | Wall-clock | Human review % |
|---|---|---|---|
| Planning (PRD/PLAN/TODO) | 0 | ~30 min | 100 % |
| Scaffolding | 0 cont. | ~20 min | 80 % |
| Math layers (TDD) | 3 + 4 + 5 | ~60 min | 60 % |
| End-to-end PPO + eval | 6 + 7 | ~45 min | 40 % |
| SDK + CLI + GUI | 8 + 14 | ~60 min | 30 % |
| Empirical sweeps | 10 + 11 + 13 | ~90 min | 30 % |
| Originality (Graphify) | 9 | ~30 min | 70 % |
| Audit + docs | 12 + 15 + 16 + 17 | ~120 min | 80 % |
| CI red-pen cycles | 3 fix commits | ~20 min | 100 % |

**Total wall-clock: ~7-8 hours**. Compare to V3 § 1.4's claim ("≥ 16× speedup over solo coding") — this matches the spec for a project of this scope.

## 13. The single biggest lesson

The V3 PDF's § 1.4 stake-claim: *"The first and most important rule of professional coding with AI agents: define requirements clearly and write complete documentation **before** any line of code."*

The session that produced this project followed that rule. The session that almost missed it was Layer 16 (originality hook) — I prompted with *"What more should we add?"* without writing a PRD first, then partially backed into one via the conversation. It worked but was less clean than the layers with explicit pre-written PRDs (especially Layer 9 — Graphify — which had a 132-line `PRD_graphify.md` written first).

The lesson generalises: **for any non-trivial addition, write the PRD first** even when the project is "done."
