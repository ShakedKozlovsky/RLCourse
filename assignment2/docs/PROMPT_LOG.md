# Prompt Engineering Log

> Documents the significant prompts used during AI-assisted development of this project, per the coding guidelines V3 §8.3.

## Tools and models used

| Tool | Model | Role |
|---|---|---|
| Claude Code (CLI + VS Code extension) | Claude Opus 4.7 | Primary development partner — code generation, testing, documentation, experiment analysis |

## Prompt categories

### 1. Scope definition and planning

**Prompt (paraphrased):** "We have a project to do. I'm attaching three PDFs — assignment instructions (Hebrew), lecture slides, and coding guidelines. We need to follow everything. Create PRD, TODO, README. Reference the ex1 files in RLCourse for structure."

**What the AI produced:** PRD.md, PLAN.md, TODO.md, 8 per-mechanism PRDs, project scaffold with all `__init__.py` files, `pyproject.toml`, configs, `.gitignore`.

**Iteration:** The AI proposed choices (DQN variant, interface type, approach) and the student made selections. This upfront alignment saved significant rework later.

**Lesson learned:** providing all three reference documents (assignment, slides, coding rules) in one prompt gives the AI enough context to produce a plan that cross-references all three — rather than having to go back and add slide references later.

### 2. Layer-by-layer implementation

**Prompt pattern (repeated 9 times):** "continue" or "ok" — triggering the next layer from the TODO.md plan.

**What the AI did each time:** wrote source modules, wrote tests, ran pytest + ruff + coverage, fixed issues, updated docs, committed with a descriptive message.

**Iteration example (Layer 1):** The first integration test failed because the per-slice warmup dropped too many rows from val/test. The AI diagnosed the root cause (feature computation order), proposed ADR-007, implemented the fix, re-ran tests, and committed. This entire cycle was unprompted — the test failure drove the fix.

**Lesson learned:** having a clear TODO with per-layer Definition of Done means "continue" is a sufficient prompt — the AI knows what to build next and how to verify it.

### 3. Full experiment execution

**Prompt:** "do everything" — meaning: fetch real data, train, run all experiments, generate plots, capture screenshots, update README.

**What the AI did:** fetched AAPL + SPY, reduced training episodes for time budget, created driver scripts under `scripts/`, ran 8 conditions in parallel, discovered the backtest-overwriting bug and patched it mid-session, generated 14 PNG artefacts, embedded them in README.

**Iteration:** the backtest naming bug was discovered when the AI noticed all conditions wrote to the same `test_backtest.npz`. It patched `sdk.py` and `experiment_service.py` with a `report_name` parameter, re-ran backtests cheaply (no retraining), and added a test covering the new parameter.

**Lesson learned:** "do everything" works when the codebase has a clear SDK boundary — the AI can orchestrate multi-step workflows through the same facade the user would.

### 4. Quality compliance

**Prompt:** "The professor gave these important notes: (8 numbered points about coverage, OOP, coding rules, docstrings, reports, relative grading, token costs)."

**What the AI did:** audited each point, identified gaps (missing docstrings, missing research report, missing token costs), proposed a prioritised remediation plan, and began executing.

**Lesson learned:** giving the AI the *professor's rubric* as an explicit checklist produces a targeted audit — more useful than a vague "check if everything's good."

## Prompts that didn't work well

| Prompt | Problem | Fix |
|---|---|---|
| Running pytest inside `env -u VIRTUAL_ENV uv run pytest` | The system `pytest` was picked up instead of the venv's | Used `.venv/bin/python3 -m pytest` or ensured VIRTUAL_ENV was not set |
| `np.arange(total, dtype="datetime64[D]")` in test fixtures | NumPy 2.x requires explicit start for datetime aranges | Replaced with explicit list comprehension |
| `lr=1e-2` in the DQN backward test | One-step overshooting on random targets | Changed to 50 steps at lr=1e-3 for a reliable convergence check |

## Summary statistics

- **Total prompts from the student:** ~15 substantive messages over one session
- **Total AI turns:** ~100+ (including tool calls, test runs, and iterative fixes)
- **Percentage of code written by AI:** ~95%
- **Percentage of *decisions* made by student:** ~40% (scope, variant choices, "do everything", priority calls)
- **Bugs discovered by tests, not by the student or AI code review:** 3 (ADR-007, ADR-008, np.arange datetime)
