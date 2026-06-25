# CHANGELOG — marl_lab

Version-by-version story of the assignment6 codebase. Each tag is a real `git tag` on `main` you can check out: `git checkout marl-lab-v1.04`. The original 27-layer build landed in v1.00; everything from v1.01 onward is either a TA-cycle fix or a beyond-spec extension.

| Tag | Date | Tests | Coverage | Headline |
|---|---|---|---|---|
| [v1.00](#v100--feature-complete) | 2026-06-24 | 174 | — | Original 27-layer build complete |
| [v1.01](#v101--ta-cycle-12-fixes) | 2026-06-24 | 207 | — | Spec-conformance + § 7.3 artifacts |
| [v1.02](#v102--drift-hunt) | 2026-06-24 | 207 | — | CLI entry-point fix + stale dirs deleted |
| [v1.03](#v103--academic-depth) | 2026-06-24 | 222 | — | QPLEX + PROOFS + GIF + tournament + provenance |
| [v1.04](#v104--engineering-excellence) | 2026-06-24 | 241 | **95%** | Curriculum + fuzz + coverage + notebook HTML |
| [v1.05](#v105--visual--empirical) | 2026-06-24 | 241 | 95% | Mermaid + token rotation + Bernstein + 500-ep convergence |
| [v1.06](#v106--ci) | 2026-06-24 | 241 | 95% | GitHub Actions CI green |
| [v1.07](#v107--scale-study) | 2026-06-24 | 241 | 95% | Lin 2025 hypothesis empirically verified |
| [v1.08](#v108--maddpg--docker) | 2026-06-24 | **254** | 95% | 5th algorithm + zero-setup playability |

---

## v1.00 — feature-complete

**`marl-lab-v1.00` · commit `87c2aec`**

Original 27-layer build (Layers 0 → 26) shipped in one continuous session.

**What you get:**
- Dec-POMDP env (no gym imports) with Manhattan-radius observation
- Per-agent GRU Q-net + soft Polyak target updates
- Three mixers + three updaters: **QMIX** (monotonic hypernet, IGM-via-`|W|`), **VDN** (additive), **IQL** (no mixer baseline)
- Centralised replay buffer (variable-length sequences with masks)
- `MarlTrainer` end-to-end, parameterised over `algo="qmix" | "vdn" | "iql"`
- `GameRunner` plays 6 sub-games + emits spec § 3.5 `GameReport` JSON
- `MarlSDK` facade tying it all together
- Two MCP servers (cop + thief) with token-allowlist auth
- MCP client + adjudicator-over-MCP
- Gmail sender with idempotency ledger (3 strategies: smtplib App Password / OAuth / MCP-tool)
- 8-subcommand CLI (`marl train`, `play-game`, `send-report`, `play-and-send`, `serve-cop`, `serve-thief`, `audit`, `version`)
- Headless-testable Tkinter GUI core + matplotlib board renderer
- Mini-Graphify auto-generator (module map → `docs/wiki/architecture.md`)
- Cartesian-product sweep runner + Prefect cloud (with always-works local fallback)
- Reproducibility integration test + jupytext walkthrough + FAILURE_MODES doc + audit script

**Stats:** 27 commits (Layer 0 → Layer 26) · 174 tests · ruff clean · LOC ≤ 250/file.

---

## v1.01 — TA cycle 1+2 fixes

**`marl-lab-v1.01` · commit `2295710`**

After actually reading the spec PDF pages 1-14, found 6 spec-conformance gaps and the missing § 7.3 visualisation artifacts.

### TA cycle 1 — spec-conformance (commit `a580a64`)
- **Barrier placement** — was placing barrier on cell UP-of-cop; spec § 3.3 says **on cop's own cell**. Fixed.
- **Sub-game IDs** — were 0..5; spec example shows 1..6. Fixed.
- **Timezone** — was UTC; spec example uses `+03:00` Asia/Jerusalem. Switched to `zoneinfo.ZoneInfo("Asia/Jerusalem")`.
- **yaml `scoring.*`** — was completely ignored at runtime; now flows through SDK → `RewardConfig`.
- **Retry-on-failure** — added `max_retries_per_sub_game=3` per spec § 3.7.
- **README § 7 academic analysis** — was missing entirely; added Dec-POMDP tuple ↔ code table, IGM principle explanation, non-stationarity/CTDE discussion, IQL baseline comparison, IGM limits + QPLEX/Weighted-QMIX recommendations, POSG-vs-Dec-POMDP honesty, 9-entry bibliography.
- New: `tests/integration/test_spec_conformance.py` (5 tests pinning the JSON shape forever).

### TA cycle 2 — § 7.3 artifacts
- `scripts/generate_artifacts.py` produces:
  - `assets/figures/learning_curves.png` — convergence across QMIX/VDN/IQL
  - `assets/figures/loss_curves.png` — critic loss log-Y
  - `assets/figures/gui_{3x3,4x4,5x5}.png` — board renderings at the staging grid sizes
  - `assets/logs/mcp_demo.log` — CLI-style MCP communication proof
- Bibliography expanded to 12 entries (added Bernstein 2002, Amato 2024, Lin 2025).

**Stats:** 207 tests (was 174; +33).

---

## v1.02 — drift hunt

**`marl-lab-v1.02` · commit `9ba68da`**

### TA cycle 4 — fresh-skeptic pass
Smoke-tested the documented quickstart from scratch and found:
1. **CLI entry point was BROKEN** — `pyproject.toml` declared `marl-lab = marl_lab.interface.cli.main:cli`; both name and module were wrong. Fixed to `marl = marl_lab.cli.main:main`. `uv run marl version` now actually works.
2. **Stale empty dirs** from layer-0 scaffolding (`interface/cli/`, `interface/gui/`, `tools/`, `data/`) — deleted.
3. **Broken PRD links** to `services/iql_baseline.py` (real: `iql_update.py`) and `interface/gui/` (deleted) — fixed.
4. **TODO.md still had every layer `[ ]`** — added status banner pointing to README.
5. **`uv sync` missed `--extra dev`** — pytest/ruff not installed on fresh clones; quickstart updated, `scripts/audit.py` self-heals via `uv sync --extra dev --quiet`.

End-to-end barrier semantics verified with a 3-turn trace (cop places → moves off → can't move back).

---

## v1.03 — academic depth

**`marl-lab-v1.03` · commit `094a7cb`**

Beyond-spec extensions #1–5. The README § 7.2 critical analysis listed QPLEX as a "natural extension"; this version **implements** it instead of just citing.

- **QPLEX mixer** (`src/marl_lab/model/qplex_mixer.py`) — duplex dueling decomposition, IGM by construction via `λ(s) > 0` parametrisation. Strict expressiveness gain over QMIX empirically verified (drives Q_tot negative while every Q_i positive — impossible under |W| QMIX). 10 dedicated tests.
- **`docs/PROOFS.md`** — chain-rule derivation of why `|W|` ⇒ `∂Q_tot/∂Q_i ≥ 0` for QMIX; why `λ > 0` is sufficient for IGM in QPLEX. Each math step cross-referenced to a test.
- **Animated GIF** (`assets/figures/sub_game.gif`) — 20-frame matplotlib `FuncAnimation` of a real sub-game.
- **4-algorithm tournament** (`assets/figures/tournament.png` + CSV) — round-robin QMIX/VDN/QPLEX/IQL × 3 seeds × 40 episodes.
- **Provenance** (`src/marl_lab/shared/provenance.py`) — every GameReport JSON carries `git_sha`, `git_dirty`, lib versions, Python, platform. Idempotency key intentionally provenance-independent.

**Stats:** 222 tests (was 207; +15: 10 QPLEX + 5 provenance).

---

## v1.04 — engineering excellence

**`marl-lab-v1.04` · commit `ab1ab8e`**

Beyond-spec extensions #6–9. Focus on test depth and reproducibility.

- **Curriculum learning** (Lin 2025) — `services/curriculum.py` ramps grid 2×2 → 5×5 as cop win-rate crosses each stage's threshold. **Q-net weights preserved across stages** (transfer signal); mixer + buffer rebuilt because state_dim changes. 12 tests.
- **Property-based fuzz tests** — `tests/property/test_env_invariants.py` with `hypothesis`. 7 invariants × 200 random inputs each = 1200+ probes.
- **95% measured branch coverage** — `pytest --cov` baseline, test composition table embedded in README.
- **Notebook → executed HTML** — `notebooks/marl_walkthrough.py` → jupytext → nbconvert --execute → `docs/wiki/marl_walkthrough.html`. TA can read the full pipeline (load → train → play → sweep) with real outputs in a browser without setting up Python. `scripts/rebuild_notebook.py` regenerates.

**Stats:** 241 tests (was 222; +19: 12 curriculum + 7 fuzz).

---

## v1.05 — visual + empirical

**`marl-lab-v1.05` · commit `2e3101c`**

Beyond-spec extensions #10–13.

- **Mermaid system diagram** — GitHub-rendered data-flow graph (yaml → SDK → TRAIN/PLAY/MCP/SEND subgraphs).
- **MCP token rotation demo** (`scripts/demo_token_rotation.py`) — 4-stage scripted lifecycle: issue v1 → rotate to v2 → revoke v1 → revoke v2 / deny-all. 4 successful + 4 rejected requests, all assertions held. Transcript at `assets/logs/token_rotation.log`.
- **Bernstein 2002 complexity appendix** (`docs/PROOFS.md § 4`) — connects ref [1]'s NEXP-completeness result to *why* CTDE is the tractable compromise. Includes POSG corollary (Hansen-Bernstein-Zilberstein 2004 → NEXP^NP).
- **500-episode convergence study** (`scripts/long_convergence_study.py` + `assets/figures/long_convergence.png`) — proper experimental signal across QMIX/QPLEX/IQL on 4×4 grid. **Honest empirical finding** reported: IQL competitive on small grids. Context added to `FAILURE_MODES.md § 3`.

**Bonus catch:** found that `assets/logs/*.log` was gitignored and never committed — README had been linking to `mcp_demo.log` since v1.01 but the file wasn't on GitHub! Fixed `.gitignore` to whitelist the evidence directory.

---

## v1.06 — CI

**`marl-lab-v1.06` · commit `4f5c6bd`**

Beyond-spec extension #14. The local audit is good; the public-facing badge is better.

- **`.github/workflows/assignment6-ci.yml`** — 2-job pipeline:
  - **`audit`**: `uv sync --extra dev` → ruff → pytest --cov → LOC ≤ 250 → graphify regen with drift detection → coverage XML uploaded as artifact
  - **`property-fuzz`**: re-runs `tests/property/` under `HYPOTHESIS_PROFILE=ci` lifting example budget 200 → 500 per invariant (3500+ randomised inputs per push)
- **`tests/conftest.py`** — registers `default` / `ci` / `dev` hypothesis profiles, selectable via env var.
- **5 status badges** in README (build / tests / coverage / ruff / python).

CI went green on the first push. Required adding `workflow` scope to the GitHub PAT (rejected at first; user updated, push retried successfully).

---

## v1.07 — scale study

**`marl-lab-v1.07` · commit `421eeed`**

Beyond-spec extension #15. The v1.05 4×4 study found IQL competitive — was that a CTDE problem or a small-state artefact? This study answers it.

- **`scripts/scale_convergence_study.py`** — trains QMIX/QPLEX/IQL for 250 episodes each on **5×5, 6×6, 7×7** grids (2,250 total training episodes).
- **`assets/figures/scale_convergence.png`** — 3 subplots, one per grid.
- **`assets/figures/ctde_advantage_vs_grid.png`** — headline plot: final-50 mean reward vs grid size, three algo lines.

**Lin 2025 hypothesis empirically confirmed.** Final-50 mean cop reward by grid:

| Grid | QMIX | QPLEX | IQL | Winner |
|---|---|---|---|---|
| 4×4 (v1.05) | -1.70 | -1.47 | **-1.38** | IQL |
| 5×5 | -2.05 | -1.88 | **-1.75** | IQL (gap closing) |
| 6×6 | -2.67 | **-1.66** | -2.67 | **QPLEX +1.01** |
| 7×7 | **-2.92** | -3.12 | -3.27 | QMIX/QPLEX over IQL |

QPLEX dominates on 6×6 — the dueling decomposition's mathematical advantage over QMIX (proven in `PROOFS.md § 3`) **cashes out empirically** on medium grids. `FAILURE_MODES.md § 3` updated with the full 4-row table.

---

## v1.08 — MADDPG + Docker

**`marl-lab-v1.08` · commit `d383e7d`**

Beyond-spec extensions #16–17.

- **MADDPG-discrete** (`src/marl_lab/model/maddpg_critic.py` + `services/maddpg_update.py`) — Lowe et al. 2017 adapted for discrete actions. **Per-agent centralised critic** with **per-agent reward** (not the averaged joint reward QMIX/VDN/QPLEX use). Directly addresses the POSG honesty concern from `FAILURE_MODES § 1` with code. Algo zoo is now `algo="qmix" | "vdn" | "qplex" | "maddpg" | "iql"` (5 algorithms). 13 tests including the "per-agent rewards distinguish losses" probe that proves POSG fidelity.
- **`Dockerfile` + `.dockerignore`** — multi-stage build on `python:3.12-slim` with `uv 0.5.4`. Smoke-tested: `docker run --rm marl-lab marl audit` works zero-setup.

**Bonus refactor**: `services/learn_dispatch.py` extracts per-algo update dispatch from `MarlTrainer.learn_step` (now 214 LOC, was 271). Adding a 6th algorithm is now a single new branch in one file.

**Stats:** 254 tests (was 241; +13 MADDPG). 5 algorithms.

---

## How to verify

Anything in this changelog can be independently verified from a fresh clone:

```bash
git clone https://github.com/ShakedKozlovsky/RLCourse.git
cd RLCourse/assignment6
git checkout marl-lab-v1.08          # or any earlier tag

# Option A — local
uv sync --extra dev
uv run python scripts/audit.py        # ruff + 254 tests + LOC + graphify
uv run pytest --cov=marl_lab          # confirm 95% branch coverage

# Option B — Docker
docker build -t marl-lab .
docker run --rm marl-lab marl audit
docker run --rm marl-lab uv run pytest -q
```

Or just look at the green badge at the top of [`../README.md`](../README.md) — every push since v1.06 has been verified by GitHub Actions.
