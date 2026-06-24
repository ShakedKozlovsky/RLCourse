# PRD — Game rules + sub-games + barriers

> Per-mechanism PRD for the **game logic** layer (`src/marl_lab/game/`). Spec § 3 is the contract.

## 1. The Cops-and-Robbers framing (L10 § 2.2)

A pursuit-evasion problem on a grid. **One cop** chases **one thief**. Each agent has partial (Manhattan-radius) visibility. Win conditions are per-sub-game (see § 3 below). The full **game** is **6 sub-games**; one Gmail report is sent after the game ends.

## 2. Game ≠ Sub-game

This is the *most-confused* terminology in the spec:

| Term | Meaning | Count |
|---|---|---|
| **Sub-game** | One round between cop and thief; capped at 25 moves | 6 per game |
| **Game** | The full sequence of 6 sub-games | 1 per Gmail report |

The Gmail report contains one entry per sub-game + a `totals` row.

## 3. Win conditions (spec § 3.2)

| Sub-game outcome | Cop score | Thief score |
|---|---|---|
| Cop catches thief (same cell) within 25 moves | **+20 (win)** | **−5 (loss)** *recorded as 5* |
| 25 moves elapse without capture | **−5 (loss)** *recorded as 5* | **+10 (win)** |

*Note*: the spec lists "thief loss" and "cop loss" both as +5 (positive). Interpretation: it's the **score that gets recorded** (positive integer for the loss row), not a negative reward. The reward function (per-agent, used for learning) is separate from the report-card scoring (used in the Gmail JSON).

## 4. Board + moves

- Grid: H × W (default 5 × 5; configurable 2 × 2 … 5 × 5 per spec § 5.1 Table 2).
- Each agent occupies one cell.
- Cop and thief start on **different cells** at sub-game start (random valid).
- One **move** per agent per turn (simultaneous). Five base actions: `UP / DOWN / LEFT / RIGHT / STAY`. Cop has a sixth: `PLACE_BARRIER` (advanced option, see § 5).
- Illegal moves (off-grid / into a barrier) are no-ops; the move counter still advances.

## 5. Barriers (advanced option — spec § 3.3)

- The cop may, instead of moving, place a **barrier** on a cell they **do not currently occupy**.
- Barriers are static obstacles for the rest of the game (the *game*, not the sub-game — they persist across all 6 sub-games of a single game).
- Maximum **5 barriers** per game.
- Toggleable via `configs/setup.yaml::game.enable_barriers`.

Trade-off: spending a turn placing a barrier costs movement but constrains the thief's escape topology.

## 6. Scoring + totals

Per the spec § 3.4 Table 1 (defaults; configurable):

```yaml
scoring:
  cop_win: 20
  thief_win: 10
  cop_loss: 5
  thief_loss: 5
```

The `totals` row at the end of the report sums per-agent scores across all 6 sub-games.

## 7. Technical losses (spec § 3.7)

Sub-games that fail to complete because of a technical fault (e.g., MCP timeout, network error) are **not counted** — they must be replayed. The `game_runner` automates this: tracks the retry count + writes diagnostic logs.

## 8. Test plan

| Test | Pass criterion |
|---|---|
| Board init | grid size matches config; cop/thief on different cells |
| Move legality | off-grid moves are no-ops; barrier cells block |
| Simultaneous moves | both agents resolve in one turn |
| Capture | cop position == thief position → sub-game ends, cop wins |
| Timeout | 25 moves without capture → thief wins |
| Barrier placement | cop on cell X → cannot place on X; max 5 enforced |
| 6-sub-game game | exactly 6 sub-games run; report JSON has 6 entries; totals match per-sub sum |
| Technical loss retry | mocked timeout → sub-game replays; doesn't increment counter |
| JSON shape | matches spec § 3.5 example exactly (round-trip via JSON schema) |

## 9. Acceptance criteria

1. `game/` package contains pure functions only (no I/O, no env mutation outside the `Board` instance passed in).
2. Every numeric in this PRD comes from `configs/setup.yaml::game` + `configs/setup.yaml::scoring`.
3. `Game.play_one_game(seed)` produces an identical `GameReport` on identical seeds.
4. JSON output validates against the spec § 3.5 example.

## 10. Non-goals

- Real-time multiplayer with humans
- More than 2 agents per side (extensible but not required)
- Non-grid topology
