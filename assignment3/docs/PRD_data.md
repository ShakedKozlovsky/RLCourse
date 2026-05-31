# PRD — Data Pipeline (Part B)

## End-to-end flow

```
data/raw/program_summary.csv  +  data/raw/programs_detailed_boostcamp_kaggle.csv
    ▼
KaggleLoader.load()
    ▼
Preprocessor.clean()          ← handle negative sets/reps, type coercion
    ▼
ProgramSelector.pick()        ← one program matching 8 criteria
    ▼
ProgramSelector.filter_detailed(chosen_program)
    ▼
TrajectoryBuilder.build()     ← group by week+day, compute total_volume,
                              ←   muscle_distribution, session_duration, etc.
                              ←   insert Rest Days for empty week+day slots
    ▼
FeatureEngineer.transform()   ← produce 16-dim state vector per day
    ▼
DataService.load() → (trajectory: List[State], actions: List[Action])
```

## File contents (per assignment §7.2.3)

### `program_summary.csv` (one row per program)

Schema (relevant fields): `title, description, goal, level, equipment, program_length (weeks), time_per_workout, total_exercises, created, last_edit`.

### `programs_detailed_boostcamp_kaggle.csv` (one row per exercise in a program-day)

Schema (relevant fields): `program_title, day_of_week, week_number, exercise_name, sets, reps, level, goal, target_muscle, equipment, training_time`.

## Data sanity

Per assignment §7.2.3: **some negative values in sets/reps actually represent seconds**. Our `Preprocessor.clean()`:

1. Negative `sets` or `reps` → take absolute value if magnitude < 600 (seconds), else NaN.
2. NaN sets/reps → drop that row (cannot compute volume).
3. Missing target_muscle → assign "unknown" group (not used in any reward).

## Program selection criteria (§7.2.4)

`ProgramSelector.pick()` filters `program_summary.csv` by:

| Field | Filter |
|---|---|
| `equipment` | `== "Full Gym"` |
| `program_length` | `4 <= length <= 12` weeks |
| `time_per_workout` | `45 <= time <= 120` minutes |

If multiple programs match, the first by sorted `program_id` is taken (reproducible). The chosen program is recorded in `results/chosen_program.json` for transparency.

## Trajectory construction (§7.2.4 step 3)

`TrajectoryBuilder.build(detailed_rows)`:

```python
for week in 1..program_length:
    for day in 1..7:
        exercises = detailed_rows[(detailed_rows.week == week) & (detailed_rows.day == day)]
        if exercises.empty:
            yield RestDay(week=week, day=day)
            continue
        total_volume = sum(e.sets * e.reps for e in exercises)
        muscle_dist  = aggregate_by_muscle_group(exercises)
        session_dur  = sum(e.training_time for e in exercises)
        yield TrainingDay(
            week=week, day=day,
            total_volume=total_volume,
            muscle_distribution=muscle_dist,
            session_duration=session_dur,
        )
```

The output is a list of length `T = program_length * 7` days, one row per day, alternating training days and rest days.

## State vector construction (§7.2.4 step 4)

`FeatureEngineer.transform(trajectory)`:

```python
state_t = concat(
    [total_volume_t / V_max],                      # 1d
    [muscle_dist_t (5d)],                          # 5d
    [session_duration_t / 120],                    # 1d
    [week_index_t / program_length],               # 1d
    one_hot_day_of_week(day_in_cycle_t, 7),        # 7d
    [1.0 if rest_day else 0.0],                    # 1d
)  # total = 16
```

`V_max` is the maximum `total_volume` across the entire trajectory — computed once and persisted.

## Per-day action labels (for LSTM training)

The LSTM world model needs an action at each step. Since the dataset doesn't contain actions per se, we **infer** the action from the muscle group with highest volume on that day:

```python
action_t = {
    "chest|shoulders|triceps": PUSH,
    "back|biceps":             PULL,
    "legs|glutes|core":        LEGS,
    "cardio":                  CARDIO,
    None (rest day):           REST,
}.lookup(top_muscle_group)
```

This is documented as an inference assumption — the LSTM models the dynamics of "what state follows after taking action X", and we need labelled actions to train it supervised.

## Acceptance criteria

- `test_kaggle_loader.py::test_loads_both_csvs` — reads both files; raises if missing.
- `test_preprocessor.py::test_negative_sets_handled` — `-30` becomes `30` (interpreted as seconds, abs taken).
- `test_program_selector.py::test_picks_matching_criteria` — chosen program meets all 8 criteria.
- `test_trajectory_builder.py::test_rest_days_inserted` — empty week+day slots produce Rest Day rows.
- `test_feature_engineer.py::test_state_vector_dim_16` — output is exactly 16-dim.

## Caveats explicitly documented

- This is **one** synthetic trainee, not a population. Generalisation across programs is out of scope (ADR-003).
- Action labels are **inferred** from muscle groups; real-world labels would come from a trainer's actual prescription.
- The data is workout *programs*, not training *outcomes*. The LSTM models program structure, not human physiology.
