"""Classify an exercise name into a coarse muscle group.

The Kaggle dataset has no explicit muscle column, so we infer the group
from keywords in the exercise name. Matching uses word boundaries so
"crunch" doesn't match "run" and "press" matches "bench press" but not
"impress". Order of groups matters — more specific patterns first.

This is a heuristic — documented as a limitation in PRD_data.md.
"""

from __future__ import annotations

import re

from fitness_rl.shared.types import MuscleGroup

# Each keyword is matched as a whole word (word-boundary regex). Order of
# the dict matters: more-specific groups first so "leg press" wins over
# "press", and "crunch" wins over "run".
_KEYWORDS: dict[MuscleGroup, tuple[str, ...]] = {
    MuscleGroup.CORE: (
        "plank", "crunch", "sit-up", "situp", "russian twist", "leg raise",
        "knee raise", "hollow hold", "core", "oblique", "ab",
        "mobility", "stretch", "ankle", "wrist", "hip flex", "dorsiflexion",
        "banded", "distraction",
    ),
    MuscleGroup.LEGS: (
        "squat", "lunge", "leg press", "leg extension", "leg curl",
        "calf raise", "hip thrust", "glute", "quad", "hamstring",
        "step-up", "stepup", "bulgarian", "leg",
    ),
    MuscleGroup.PULL: (
        "row", "pull-up", "pullup", "chin-up", "chinup", "lat pulldown",
        "deadlift", "curl", "face pull", "shrug", "back ext", "rear delt",
        "bicep", "lat",
    ),
    MuscleGroup.PUSH: (
        "bench", "press", "push-up", "pushup", "dip", "tricep", "chest",
        "shoulder", "lateral raise", "front raise", "fly", "skull crusher",
        "overhead",
    ),
    MuscleGroup.CARDIO: (
        "run", "running", "jog", "sprint", "cycle", "bike", "treadmill",
        "burpee", "jump rope", "jumping jack", "mountain climber",
        "cardio", "interval", "hiit", "elliptical",
    ),
}


def classify(exercise_name: str) -> MuscleGroup:
    """Return the most specific muscle group implied by the exercise name."""
    if not isinstance(exercise_name, str) or not exercise_name.strip():
        return MuscleGroup.UNKNOWN
    name = exercise_name.lower()
    for group, keywords in _KEYWORDS.items():
        for kw in keywords:
            # Word-boundary match so "run" doesn't match "crunch", but
            # "press" still matches "bench press" or "incline press".
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, name):
                return group
    return MuscleGroup.UNKNOWN
