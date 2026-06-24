"""Format a GameReport into the spec § 3.5 JSON body + a Gmail-friendly subject."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from marl_lab.shared.types import GameReport


def report_to_json(report: GameReport, include_provenance: bool = True) -> str:
    """Spec § 3.5 JSON shape, sorted, deterministic — string suitable as email body.

    If ``include_provenance`` is true (default), the payload gains a top-level
    ``provenance`` block carrying the git SHA, library versions, and Python
    version that produced the report. This lets the TA verify the email
    against the exact commit they're grading.
    """
    payload = asdict(report)
    for sg in payload["sub_games"]:
        if "start" in sg and hasattr(sg["start"], "isoformat"):
            sg["start"] = sg["start"].isoformat()
        if "end" in sg and hasattr(sg["end"], "isoformat"):
            sg["end"] = sg["end"].isoformat()
    if include_provenance:
        from marl_lab.shared.provenance import collect_provenance
        payload["provenance"] = collect_provenance()
    return json.dumps(payload, sort_keys=True, indent=2)


def build_idempotency_key(report: GameReport) -> str:
    """SHA-256 of the deterministic JSON — same GAME content → same id (ADR-010).

    Excludes provenance so the key is stable across re-runs from different
    machines / library versions. Two distinct game outcomes still produce
    distinct keys because the per-sub-game scores + winners differ."""
    js = report_to_json(report, include_provenance=False)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()[:16]


def email_subject(report: GameReport, prefix: str = "[MARL Game]") -> str:
    """``[MARL Game] <group_name> <group_code> totals: cop=X thief=Y``."""
    totals = report.totals or {}
    return (
        f"{prefix} {report.group_name} {report.group_code} "
        f"totals: cop={totals.get('cop', 0)} thief={totals.get('thief', 0)}"
    )
