"""Spec § 9.4 bonus-game JSON serialisation + subject + idempotency +
peer-agreement verification.

Separate module from `gmail/formatter.py` because the bonus payload has a
different shape (report_type, groups, sub_games with cop_group/thief_group,
totals_by_group, bonus_claim, mutual_agreement). Keeping them apart makes
the § 9-vs-§ 3.5 distinction visible in imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from marl_lab.shared.types import BonusGameReport


def bonus_report_to_json(report: BonusGameReport,
                          include_provenance: bool = True) -> str:
    """Spec § 9.4 JSON shape (with ``report_type`` header) — deterministic
    output. If ``include_provenance``, a ``provenance`` block carrying the
    git SHA + lib versions is added at the top level (same as
    `gmail.formatter.report_to_json`)."""
    payload = {"report_type": "bonus_game", **asdict(report)}
    if include_provenance:
        from marl_lab.shared.provenance import collect_provenance
        payload["provenance"] = collect_provenance()
    return json.dumps(payload, sort_keys=True, indent=2)


def build_bonus_idempotency_key(report: BonusGameReport) -> str:
    """SHA-256 of the deterministic bonus JSON — same match content →
    same id (ADR-010 extended for § 9).

    Excludes provenance + mutual_agreement (both are environment- or
    coordination-dependent — the match content itself is what identifies
    the game)."""
    canonical = _canonical_match_content(report)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


def bonus_email_subject(report: BonusGameReport,
                          prefix: str = "[MARL Bonus Game]") -> str:
    """Spec § 9 subject line: ``[MARL Bonus Game] <Group1> vs <Group2> – Final Report``."""
    g1 = report.groups.get("group_1", "?")
    g2 = report.groups.get("group_2", "?")
    return f"{prefix} {g1} vs {g2} – Final Report"


def _canonical_match_content(report: BonusGameReport) -> dict:
    """The subset of the report that BOTH groups must agree on.

    Excludes: provenance (environment-dependent), mutual_agreement
    (coordination output), students (each group only knows their own IDs;
    the OTHER group's list might not have been shared before match time)."""
    return {
        "groups": dict(sorted(report.groups.items())),
        "sub_games": [
            {"id": sg.id, "cop_group": sg.cop_group,
             "thief_group": sg.thief_group, "winner": sg.winner,
             "scores": dict(sorted(sg.scores.items()))}
            for sg in sorted(report.sub_games, key=lambda s: s.id)
        ],
        "totals_by_group": dict(sorted(report.totals_by_group.items())),
        "bonus_claim": dict(sorted(report.bonus_claim.items())),
    }


def verify_peer_agreement(local_report: BonusGameReport,
                            peer_report_json: str) -> tuple[bool, str]:
    """Compare local canonical content against the peer's report JSON.

    Returns (agreed, reason). If ``agreed`` is True the caller should set
    ``local_report.mutual_agreement = True`` before sending. If False,
    ``reason`` names the first field where the two disagreed (helpful when
    the peer's env or config differs)."""
    local_canon = _canonical_match_content(local_report)
    try:
        peer_payload = json.loads(peer_report_json)
    except json.JSONDecodeError as e:
        return (False, f"peer JSON parse error: {e}")
    if peer_payload.get("report_type") != "bonus_game":
        return (False, "peer report_type != 'bonus_game'")
    # Build the peer's canonical view from their payload (same schema)
    peer_canon = {
        "groups": dict(sorted(peer_payload.get("groups", {}).items())),
        "sub_games": [
            {"id": sg["id"], "cop_group": sg["cop_group"],
             "thief_group": sg["thief_group"], "winner": sg["winner"],
             "scores": dict(sorted(sg["scores"].items()))}
            for sg in sorted(peer_payload.get("sub_games", []),
                              key=lambda s: s["id"])
        ],
        "totals_by_group": dict(sorted(peer_payload.get("totals_by_group", {}).items())),
        "bonus_claim": dict(sorted(peer_payload.get("bonus_claim", {}).items())),
    }
    for field in ("groups", "sub_games", "totals_by_group", "bonus_claim"):
        if local_canon[field] != peer_canon[field]:
            return (False, f"disagreement on '{field}'")
    return (True, "match")
