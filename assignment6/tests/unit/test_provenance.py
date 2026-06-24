"""Provenance metadata embedded in GameReport — proves the report's origin."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from marl_lab.gmail.formatter import build_idempotency_key, report_to_json
from marl_lab.shared.provenance import collect_provenance
from marl_lab.shared.types import GameReport, StudentEntry, SubGameResult


def _dummy_report() -> GameReport:
    now = datetime(2026, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
    sg = SubGameResult(id=1, start=now, end=now, moves=10, winner="cop",
                        scores={"cop": 20, "thief": 5})
    return GameReport(
        group_name="g", group_code="C0DE1234",
        students=[StudentEntry(role="A", full_name="Shaked", id="1")],
        github_repo="https://github.com/x/y", timezone="UTC",
        sub_games=[sg], totals={"cop": 20, "thief": 5},
    )


def test_provenance_contains_required_keys() -> None:
    prov = collect_provenance()
    expected = {"marl_lab_version", "git_sha", "git_dirty", "python", "platform",
                "numpy_version", "torch_version", "matplotlib_version"}
    assert expected.issubset(set(prov.keys()))


def test_provenance_is_embedded_in_report_json() -> None:
    js = report_to_json(_dummy_report())
    payload = json.loads(js)
    assert "provenance" in payload
    assert payload["provenance"]["marl_lab_version"]
    assert "torch_version" in payload["provenance"]


def test_provenance_can_be_excluded() -> None:
    js = report_to_json(_dummy_report(), include_provenance=False)
    payload = json.loads(js)
    assert "provenance" not in payload


def test_idempotency_key_is_independent_of_provenance() -> None:
    """Same game content → same key even when provenance changes (machine swap).

    This is the load-bearing property: the idempotency ledger must catch
    duplicate sends regardless of where the report was generated."""
    key_a = build_idempotency_key(_dummy_report())
    key_b = build_idempotency_key(_dummy_report())
    assert key_a == key_b
    # Both should also match the deterministic JSON (no provenance)
    js = report_to_json(_dummy_report(), include_provenance=False)
    import hashlib
    expected = hashlib.sha256(js.encode("utf-8")).hexdigest()[:16]
    assert key_a == expected


def test_git_sha_format_when_inside_repo() -> None:
    """If we're inside a git repo, sha should be hex chars not 'unknown'."""
    prov = collect_provenance()
    sha = prov["git_sha"]
    if sha != "unknown":
        assert all(c in "0123456789abcdef" for c in sha)
        assert 6 <= len(sha) <= 12
