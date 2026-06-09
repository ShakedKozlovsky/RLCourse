"""ComparisonService — aggregate per-seed finals into mean ± 95 % CI."""

from __future__ import annotations

import pytest

from proximal_lab.services.comparison_service import (
    CellSummary,
    ComparisonReport,
    ComparisonService,
    aggregate_with_ci,
)


def test_aggregate_with_ci_single_seed_zero_ci() -> None:
    mean, ci = aggregate_with_ci([3.0])
    assert mean == 3.0
    assert ci == 0.0


def test_aggregate_with_ci_two_seeds_positive_ci() -> None:
    mean, ci = aggregate_with_ci([1.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert ci > 0.0


def test_aggregate_with_ci_empty_returns_zero() -> None:
    mean, ci = aggregate_with_ci([])
    assert mean == 0.0 and ci == 0.0


def test_comparison_report_serialises() -> None:
    svc = ComparisonService()
    report = svc.report("lambda_sweep",
                         per_cell_finals={"lambda=0.0": [1.0, 2.0],
                                          "lambda=0.95": [10.0, 11.0]})
    assert isinstance(report, ComparisonReport)
    assert len(report.cells) == 2
    payload = report.to_dict()
    assert payload["label"] == "lambda_sweep"
    assert len(payload["cells"]) == 2
    assert "final_reward_mean" in payload["cells"][0]


def test_cell_summary_construction() -> None:
    cell = CellSummary(name="x", n_seeds=3, final_reward_mean=5.0,
                       final_reward_ci_95=0.5, overall_reward_mean=4.5)
    assert cell.n_seeds == 3
    assert cell.to_dict()["final_reward_mean"] == 5.0
