"""Layer 4 — Polyak soft-target update 4-test math battery.

PRD_soft_target_updates.md § 5 + EX05 § Item 2 ('show the exact code lines that
implement the Polyak averaging mechanism for updating target networks')."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from roomba_lab.model.soft_update import hard_copy, polyak_update


def _two_linears(seed_target: int, seed_source: int) -> tuple[nn.Linear, nn.Linear]:
    torch.manual_seed(seed_target)
    target = nn.Linear(4, 4)
    torch.manual_seed(seed_source)
    source = nn.Linear(4, 4)
    return target, source


def test_tau_zero_target_unchanged() -> None:
    target, source = _two_linears(1, 2)
    before = target.weight.clone()
    polyak_update(target.parameters(), source.parameters(), tau=0.0)
    torch.testing.assert_close(target.weight, before)


def test_tau_one_hard_copy() -> None:
    target, source = _two_linears(1, 2)
    polyak_update(target.parameters(), source.parameters(), tau=1.0)
    torch.testing.assert_close(target.weight, source.weight)


def test_tau_half_midpoint() -> None:
    target = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
        source.weight.fill_(1.0)
        source.bias.fill_(1.0)
    polyak_update(target.parameters(), source.parameters(), tau=0.5)
    assert torch.all(target.weight == 0.5)
    assert torch.all(target.bias == 0.5)


def test_repeated_calls_converge() -> None:
    target = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
        source.weight.fill_(1.0)
        source.bias.fill_(1.0)
    for _ in range(200):
        polyak_update(target.parameters(), source.parameters(), tau=0.05)
    assert torch.allclose(target.weight, source.weight, atol=1e-3)


def test_invalid_tau_raises() -> None:
    target, source = _two_linears(0, 0)
    with pytest.raises(ValueError):
        polyak_update(target.parameters(), source.parameters(), tau=1.5)
    with pytest.raises(ValueError):
        polyak_update(target.parameters(), source.parameters(), tau=-0.1)


def test_hard_copy_matches_tau_one() -> None:
    target, source = _two_linears(1, 2)
    hard_copy(target, source)
    torch.testing.assert_close(target.weight, source.weight)
