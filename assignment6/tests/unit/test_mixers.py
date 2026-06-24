"""Layer 5+6 — VDN and QMIX mixer tests.

QMIX monotonicity test is the headline (PRD_ctde § 4 — IGM principle)."""

from __future__ import annotations

import pytest
import torch

from marl_lab.model.qmix_mixer import QMIXMixer
from marl_lab.model.vdn_mixer import VDNMixer

# ----- VDN: additive identity -----

def test_vdn_n_agents_validates() -> None:
    with pytest.raises(ValueError):
        VDNMixer(n_agents=0)


def test_vdn_sum_identity_2_agents() -> None:
    mixer = VDNMixer(n_agents=2)
    q = torch.tensor([[[3.0, 4.0]]])             # (B=1, T=1, n_agents=2)
    q_tot = mixer(q)
    assert q_tot.item() == pytest.approx(7.0)


def test_vdn_sum_identity_n_agents() -> None:
    mixer = VDNMixer(n_agents=5)
    q = torch.ones(2, 3, 5)
    q_tot = mixer(q)
    assert q_tot.shape == (2, 3)
    assert torch.all(q_tot == 5.0)


def test_vdn_rejects_wrong_last_dim() -> None:
    mixer = VDNMixer(n_agents=2)
    with pytest.raises(ValueError):
        mixer(torch.ones(2, 3, 5))


def test_vdn_ignores_global_state() -> None:
    mixer = VDNMixer(n_agents=2)
    q = torch.tensor([[[3.0, 4.0]]])
    q_tot_a = mixer(q, global_state=torch.zeros(1, 1, 10))
    q_tot_b = mixer(q, global_state=torch.randn(1, 1, 10))
    torch.testing.assert_close(q_tot_a, q_tot_b)


def test_vdn_no_params() -> None:
    mixer = VDNMixer(n_agents=2)
    assert len(list(mixer.parameters())) == 0


# ----- QMIX: shape contracts -----

def test_qmix_output_shape() -> None:
    mixer = QMIXMixer(n_agents=2, state_dim=10, embed_dim=32, hyper_hidden=64)
    q = torch.randn(4, 5, 2)         # B=4, T=5, n_agents=2
    s = torch.randn(4, 5, 10)
    q_tot = mixer(q, s)
    assert q_tot.shape == (4, 5)


def test_qmix_rejects_wrong_n_agents() -> None:
    mixer = QMIXMixer(n_agents=2, state_dim=10)
    q = torch.randn(4, 5, 3)
    s = torch.randn(4, 5, 10)
    with pytest.raises(ValueError):
        mixer(q, s)


# ----- QMIX: monotonicity (IGM principle) -----

def test_qmix_monotonicity_finite_difference() -> None:
    """∂Q_tot/∂Qᵢ ≥ 0 for all i — the IGM-preserving constraint."""
    torch.manual_seed(0)
    mixer = QMIXMixer(n_agents=2, state_dim=10, embed_dim=32, hyper_hidden=64)
    # 100 random (q, s) probes
    for _ in range(100):
        q = torch.randn(1, 1, 2, requires_grad=True)
        s = torch.randn(1, 1, 10)
        q_tot = mixer(q, s)
        # ∂Q_tot/∂qᵢ via autograd
        grads = torch.autograd.grad(q_tot.sum(), q)[0]
        assert (grads >= 0).all(), f"Monotonicity violated: {grads}"


def test_qmix_monotonicity_finite_difference_n5() -> None:
    """Monotonicity holds for n=5 agents too (generalisation)."""
    torch.manual_seed(0)
    mixer = QMIXMixer(n_agents=5, state_dim=20)
    for _ in range(50):
        q = torch.randn(1, 1, 5, requires_grad=True)
        s = torch.randn(1, 1, 20)
        q_tot = mixer(q, s)
        grads = torch.autograd.grad(q_tot.sum(), q)[0]
        assert (grads >= 0).all()


def test_qmix_state_dependence() -> None:
    """Same q, different s → different Q_tot (hypernet conditioning works)."""
    torch.manual_seed(0)
    mixer = QMIXMixer(n_agents=2, state_dim=10)
    q = torch.ones(1, 1, 2)
    s1 = torch.randn(1, 1, 10)
    s2 = torch.randn(1, 1, 10)
    q_tot_1 = mixer(q, s1)
    q_tot_2 = mixer(q, s2)
    assert not torch.isclose(q_tot_1, q_tot_2).item()


def test_qmix_gradient_flow() -> None:
    mixer = QMIXMixer(n_agents=2, state_dim=10)
    q = torch.randn(2, 3, 2, requires_grad=True)
    s = torch.randn(2, 3, 10)
    q_tot = mixer(q, s)
    q_tot.sum().backward()
    for p in mixer.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_qmix_increasing_q_increases_q_tot() -> None:
    """Direct monotonicity probe: increasing one qᵢ should not decrease Q_tot."""
    torch.manual_seed(42)
    mixer = QMIXMixer(n_agents=2, state_dim=10)
    s = torch.randn(1, 1, 10)
    q_low = torch.tensor([[[0.0, 0.0]]])
    q_high = torch.tensor([[[1.0, 0.0]]])    # increase only agent 0
    q_tot_low = mixer(q_low, s).item()
    q_tot_high = mixer(q_high, s).item()
    assert q_tot_high >= q_tot_low
