"""Layer 7 — OLoRA tests (QR orthonormality + zero-perturbation + gradient flow)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from marl_lab.model.olora import OLoRAAdapter, wrap_with_olora


def test_olora_validates_rank() -> None:
    base = nn.Linear(8, 4)
    with pytest.raises(ValueError):
        OLoRAAdapter(base, rank=0)


def test_olora_rejects_rank_above_min_dim() -> None:
    base = nn.Linear(8, 4)
    with pytest.raises(ValueError):
        OLoRAAdapter(base, rank=5)        # min(8, 4) = 4


def test_olora_a_columns_orthonormal() -> None:
    """A.T @ A == I_rank (orthonormal columns)."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    adapter = OLoRAAdapter(base, rank=8)
    gram = adapter.A.t() @ adapter.A
    torch.testing.assert_close(gram, torch.eye(8), atol=1e-5, rtol=1e-5)


def test_olora_zero_perturbation_at_init() -> None:
    """At init B == 0, so wrapped output equals base output exactly."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    adapter = OLoRAAdapter(base, rank=2)
    x = torch.randn(3, 8)
    base_out = base(x)
    adapter_out = adapter(x)
    torch.testing.assert_close(adapter_out, base_out)


def test_olora_base_layer_frozen() -> None:
    """W_pre must not receive gradients; only B (and optionally A) do."""
    base = nn.Linear(8, 4)
    adapter = OLoRAAdapter(base, rank=2, freeze_a=True)
    x = torch.randn(3, 8)
    out = adapter(x)
    out.sum().backward()
    assert base.weight.grad is None
    assert base.bias.grad is None
    assert adapter.B.grad is not None
    assert adapter.A.grad is None        # frozen


def test_olora_a_trainable_when_unfrozen() -> None:
    base = nn.Linear(8, 4)
    adapter = OLoRAAdapter(base, rank=2, freeze_a=False)
    x = torch.randn(3, 8)
    out = adapter(x)
    out.sum().backward()
    assert adapter.A.grad is not None
    assert adapter.B.grad is not None


def test_olora_parameter_count_lower_than_full_finetune() -> None:
    """(d_in + d_out) · r should be much smaller than d_in · d_out."""
    base = nn.Linear(256, 128)
    adapter = OLoRAAdapter(base, rank=8)
    # Trainable: B (128, 8) = 1024. A is frozen by default.
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    full_finetune = 256 * 128 + 128       # bias
    assert trainable < full_finetune


def test_olora_alpha_scales_adaptation() -> None:
    """After perturbing B, adapter output scales linearly with alpha."""
    torch.manual_seed(0)
    base = nn.Linear(4, 4)
    a1 = OLoRAAdapter(base, rank=2, alpha=1.0)
    a2 = OLoRAAdapter(base, rank=2, alpha=2.0)
    with torch.no_grad():
        a1.B.copy_(torch.randn_like(a1.B))
        a2.A.copy_(a1.A)
        a2.B.copy_(a1.B)
    x = torch.randn(3, 4)
    delta1 = a1(x) - base(x)
    delta2 = a2(x) - base(x)
    torch.testing.assert_close(delta2, 2.0 * delta1, atol=1e-5, rtol=1e-5)


def test_wrap_with_olora_replaces_all_linears() -> None:
    """wrap_with_olora should swap every nn.Linear in the model tree."""
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(8, 16)
            self.l2 = nn.Linear(16, 4)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    m = Tiny()
    n_wrapped = wrap_with_olora(m, rank=2)
    assert n_wrapped == 2
    assert isinstance(m.l1, OLoRAAdapter)
    assert isinstance(m.l2, OLoRAAdapter)


def test_wrap_with_olora_preserves_forward_output() -> None:
    """At init, wrapped model output should equal un-wrapped output."""
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(8, 16)
            self.l2 = nn.Linear(16, 4)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    torch.manual_seed(0)
    m_a, m_b = Tiny(), Tiny()
    m_b.load_state_dict(m_a.state_dict())   # identical weights
    wrap_with_olora(m_b, rank=2)
    x = torch.randn(3, 8)
    out_a = m_a(x)
    out_b = m_b(x)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)
