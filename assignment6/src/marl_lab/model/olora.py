"""OLoRA — Orthonormal Low-Rank Adaptation (Büyükakyüz 2024, arXiv:2406.01775).

QR-decomposition-based orthonormal init for the low-rank A factor. At step 0
the OLoRA output equals the pre-trained layer's output exactly (B=0 → BAᵀ=0),
so the wrapper is a no-op at init; gradient updates to B (and optionally A)
then adapt the layer in a well-conditioned subspace.

Used for PEFT of pre-trained Q-net backbones in unstable MARL training."""

from __future__ import annotations

import torch
import torch.nn as nn


class OLoRAAdapter(nn.Module):
    """Wrap a frozen pre-trained ``nn.Linear`` with an orthonormal low-rank update.

    Forward computation:  y = W_pre @ x + α · B @ Aᵀ @ x.

    Init:
      A ← Q from QR-decomposition of W_pre^T  (columns orthonormal)
      B ← 0  (so BAᵀ = 0 at init → wrapped output == W_pre output)

    Trainable parameters: B (and A if ``freeze_a=False``). W_pre is frozen.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 1.0,
                 freeze_a: bool = True) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        d_in = base_layer.in_features
        d_out = base_layer.out_features
        if rank > min(d_in, d_out):
            raise ValueError(
                f"rank ({rank}) cannot exceed min(d_in, d_out) = {min(d_in, d_out)}"
            )
        self.rank = int(rank)
        self.alpha = float(alpha)
        # Freeze the base layer
        for p in base_layer.parameters():
            p.requires_grad = False
        self.base = base_layer
        # Orthonormal init via QR of W_pre^T → A has orthonormal COLUMNS
        with torch.no_grad():
            wpre_t = base_layer.weight.detach().T   # (d_in, d_out)
            # Pad/truncate to rank — Q has shape (d_in, min(d_in, d_out))
            q_full, _ = torch.linalg.qr(wpre_t, mode="reduced")
            a_init = q_full[:, :rank].contiguous()  # (d_in, rank)
        # A: (d_in, rank), B: (d_out, rank); update = α · B @ Aᵀ
        self.A = nn.Parameter(a_init, requires_grad=not freeze_a)
        self.B = nn.Parameter(torch.zeros(d_out, rank), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = W_pre @ x + α · B @ Aᵀ @ x. Equivalent to ``base(x)`` at init."""
        base_out = self.base(x)
        # x: (..., d_in); compute x @ A → (..., rank); then @ B^T → (..., d_out)
        delta = self.alpha * (x @ self.A) @ self.B.t()
        return base_out + delta


def wrap_with_olora(model: nn.Module, rank: int = 8, alpha: float = 1.0,
                    freeze_a: bool = True, name_filter: str | None = None) -> int:
    """Replace every ``nn.Linear`` in ``model`` (in-place) with an OLoRAAdapter.

    Returns the number of layers wrapped. ``name_filter`` (substring match)
    optionally restricts which submodule paths get wrapped."""
    wrapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if name_filter is not None and name_filter not in full_name:
                continue
            adapter = OLoRAAdapter(child, rank=rank, alpha=alpha, freeze_a=freeze_a)
            setattr(module, child_name, adapter)
            wrapped += 1
    return wrapped
