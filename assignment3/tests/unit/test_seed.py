"""Coverage of shared/seed.py — exercises the torch + CUDA paths."""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import torch

from fitness_rl.shared.seed import set_global_seed


def test_set_global_seed_seeds_random_and_numpy() -> None:
    set_global_seed(123)
    r1 = random.random()
    n1 = float(np.random.random())
    set_global_seed(123)
    r2 = random.random()
    n2 = float(np.random.random())
    assert r1 == r2
    assert n1 == n2


def test_set_global_seed_seeds_torch() -> None:
    set_global_seed(456)
    t1 = torch.rand(5)
    set_global_seed(456)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)


def test_set_global_seed_cuda_path_invoked_when_available() -> None:
    """Force ``torch.cuda.is_available`` to return True to cover the CUDA branch."""
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "manual_seed_all") as mock_cuda_seed:
        set_global_seed(789)
        # torch.manual_seed also calls cuda.manual_seed_all internally on CUDA
        # builds; the important thing is *our* call ran with the right seed.
        assert mock_cuda_seed.call_count >= 1
        assert mock_cuda_seed.call_args_list[-1].args == (789,)


def test_set_global_seed_returns_silently_if_torch_missing() -> None:
    """Simulate the ImportError fallback path."""
    import builtins
    orig_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        # Should not raise.
        set_global_seed(42)
