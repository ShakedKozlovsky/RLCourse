"""PPO clip math battery — slide-11/12 sign × clip-window cases.

Four headline tests proving the slide-11 + slide-12 intuition:

    Â > 0,  r in window   → unclipped branch wins  (regular PG)
    Â > 0,  r > 1+ε       → clipped branch wins    (flat surrogate → no further push)
    Â < 0,  r in window   → unclipped branch wins  (regular PG)
    Â < 0,  r > 1+ε       → UNCLIPPED branch wins  (the safety case — pulls policy back)
    Â < 0,  r < 1−ε       → clipped branch wins    (no further push)

The `min` is what makes PPO "deliberately pessimistic" (slide 13).
"""

from __future__ import annotations

import pytest
import torch

from proximal_lab.services.ppo_clip import approx_kl, ppo_clip_loss


def test_invalid_clip_eps_raises() -> None:
    ratio = torch.tensor([1.0])
    adv = torch.tensor([1.0])
    with pytest.raises(ValueError):
        ppo_clip_loss(ratio, adv, clip_eps=0.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ppo_clip_loss(torch.zeros(3), torch.zeros(2), clip_eps=0.2)


def test_ratio_inside_window_positive_adv_uses_unclipped() -> None:
    """Â > 0, r ∈ [1−ε, 1+ε] ⇒ loss = −r·Â (unclipped = clipped, both fine)."""
    ratio = torch.tensor([1.0])
    adv = torch.tensor([1.0])
    loss, frac = ppo_clip_loss(ratio, adv, clip_eps=0.2)
    assert loss.item() == pytest.approx(-1.0)
    assert frac.item() == 0.0


def test_ratio_above_window_positive_adv_uses_clipped_branch() -> None:
    """Â > 0, r > 1+ε ⇒ clipped branch wins (less surrogate gain than unclipped)."""
    eps = 0.2
    ratio = torch.tensor([2.0])  # well above 1+ε
    adv = torch.tensor([1.0])
    loss, frac = ppo_clip_loss(ratio, adv, clip_eps=eps)
    # clipped value = (1+ε)·Â = 1.2; unclipped = 2.0. min(2.0, 1.2) = 1.2.
    # loss = -1.2
    assert loss.item() == pytest.approx(-(1.0 + eps))
    assert frac.item() == 1.0


def test_ratio_above_window_negative_adv_uses_unclipped_branch() -> None:
    """Â < 0, r > 1+ε ⇒ UNCLIPPED branch wins — slide-12 safety case.

    The unclipped surrogate is more negative (r=5, Â=-1 → -5) than the clipped
    one (1+ε=1.2, Â=-1 → -1.2). min(-5, -1.2) = -5 ⇒ loss = +5, pulls policy back.
    """
    ratio = torch.tensor([5.0])
    adv = torch.tensor([-1.0])
    loss, frac = ppo_clip_loss(ratio, adv, clip_eps=0.2)
    assert loss.item() == pytest.approx(5.0)
    assert frac.item() == 1.0


def test_ratio_below_window_negative_adv_uses_clipped_branch() -> None:
    """Â < 0, r < 1−ε ⇒ clipped branch wins (no further push on bad action already weakened)."""
    eps = 0.2
    ratio = torch.tensor([0.1])  # below 1-ε
    adv = torch.tensor([-1.0])
    loss, frac = ppo_clip_loss(ratio, adv, clip_eps=eps)
    # clipped value = (1-ε)·Â = -0.8; unclipped = -0.1. min(-0.1, -0.8) = -0.8.
    # loss = +0.8
    assert loss.item() == pytest.approx(1.0 - eps)
    assert frac.item() == 1.0


def test_loss_is_differentiable_wrt_ratio() -> None:
    ratio = torch.tensor([1.5], requires_grad=True)
    adv = torch.tensor([1.0])
    loss, _ = ppo_clip_loss(ratio, adv, clip_eps=0.2)
    loss.backward()
    assert ratio.grad is not None


def test_clip_fraction_mixed_batch() -> None:
    """3/4 transitions outside window → clip_fraction = 0.75."""
    ratio = torch.tensor([0.5, 2.0, 1.0, 5.0])
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    _, frac = ppo_clip_loss(ratio, adv, clip_eps=0.2)
    assert frac.item() == pytest.approx(0.75)


def test_approx_kl_zero_when_log_probs_equal() -> None:
    lp_old = torch.tensor([0.1, 0.2, 0.3])
    lp_new = lp_old.clone()
    assert approx_kl(lp_new, lp_old).item() == pytest.approx(0.0)


def test_approx_kl_positive_when_new_is_lower() -> None:
    lp_old = torch.tensor([-1.0, -1.0])
    lp_new = torch.tensor([-2.0, -2.0])
    assert approx_kl(lp_new, lp_old).item() == pytest.approx(1.0)
