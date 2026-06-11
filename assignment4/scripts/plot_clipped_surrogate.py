"""Visualise the PPO clipped surrogate L^CLIP(r, Â) — slide 10 made visceral.

Four panels covering the four (sign of Â) × (r region) cases the slide-11/12
intuition describes:

    Â > 0, r ∈ window  — both branches equal: regular policy gradient
    Â > 0, r > 1+ε     — clipped branch wins: surrogate flattens, no further push
    Â < 0, r > 1+ε     — UNCLIPPED branch wins: more negative, pulls policy back
    Â < 0, r < 1−ε     — clipped branch wins: no further push on bad action
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _surrogates(ratios: np.ndarray, adv: float, eps: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unclipped = ratios * adv
    clipped = np.clip(ratios, 1.0 - eps, 1.0 + eps) * adv
    l_clip = np.minimum(unclipped, clipped)  # min(), as in PPO
    return unclipped, clipped, l_clip


def _panel(ax, adv: float, eps: float = 0.2) -> None:
    ratios = np.linspace(0.0, 2.0, 500)
    unclipped, clipped, l_clip = _surrogates(ratios, adv, eps)
    ax.plot(ratios, unclipped, "--", color="#888888", linewidth=1.5,
             label="r·Â (unclipped)")
    ax.plot(ratios, clipped, ":", color="#cc6677", linewidth=1.5,
             label="clip(r)·Â")
    ax.plot(ratios, l_clip, "-", color="#4477aa", linewidth=2.5,
             label="L^CLIP = min(·, ·)")
    ax.axvline(1.0, color="#aaaaaa", linewidth=0.6)
    ax.axvline(1 - eps, color="#117733", linestyle="-.", linewidth=0.8, alpha=0.6)
    ax.axvline(1 + eps, color="#117733", linestyle="-.", linewidth=0.8,
                alpha=0.6, label=f"clip window [1±ε], ε={eps}")
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.4)
    sign = "positive" if adv > 0 else "negative"
    ax.set(
        title=f"Â = {adv:+.1f} ({sign} advantage)",
        xlabel="r_t(θ) = π_θ(a|s) / π_θ_old(a|s)",
        ylabel="Surrogate value",
        xlim=(0.0, 2.0),
    )
    ax.legend(fontsize=8, loc="upper left" if adv > 0 else "lower left")
    ax.grid(alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _panel(axes[0], adv=+1.0)
    _panel(axes[1], adv=-1.0)
    fig.suptitle(
        "PPO clipped surrogate L^CLIP — the slide-10 picture\n"
        "Note the asymmetry: for Â<0 in the right tail (r > 1+ε), the unclipped "
        "branch wins (more negative ⇒ stronger correction pulling policy back).",
        fontsize=11,
    )
    fig.tight_layout()
    out = ROOT / "assets" / "plots" / "clipped_surrogate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
