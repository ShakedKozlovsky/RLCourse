"""Extract 4 representative frames from cleaning_episode.gif as PNGs.

TA Mod6 follow-up: the grader can verify the cleaning behaviour without
playing the GIF. The 4 frames are spaced evenly across the episode."""

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIF = ROOT / "assets" / "gifs" / "cleaning_episode.gif"
OUT_DIR = ROOT / "assets" / "diagrams"


def main() -> None:
    reader = imageio.get_reader(GIF)
    frames = list(reader)
    n = len(frames)
    if n < 4:
        raise RuntimeError(f"GIF has only {n} frames; need ≥ 4")
    picks = [0, n // 3, (2 * n) // 3, n - 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ax, idx in zip(axes, picks, strict=True):
        ax.imshow(frames[idx])
        ax.set_title(f"frame {idx} / {n - 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Cleaning-episode key frames — verifiable visual evidence "
                  "(Layer 27)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "cleaning_frames.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n} frames in source GIF; picked {picks})")


if __name__ == "__main__":
    main()
