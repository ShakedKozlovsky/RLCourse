"""Render the architecture diagram from PLAN.md as a real PNG.

Layered architecture: interface → sdk → services → {model, environment, memory,
simulator, sensor, noise} → {data, shared} + tools/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _box(ax, x, y, w, h, text, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                        facecolor=color, edgecolor="#222", linewidth=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#222")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Tier colours
    iface = "#cbe5ff"
    sdk = "#c6e8b3"
    svc = "#ffd8a8"
    domain = "#ffe0b8"
    leaf = "#f4d3d3"

    # Tier 1 — Interface
    _box(ax, 2, 7.8, 4, 0.7, "interface/cli/  (Click)", iface)
    _box(ax, 7, 7.8, 4, 0.7, "interface/gui/  (PyQt6)", iface)
    # Tier 2 — SDK
    _box(ax, 3.5, 6.6, 6, 0.7, "sdk/  ·  RoombaLab · env_builder · trainers · experiments", sdk)
    # Tier 3 — Services
    _box(ax, 0.5, 5.4, 4, 0.7, "services/ddpg_update", svc)
    _box(ax, 4.7, 5.4, 4, 0.7, "services/ddpg_service", svc)
    _box(ax, 8.9, 5.4, 3.8, 0.7, "services/evaluation", svc)
    # Tier 4 — Domain
    _box(ax, 0.5, 4.0, 3.6, 0.8, "model/\nactor · critic · soft_update", domain)
    _box(ax, 4.4, 4.0, 4.0, 0.8, "environment/\nroomba_env · reward (no gym!)", domain)
    _box(ax, 8.7, 4.0, 4.0, 0.8, "memory/\nreplay_buffer", domain)
    # Tier 5 — Domain leaves
    _box(ax, 0.5, 2.6, 3.6, 0.8, "simulator/\nkinematics · world · collision · robot", leaf)
    _box(ax, 4.4, 2.6, 4.0, 0.8, "sensor/\nlidar (shapely ray-cast)", leaf)
    _box(ax, 8.7, 2.6, 4.0, 0.8, "noise/\nGaussian · OU · schedule", leaf)
    # Tier 6 — Bottom
    _box(ax, 0.5, 1.2, 6.0, 0.8, "data/\nhouseexpo_loader (10 sample apartments)", "#f0f0f0")
    _box(ax, 6.8, 1.2, 5.9, 0.8, "shared/\nconfig · logger · seed · types · version", "#f0f0f0")
    # Tools (off to the side)
    _box(ax, 0.2, 0.0, 6.0, 0.7, "tools/graphify  (98-node Obsidian wiki)", "#e3d4f6")
    _box(ax, 6.5, 0.0, 6.3, 0.7, "tools/viz  (plots + GIF recorder)", "#e3d4f6")

    # Title + arrow
    ax.text(6.5, 8.7, "roomba-lab — layered architecture",
             ha="center", fontsize=14, fontweight="bold")
    ax.annotate("", xy=(0.2, 0.9), xytext=(0.2, 7.7),
                  arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1.5})
    ax.text(0.05, 4.3, "imports flow down\n(reverse-blocked)", rotation=90,
             ha="center", va="center", fontsize=8, color="#666")

    out = ROOT / "assets" / "diagrams" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
