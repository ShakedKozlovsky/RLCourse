"""Render docs/wiki/graph.json as a node-link diagram PNG.

Closes the loop on the Active Knowledge Architecture: the lecturer's
methodology produces an actual visual artefact you can stare at.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _node_colour(layer: str) -> str:
    return {
        "interface": "#dd9988",
        "sdk": "#fff5e6",
        "services": "#e8f5e9",
        "model": "#fde7f3",
        "environment": "#e6e6fa",
        "tools": "#fff0ee",
        "shared": "#dddddd",
    }.get(layer, "#cccccc")


def _layer_of(module: str) -> str:
    parts = module.split(".")
    if len(parts) < 2:
        return "shared"
    return parts[1]


def main() -> None:
    blob = json.loads((ROOT / "docs" / "wiki" / "graph.json").read_text())
    g = nx.DiGraph()
    for n in blob["nodes"]:
        if n["type"] != "module":
            continue
        g.add_node(n["module"], layer=_layer_of(n["module"]))
    for e in blob["edges"]:
        if e["type"] != "imports":
            continue
        src = e["src"].removeprefix("module:")
        dst = e["dst"].removeprefix("module:")
        if src in g.nodes and dst in g.nodes:
            g.add_edge(src, dst)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(g, k=2.5, seed=42, iterations=80)
    colours = [_node_colour(_layer_of(n)) for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, edge_color="#888",
                            arrows=True, arrowsize=10, width=0.6)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours,
                            node_size=2500, edgecolors="#333")
    labels = {n: n.split(".")[-1] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    layers = sorted({_layer_of(n) for n in g.nodes})
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_node_colour(layer), ec="#333",
                       label=layer)
        for layer in layers
    ]
    ax.legend(handles=handles, loc="upper left", title="Layer")
    ax.set_title(f"proximal_lab — auto-generated architecture diagram\n"
                  f"{g.number_of_nodes()} modules, {g.number_of_edges()} import edges "
                  f"(emitted by tools/graphify)",
                  fontsize=12)
    ax.axis("off")
    out = ROOT / "assets" / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'architecture.png'} — {g.number_of_nodes()} modules")


if __name__ == "__main__":
    main()
