"""Tensor Compute Graph visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

from qcsc.config import DeviceType
from qcsc.graph.tcg import TensorComputeGraph


DEVICE_COLORS = {
    DeviceType.CPU: "#4CAF50",
    DeviceType.GPU: "#2196F3",
    DeviceType.AI_ACCELERATOR: "#FF9800",
    DeviceType.QPU: "#9C27B0",
    None: "#CCCCCC",
}


class GraphVisualizer:
    """Visualizes the Tensor Compute Graph with device assignment coloring."""

    @staticmethod
    def render(
        tcg: TensorComputeGraph,
        output_path: str | Path = "results/tcg_graph.png",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        G = tcg._graph
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use layered layout for DAG
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            for layer, nodes in enumerate(nx.topological_generations(G)):
                for i, node in enumerate(nodes):
                    if node not in G:
                        continue
            pos = {}
            layers = list(nx.topological_generations(G))
            for layer_idx, layer_nodes in enumerate(layers):
                for i, node_id in enumerate(layer_nodes):
                    x = (i - len(layer_nodes) / 2) * 2.0
                    y = -layer_idx * 2.0
                    pos[node_id] = (x, y)

        # Color nodes by assigned device
        node_colors = []
        for node_id in G.nodes():
            node = tcg.get_node(node_id)
            if node.assigned_device:
                # Determine device type from device id
                if "cpu" in node.assigned_device:
                    node_colors.append(DEVICE_COLORS[DeviceType.CPU])
                elif "gpu" in node.assigned_device:
                    node_colors.append(DEVICE_COLORS[DeviceType.GPU])
                elif "ai_accel" in node.assigned_device:
                    node_colors.append(DEVICE_COLORS[DeviceType.AI_ACCELERATOR])
                elif "qpu" in node.assigned_device:
                    node_colors.append(DEVICE_COLORS[DeviceType.QPU])
                else:
                    node_colors.append(DEVICE_COLORS[None])
            else:
                node_colors.append(DEVICE_COLORS[None])

        # Draw
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=800,
            edgecolors="black", linewidths=1.5, ax=ax,
        )
        nx.draw_networkx_labels(
            G, pos, font_size=7, font_weight="bold", ax=ax,
        )

        # Edge widths based on data size
        edge_widths = []
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            edge_widths.append(max(0.5, min(4.0, w)))

        nx.draw_networkx_edges(
            G, pos, width=edge_widths, edge_color="#666666",
            arrows=True, arrowsize=15, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1", ax=ax,
        )

        # Legend
        patches = [mpatches.Patch(color=c, label=dt.name if dt else "Unassigned")
                   for dt, c in DEVICE_COLORS.items()]
        ax.legend(handles=patches, loc="upper left", fontsize=9)

        ax.set_title(f"Tensor Compute Graph - {tcg.name}", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graph visualization saved to {output_path}")
