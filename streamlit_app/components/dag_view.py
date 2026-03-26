"""Live DAG visualization using Plotly."""

import plotly.graph_objects as go
import networkx as nx


STATUS_COLORS = {
    "PENDING": "rgba(200, 200, 200, 0.4)",
    "READY": "rgba(255, 235, 59, 0.9)",
    "RUNNING": "rgba(255, 152, 0, 1.0)",
    "COMPLETED": "rgba(76, 175, 80, 0.9)",
    "REALLOCATED": "rgba(244, 67, 54, 0.9)",
}

STATUS_SIZES = {
    "PENDING": 20,
    "READY": 25,
    "RUNNING": 30,
    "COMPLETED": 22,
    "REALLOCATED": 25,
}

DEVICE_BORDER_COLORS = {
    "cpu": "#4CAF50",
    "gpu": "#2196F3",
    "ai_accel": "#FF9800",
    "qpu": "#9C27B0",
}


def compute_dag_positions(tcg):
    """Compute node positions using topological generations."""
    G = tcg._graph
    positions = {}
    layers = list(nx.topological_generations(G))
    for layer_idx, layer_nodes in enumerate(layers):
        for i, node_id in enumerate(layer_nodes):
            x = (i - len(layer_nodes) / 2) * 2.5
            y = -layer_idx * 1.8
            positions[node_id] = (x, y)
    return positions


def render_dag(tcg, node_statuses, running_tasks, positions):
    """Return a Plotly figure of the DAG colored by task status."""
    G = tcg._graph

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#999999"),
        hoverinfo="none",
        mode="lines",
    )

    # Build arrow annotations
    annotations = []
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        annotations.append(dict(
            ax=x0, ay=y0,
            x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#999999",
            opacity=0.5,
        ))

    # Build node traces
    node_x, node_y = [], []
    node_colors, node_sizes, node_text, hover_text = [], [], [], []
    node_borders = []

    for node_id in G.nodes():
        x, y = positions[node_id]
        node_x.append(x)
        node_y.append(y)

        status = node_statuses.get(node_id, "PENDING")
        node_colors.append(STATUS_COLORS.get(status, STATUS_COLORS["PENDING"]))
        node_sizes.append(STATUS_SIZES.get(status, 20))
        node_text.append(node_id.replace("_", "\n", 1) if len(node_id) > 12 else node_id)

        # Hover info
        node = tcg.get_node(node_id)
        device_info = ""
        if node.assigned_device:
            device_info = f"<br>Device: {node.assigned_device}"
        running_info = ""
        if node_id in running_tasks:
            rt = running_tasks[node_id]
            running_info = f"<br>Started: tick {rt['start']:.1f}<br>Est end: tick {rt['est_end']:.1f}"

        hover_text.append(
            f"<b>{node_id}</b><br>"
            f"Type: {node.task_type.name}<br>"
            f"Cost: {node.compute_cost}<br>"
            f"Status: {status}"
            f"{device_info}{running_info}"
        )

        # Border color by assigned device
        border = "#333333"
        if node.assigned_device:
            for prefix, color in DEVICE_BORDER_COLORS.items():
                if node.assigned_device.startswith(prefix):
                    border = color
                    break
        node_borders.append(border)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="middle center",
        textfont=dict(size=7, color="black"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color=node_borders),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=30),
            height=400,
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text="Workflow DAG", font=dict(size=14)),
            plot_bgcolor="white",
        ),
    )

    # Add status legend as shapes
    legend_items = [
        ("Pending", STATUS_COLORS["PENDING"]),
        ("Running", STATUS_COLORS["RUNNING"]),
        ("Completed", STATUS_COLORS["COMPLETED"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=1.0, y=1.0 - i * 0.08,
            xref="paper", yref="paper",
            text=f"  {label}",
            showarrow=False,
            font=dict(size=10),
            bgcolor=color,
            borderpad=3,
        )

    return fig
