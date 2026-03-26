"""Incremental Gantt chart that grows as tasks complete."""

import plotly.graph_objects as go


TASK_TYPE_COLORS = {
    "CLASSICAL_DATA_OP": "#4CAF50",
    "LINEAR_ALGEBRA": "#2196F3",
    "QUANTUM_CIRCUIT": "#9C27B0",
    "PARAMETER_OPT": "#FF9800",
    "NN_INFERENCE": "#F44336",
    "COMBINATORIAL_OPT": "#00BCD4",
    "SAMPLING": "#E91E63",
}


def render_gantt(gantt_data, running_tasks, current_tick, tcg):
    """Build an incremental Gantt chart from completed + running tasks.

    gantt_data: list of dicts with keys: task_id, device_id, start, end, task_type, reallocated
    running_tasks: dict of task_id -> {device_id, start, est_end, actual_end}
    """
    fig = go.Figure()

    # Collect all devices for y-axis ordering
    all_devices = sorted(set(
        [d["device_id"] for d in gantt_data]
        + [info["device_id"] for info in running_tasks.values()]
    ))
    device_y = {d: i for i, d in enumerate(all_devices)}

    # Completed tasks (solid bars)
    for entry in gantt_data:
        y = device_y.get(entry["device_id"], 0)
        color = TASK_TYPE_COLORS.get(entry["task_type"], "#999")
        duration = entry["end"] - entry["start"]
        border_color = "red" if entry.get("reallocated") else color
        border_width = 3 if entry.get("reallocated") else 1

        fig.add_trace(go.Bar(
            x=[duration],
            y=[entry["device_id"]],
            base=[entry["start"]],
            orientation="h",
            marker=dict(
                color=color,
                line=dict(color=border_color, width=border_width),
                opacity=0.85,
            ),
            text=entry["task_id"] if duration > 1.5 else "",
            textposition="inside",
            textfont=dict(size=9, color="white"),
            hovertext=(
                f"<b>{entry['task_id']}</b><br>"
                f"Type: {entry['task_type']}<br>"
                f"Device: {entry['device_id']}<br>"
                f"Start: {entry['start']:.1f}<br>"
                f"End: {entry['end']:.1f}<br>"
                f"Duration: {duration:.1f} ticks"
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    # Running tasks (striped/lighter bars showing progress so far)
    for task_id, info in running_tasks.items():
        y = device_y.get(info["device_id"], 0)
        elapsed = current_tick - info["start"]
        node = tcg.get_node(task_id)
        color = TASK_TYPE_COLORS.get(node.task_type.name, "#999")

        if elapsed > 0:
            fig.add_trace(go.Bar(
                x=[elapsed],
                y=[info["device_id"]],
                base=[info["start"]],
                orientation="h",
                marker=dict(
                    color=color,
                    opacity=0.4,
                    line=dict(color=color, width=2),
                    pattern=dict(shape="/"),
                ),
                text=f"{task_id} ...",
                textposition="inside",
                textfont=dict(size=8, color="black"),
                hovertext=(
                    f"<b>{task_id}</b> (running)<br>"
                    f"Device: {info['device_id']}<br>"
                    f"Started: {info['start']:.1f}<br>"
                    f"Elapsed: {elapsed:.1f} ticks"
                ),
                hoverinfo="text",
                showlegend=False,
            ))

    fig.update_layout(
        barmode="overlay",
        height=max(200, len(all_devices) * 60 + 80),
        margin=dict(l=10, r=10, t=30, b=30),
        title=dict(text="Execution Timeline", font=dict(size=14)),
        xaxis=dict(title="Simulation Time (ticks)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
    )

    # Add a vertical line at current tick
    if current_tick > 0:
        fig.add_vline(
            x=current_tick, line_dash="dash",
            line_color="red", line_width=1,
            annotation_text=f"t={current_tick:.1f}",
            annotation_position="top",
        )

    # Add legend for task types as annotations
    legend_y = 1.0
    for ttype, color in TASK_TYPE_COLORS.items():
        fig.add_annotation(
            x=1.02, y=legend_y,
            xref="paper", yref="paper",
            text=f'<span style="color:{color}">&#9632;</span> {ttype}',
            showarrow=False,
            font=dict(size=8),
            align="left",
        )
        legend_y -= 0.08

    return fig
