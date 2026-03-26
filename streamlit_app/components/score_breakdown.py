"""Score breakdown table showing why a task was assigned to a specific device."""

import streamlit as st
import plotly.graph_objects as go


def render_score_breakdown(task_id, scores, selected_device_id):
    """Show a table and bar chart of device scores for a dispatched task.

    scores: dict of device_id -> {affinity, availability, transfer, memory, total}
    """
    if not scores:
        return

    st.markdown(f"**Why `{task_id}` was assigned to `{selected_device_id}`:**")

    # Build table data
    rows = []
    for dev_id, s in sorted(scores.items(), key=lambda x: -x[1]["total"]):
        is_selected = dev_id == selected_device_id
        rows.append({
            "Device": f"**{dev_id}**" if is_selected else dev_id,
            "Affinity": f"{s['affinity']:.2f}",
            "Availability": f"{s['availability']:.2f}",
            "Transfer": f"{s['transfer']:.2f}",
            "Memory": f"{s['memory']:.2f}",
            "Total": f"**{s['total']:.3f}**" if is_selected else f"{s['total']:.3f}",
        })

    # Render as markdown table
    header = "| Device | Affinity (0.4) | Avail (0.25) | Transfer (0.2) | Memory (0.15) | **Total** |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['Device']} | {r['Affinity']} | {r['Availability']} "
            f"| {r['Transfer']} | {r['Memory']} | {r['Total']} |"
        )
    st.markdown("\n".join(lines))

    # Bar chart comparing totals
    device_ids = [d for d in sorted(scores.keys(), key=lambda x: -scores[x]["total"])]
    totals = [scores[d]["total"] for d in device_ids]
    colors = ["#FF6B35" if d == selected_device_id else "#CCCCCC" for d in device_ids]

    fig = go.Figure(go.Bar(
        x=device_ids,
        y=totals,
        marker_color=colors,
        text=[f"{t:.3f}" for t in totals],
        textposition="outside",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Score", range=[0, max(totals) * 1.2] if totals else [0, 1]),
        xaxis=dict(title=""),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
