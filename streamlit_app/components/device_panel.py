"""Device status panel showing utilization and current tasks."""

import streamlit as st


DEVICE_COLORS = {
    "CPU": "#4CAF50",
    "GPU": "#2196F3",
    "AI_ACCELERATOR": "#FF9800",
    "QPU": "#9C27B0",
}

DEVICE_ICONS = {
    "CPU": "CPU",
    "GPU": "GPU",
    "AI_ACCELERATOR": "AI Accel",
    "QPU": "QPU",
}


def render_device_panel(device_states, running_tasks):
    """Render device status cards."""
    st.markdown("#### Devices")

    for dev in device_states:
        color = DEVICE_COLORS.get(dev["type"], "#999")
        label = DEVICE_ICONS.get(dev["type"], dev["type"])
        util_pct = int(dev["utilization"] * 100)

        # Find tasks running on this device
        tasks_on_device = [
            tid for tid, info in running_tasks.items()
            if info["device_id"] == dev["id"]
        ]

        with st.container():
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 4px 8px; '
                f'margin-bottom: 6px; background: #f8f9fa; border-radius: 4px;">'
                f'<strong>{dev["name"]}</strong> '
                f'<span style="color: #666; font-size: 0.85em;">({label})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(
                    min(dev["utilization"], 1.0),
                    text=f"Load: {dev['load']}/{dev['capacity']}"
                )
            with col2:
                st.markdown(f"**{util_pct}%**")

            if tasks_on_device:
                task_list = ", ".join(f"`{t}`" for t in tasks_on_device)
                st.caption(f"Running: {task_list}")

            if dev["memory_total"] > 0:
                mem_pct = dev["memory_used"] / dev["memory_total"] if dev["memory_total"] > 0 else 0
                st.caption(
                    f"Memory: {dev['memory_used']:.0f} / {dev['memory_total']:.0f} MB "
                    f"({mem_pct:.0%})"
                )
