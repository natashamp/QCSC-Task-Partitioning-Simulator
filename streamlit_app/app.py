"""QCSC Interactive Learning Simulator — Streamlit App."""

import sys
from pathlib import Path
import time

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from qcsc.parser import WorkflowParser, CostAnalyzer
from qcsc.hardware import HardwareCluster
from qcsc.config import SchedulerConfig

from streamlit_app.engine_stepper import SteppableSimulation
from streamlit_app.components.dag_view import render_dag, compute_dag_positions
from streamlit_app.components.device_panel import render_device_panel
from streamlit_app.components.gantt_builder import render_gantt
from streamlit_app.components.narration import narrate_event
from streamlit_app.components.score_breakdown import render_score_breakdown
from streamlit_app.components.concept_panel import render_concept_panels
from streamlit_app.concepts.definitions import CONCEPTS

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QCSC Simulator",
    page_icon="atom_symbol",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("QCSC Simulator")
    st.caption("Quantum-Centric Supercomputing\nTask Scheduling Learning Tool")
    st.divider()

    workflow_dir = Path(PROJECT_ROOT) / "workflows"
    workflow_files = sorted(workflow_dir.glob("*.yaml"))
    workflow_names = [f.stem.replace("_", " ").title() for f in workflow_files]
    workflow_choice = st.selectbox(
        "Workflow",
        options=range(len(workflow_files)),
        format_func=lambda i: workflow_names[i],
    )
    workflow_path = str(workflow_files[workflow_choice])

    st.divider()
    enable_feedback = st.toggle("Adaptive Feedback Loop", value=True)
    latency_variance = st.slider(
        "QPU Latency Variance", 0.0, 1.0, 0.3, 0.05,
        help="Standard deviation of the QPU latency multiplier. 0 = deterministic."
    )
    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

    st.divider()
    mode = st.radio("Mode", ["Interactive", "Guided"], index=0)
    if mode == "Guided":
        auto_speed = st.slider("Auto-play speed (seconds)", 0.5, 5.0, 1.5, 0.5)

    st.divider()
    if st.button("Reset Simulation", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("sim_"):
                del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────
def init_simulation():
    """Parse workflow, build hardware, create stepper."""
    tcg = WorkflowParser.parse(workflow_path)
    cluster = HardwareCluster.default_cluster()
    config = SchedulerConfig(
        qpu_latency_variance=latency_variance,
        random_seed=int(seed),
    )

    cost_report = CostAnalyzer.analyze_cost(tcg)
    par_report = CostAnalyzer.analyze_parallelism(tcg)

    sim = SteppableSimulation(tcg, cluster, config, enable_feedback=enable_feedback)
    positions = compute_dag_positions(tcg)

    st.session_state["sim_instance"] = sim
    st.session_state["sim_tcg"] = tcg
    st.session_state["sim_positions"] = positions
    st.session_state["sim_cost_report"] = cost_report
    st.session_state["sim_par_report"] = par_report
    st.session_state["sim_steps"] = []
    st.session_state["sim_gantt_data"] = []
    st.session_state["sim_triggered_concepts"] = set()
    st.session_state["sim_initialized"] = False
    st.session_state["sim_finished"] = False
    st.session_state["sim_auto_playing"] = False


if "sim_instance" not in st.session_state:
    init_simulation()

# Check if config changed
current_config_key = f"{workflow_path}|{enable_feedback}|{latency_variance}|{seed}"
if st.session_state.get("sim_config_key") != current_config_key:
    init_simulation()
    st.session_state["sim_config_key"] = current_config_key

sim = st.session_state["sim_instance"]
tcg = st.session_state["sim_tcg"]
positions = st.session_state["sim_positions"]
steps = st.session_state["sim_steps"]
gantt_data = st.session_state["sim_gantt_data"]

# ──────────────────────────────────────────────────────────────
# Step execution
# ──────────────────────────────────────────────────────────────
def do_step():
    """Execute one simulation step."""
    if st.session_state.get("sim_finished"):
        return

    if not st.session_state.get("sim_initialized"):
        result = sim.initialize()
        st.session_state["sim_initialized"] = True
    else:
        result = sim.step()

    steps.append(result)

    # Update gantt data with completed tasks
    for task_id in result.newly_completed:
        node = tcg.get_node(task_id)
        # Find the task's timing from metrics
        tm = sim.metrics.task_metrics.get(task_id)
        if tm and tm.end_tick > 0:
            gantt_data.append({
                "task_id": task_id,
                "device_id": tm.device_id,
                "start": tm.start_tick,
                "end": tm.end_tick,
                "task_type": node.task_type.name,
                "reallocated": tm.was_reallocated,
            })

    if result.is_finished:
        st.session_state["sim_finished"] = True


# ──────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────
st.markdown("## QCSC Task Scheduling Simulator")

# Pre-simulation overview
if not steps:
    cost = st.session_state["sim_cost_report"]
    par = st.session_state["sim_par_report"]

    st.markdown(f"### Workflow: {tcg.name}")
    st.markdown(
        f"**{cost.task_count} tasks** ({cost.classical_task_count} classical, "
        f"{cost.quantum_task_count} quantum) | "
        f"**Critical path:** {par.critical_path_length:.0f} | "
        f"**Max parallelism:** {par.max_parallelism_width} | "
        f"**Total cost:** {cost.total_cost:.0f}"
    )
    st.info(
        "This simulator walks you through how a hybrid quantum-classical workflow "
        "gets scheduled across heterogeneous hardware. Press **Next Step** to begin, "
        "and read the explanations to understand each decision the scheduler makes."
    )

# Controls row
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])

with ctrl_col1:
    step_btn = st.button(
        "Next Step" if steps else "Start Simulation",
        disabled=st.session_state.get("sim_finished", False),
        type="primary",
    )

with ctrl_col2:
    if mode == "Guided" and steps and not st.session_state.get("sim_finished"):
        auto_btn = st.button("Auto-Play")
    else:
        auto_btn = False

with ctrl_col3:
    if steps:
        current = steps[-1]
        st.metric("Tick", f"{current.tick:.1f}")

with ctrl_col4:
    if steps:
        current = steps[-1]
        completed = sum(1 for s in current.node_statuses.values() if s == "COMPLETED")
        total = len(current.node_statuses)
        st.progress(completed / total if total else 0, text=f"Progress: {completed}/{total} tasks")

# Execute step
if step_btn:
    do_step()
    st.rerun()

if auto_btn:
    st.session_state["sim_auto_playing"] = True

# Auto-play loop
if st.session_state.get("sim_auto_playing") and not st.session_state.get("sim_finished"):
    placeholder = st.empty()
    with placeholder.container():
        st.caption("Auto-playing... press Stop to pause")
        stop_btn = st.button("Stop")
        if stop_btn:
            st.session_state["sim_auto_playing"] = False
            st.rerun()

    do_step()
    time.sleep(auto_speed if mode == "Guided" else 1.0)
    st.rerun()
elif st.session_state.get("sim_auto_playing"):
    st.session_state["sim_auto_playing"] = False

# ──────────────────────────────────────────────────────────────
# Display current state
# ──────────────────────────────────────────────────────────────
if steps:
    current = steps[-1]

    # Concept panels (triggered by events)
    st.session_state["sim_triggered_concepts"] = render_concept_panels(
        current.sub_events,
        current.step_number,
        st.session_state["sim_triggered_concepts"],
    )

    # Completion concept
    if current.is_finished and "makespan" not in st.session_state["sim_triggered_concepts"]:
        summary = sim.metrics.summary()
        st.success(
            f"**Simulation Complete!**\n\n"
            f"**Makespan:** {summary.get('makespan', 0):.1f} ticks | "
            f"**Tasks:** {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)} | "
            f"**Reallocations:** {summary.get('reallocation_count', 0)} | "
            f"**Avg Latency Ratio:** {summary.get('avg_latency_ratio', 0):.2f}x\n\n"
            + CONCEPTS["makespan"]["body"]
        )
        st.session_state["sim_triggered_concepts"].add("makespan")

    # Main content: DAG + Devices
    dag_col, device_col = st.columns([3, 2])

    with dag_col:
        dag_fig = render_dag(tcg, current.node_statuses, current.running_tasks, positions)
        st.plotly_chart(dag_fig, use_container_width=True)

    with device_col:
        render_device_panel(current.device_states, current.running_tasks)

    # Gantt chart
    gantt_fig = render_gantt(gantt_data, current.running_tasks, current.tick, tcg)
    st.plotly_chart(gantt_fig, use_container_width=True)

    # Event narration
    st.markdown("---")
    st.markdown(f"#### Step {current.step_number} — Tick {current.tick:.1f}")

    for event in current.sub_events:
        title, explanation = narrate_event(event)
        icon = {
            "DISPATCH": "arrow_forward",
            "COMPLETE": "check_circle",
            "REALLOCATE": "swap_horiz",
            "CALIBRATION": "build",
        }.get(event.event_type, "info")

        with st.expander(f":{icon}: {title}", expanded=True):
            st.markdown(explanation)

            # Show score breakdown for dispatch events
            if event.event_type == "DISPATCH" and event.task_id in current.score_breakdowns:
                render_score_breakdown(
                    event.task_id,
                    current.score_breakdowns[event.task_id],
                    event.device_id,
                )

    # Event history (collapsed)
    if len(steps) > 1:
        with st.expander(f"Event History ({sum(len(s.sub_events) for s in steps)} events)"):
            for step in reversed(steps[:-1]):
                for event in step.sub_events:
                    title, _ = narrate_event(event)
                    emoji = {
                        "DISPATCH": "->", "COMPLETE": "[done]",
                        "REALLOCATE": "<->", "CALIBRATION": "[cal]",
                    }.get(event.event_type, "")
                    st.caption(f"t={event.tick:.1f} {emoji} {title}")
