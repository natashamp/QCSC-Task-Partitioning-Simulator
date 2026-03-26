"""Concept explanations that appear when relevant events occur."""

CONCEPTS = {
    "dag_dependencies": {
        "title": "DAG Dependencies",
        "icon": "info",
        "trigger": "first_step",
        "body": (
            "A **Directed Acyclic Graph (DAG)** represents the workflow as a set of tasks "
            "connected by dependency edges. A task can only start when **all** of its "
            "predecessors have completed. Tasks with no dependencies are called **root tasks** "
            "and can start immediately. The DAG structure determines what can run in parallel "
            "and what must wait."
        ),
    },
    "affinity_scoring": {
        "title": "Affinity-Based Device Mapping",
        "icon": "info",
        "trigger": "first_dispatch",
        "body": (
            "When a task is ready, the scheduler scores every available device using a "
            "**weighted formula**:\n\n"
            "> Score = 0.4 x Affinity + 0.25 x Availability + 0.2 x Transfer + 0.15 x Memory\n\n"
            "**Affinity** measures how well-suited a device type is for this task type "
            "(e.g., quantum circuits have 0.95 affinity for QPUs). "
            "**Availability** prefers less-loaded devices. "
            "**Transfer** accounts for data movement cost. "
            "**Memory** checks if the device has enough free memory. "
            "The device with the highest total score wins."
        ),
    },
    "parallelism": {
        "title": "Task Parallelism",
        "icon": "info",
        "trigger": "parallel_dispatch",
        "body": (
            "When multiple tasks have all their dependencies met at the same time, they can "
            "run **concurrently** on different devices. The **parallelism width** is the maximum "
            "number of tasks that can run at once. More parallelism means more devices stay busy "
            "and the workflow finishes faster. However, the **critical path** (the longest chain "
            "of dependent tasks) sets a hard lower bound on the total makespan, no matter how "
            "many devices are available."
        ),
    },
    "stochastic_latency": {
        "title": "Quantum Hardware Noise",
        "icon": "warning",
        "trigger": "qpu_variance",
        "body": (
            "Quantum processors are **inherently noisy**. Gate errors, decoherence, and "
            "environmental interference mean that the same circuit can take different amounts "
            "of time on each execution. The simulator models this with a **stochastic latency "
            "multiplier** sampled from a normal distribution. A multiplier of 1.3x means the "
            "task took 30% longer than expected. This unpredictability is a key challenge in "
            "hybrid quantum-classical scheduling."
        ),
    },
    "feedback_loop": {
        "title": "Adaptive Feedback & Reallocation",
        "icon": "warning",
        "trigger": "reallocation",
        "body": (
            "The **adaptive feedback loop** monitors running tasks and can **reallocate** them "
            "to different devices mid-execution. Two strategies trigger reallocation:\n\n"
            "1. **QPU Latency Fallback**: If a QPU task runs >1.5x longer than expected, "
            "it gets moved to a GPU (which can simulate quantum circuits classically).\n"
            "2. **Load Balancing**: If one device type is >85% utilized while another is "
            "underloaded, tasks migrate to spread the work.\n\n"
            "Reallocation has a cost: the task **restarts from scratch** on the new device. "
            "That's why there are cooldowns (10 ticks) and limits (max 2 reallocations per task) "
            "to prevent thrashing."
        ),
    },
    "critical_path": {
        "title": "Critical Path",
        "icon": "info",
        "trigger": "critical_path_complete",
        "body": (
            "The **critical path** is the longest chain of dependent tasks through the DAG. "
            "It determines the **theoretical minimum makespan**: even with infinite devices, "
            "you cannot finish the workflow faster than the critical path. Tasks on the critical "
            "path are given higher **urgency** scores, so the scheduler dispatches them first. "
            "Delays on critical-path tasks directly increase the total makespan."
        ),
    },
    "makespan": {
        "title": "Makespan & Results",
        "icon": "success",
        "trigger": "simulation_complete",
        "body": (
            "**Makespan** is the total time from the first task dispatch to the last task "
            "completion. It's the primary metric for scheduling quality. Other key metrics:\n\n"
            "- **Avg Latency Ratio**: actual time / estimated time across all tasks. "
            "Values >1.0 mean tasks took longer than expected (often due to QPU noise).\n"
            "- **Reallocation Count**: how many times the feedback loop moved tasks. "
            "Some reallocations save time; too many waste it.\n"
            "- **Device Utilization**: how busy each device was. Low utilization means "
            "the schedule could be improved."
        ),
    },
    "qpu_calibration": {
        "title": "QPU Calibration",
        "icon": "info",
        "trigger": "calibration",
        "body": (
            "Real quantum processors require periodic **calibration** to maintain accurate "
            "gate operations. During calibration, the QPU cannot execute tasks. The simulator "
            "models this with calibration cycles every 50 ticks. This adds realistic downtime "
            "that the scheduler must work around."
        ),
    },
}


def get_triggered_concepts(sub_events, step_number, already_triggered):
    """Return list of concept keys that should be triggered by the current events."""
    triggered = []

    for key, concept in CONCEPTS.items():
        if key in already_triggered:
            continue

        trigger = concept["trigger"]

        if trigger == "first_step" and step_number == 0:
            triggered.append(key)

        elif trigger == "first_dispatch" and step_number == 0:
            triggered.append(key)

        elif trigger == "parallel_dispatch":
            dispatches = [e for e in sub_events if e.event_type == "DISPATCH"]
            if len(dispatches) >= 2:
                triggered.append(key)

        elif trigger == "qpu_variance":
            for e in sub_events:
                if e.event_type == "DISPATCH" and e.details.get("latency_multiplier", 1.0) != 1.0:
                    mult = e.details["latency_multiplier"]
                    if abs(mult - 1.0) > 0.05:
                        triggered.append(key)
                        break

        elif trigger == "reallocation":
            for e in sub_events:
                if e.event_type == "REALLOCATE":
                    triggered.append(key)
                    break

        elif trigger == "calibration":
            for e in sub_events:
                if e.event_type == "CALIBRATION":
                    triggered.append(key)
                    break

        elif trigger == "simulation_complete":
            # Triggered externally when sim finishes
            pass

        elif trigger == "critical_path_complete":
            # Triggered externally
            pass

    return triggered
