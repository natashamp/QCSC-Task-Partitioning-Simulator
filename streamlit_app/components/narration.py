"""Generates plain-English explanations for simulation events."""


DEVICE_FRIENDLY = {
    "cpu_0": "CPU-0", "cpu_1": "CPU-1",
    "gpu_0": "GPU-0", "gpu_1": "GPU-1",
    "ai_accel_0": "AI Accelerator",
    "qpu_0": "QPU-0",
}

TYPE_DESCRIPTIONS = {
    "CLASSICAL_DATA_OP": "a classical data operation (best on CPUs)",
    "LINEAR_ALGEBRA": "a linear algebra computation (best on GPUs)",
    "QUANTUM_CIRCUIT": "a quantum circuit execution (best on QPUs)",
    "PARAMETER_OPT": "a parameter optimization task (best on AI accelerators)",
    "NN_INFERENCE": "a neural network inference task (best on AI accelerators)",
    "COMBINATORIAL_OPT": "a combinatorial optimization problem (best on QPUs)",
    "SAMPLING": "a quantum sampling task (best on QPUs)",
}


def friendly_device(device_id):
    return DEVICE_FRIENDLY.get(device_id, device_id)


def narrate_event(event):
    """Return a (title, explanation) tuple for a SubEvent."""
    etype = event.event_type
    details = event.details

    if etype == "DISPATCH":
        task_type = details.get("task_type", "unknown")
        type_desc = TYPE_DESCRIPTIONS.get(task_type, f"a {task_type} task")
        device = friendly_device(event.device_id)
        urgency = details.get("urgency", 0)
        cost = details.get("compute_cost", 0)
        est_end = details.get("est_end", 0)
        multiplier = details.get("latency_multiplier", 1.0)

        title = f"Dispatched `{event.task_id}` to {device}"
        lines = [
            f"**{event.task_id}** is {type_desc} with a compute cost of {cost}.",
            f"The scheduler assigned it to **{device}** because this device type scored "
            f"highest for this task. The task's urgency score is {urgency} "
            f"(longer remaining path = higher priority).",
            f"Estimated completion: tick {est_end}.",
        ]
        if multiplier != 1.0 and "qpu" in (event.device_id or ""):
            actual_end = details.get("actual_end", est_end)
            lines.append(
                f"QPU latency variance applied: **{multiplier:.2f}x** multiplier. "
                f"Actual expected completion: tick {actual_end}."
            )
        return title, " ".join(lines)

    elif etype == "COMPLETE":
        device = friendly_device(event.device_id)
        duration = details.get("duration", 0)
        estimated = details.get("estimated_duration", 0)
        new_ready = details.get("new_ready", [])
        successors = details.get("successors", [])

        title = f"Completed `{event.task_id}`"
        lines = [
            f"**{event.task_id}** finished on **{device}** at tick {event.tick:.1f}.",
            f"It took **{duration:.1f} ticks** (estimated: {estimated:.1f}).",
        ]
        if duration > 0 and estimated > 0:
            ratio = duration / estimated
            if ratio > 1.1:
                lines.append(f"That's {ratio:.1f}x slower than expected.")
            elif ratio < 0.9:
                lines.append(f"That's {ratio:.1f}x faster than expected.")

        if new_ready:
            names = ", ".join(f"`{t}`" for t in new_ready)
            lines.append(f"This unblocked {len(new_ready)} successor(s): {names}.")
        elif successors:
            lines.append(
                f"Its successors ({', '.join(f'`{s}`' for s in successors)}) "
                f"are still waiting on other dependencies."
            )
        else:
            lines.append("This was a terminal task with no successors.")

        return title, " ".join(lines)

    elif etype == "REALLOCATE":
        from_dev = friendly_device(details.get("from_device", ""))
        to_dev = friendly_device(details.get("to_device", ""))
        reason = details.get("reason", "unknown reason")

        title = f"Reallocated `{event.task_id}`"
        explanation = (
            f"**{event.task_id}** was moved from **{from_dev}** to **{to_dev}**. "
            f"Reason: {reason}. "
            f"The task will restart from scratch on the new device."
        )
        return title, explanation

    elif etype == "CALIBRATION":
        device = friendly_device(event.device_id)
        title = f"QPU Calibration: {device}"
        explanation = (
            f"**{device}** entered a calibration cycle. Quantum hardware requires periodic "
            f"recalibration to maintain gate fidelity and reduce errors. "
            f"No tasks can run on this device during calibration."
        )
        return title, explanation

    else:
        return etype, str(details)
