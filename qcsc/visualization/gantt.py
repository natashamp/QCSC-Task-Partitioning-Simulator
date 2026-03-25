"""Gantt chart visualization of task execution across devices."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from qcsc.config import TaskType, DeviceType
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.scheduler.metrics import MetricsCollector


# Color map for task types
TASK_COLORS = {
    TaskType.CLASSICAL_DATA_OP: "#4CAF50",
    TaskType.LINEAR_ALGEBRA: "#2196F3",
    TaskType.QUANTUM_CIRCUIT: "#9C27B0",
    TaskType.PARAMETER_OPT: "#FF9800",
    TaskType.NN_INFERENCE: "#F44336",
    TaskType.COMBINATORIAL_OPT: "#00BCD4",
    TaskType.SAMPLING: "#E91E63",
}

DEVICE_TYPE_COLORS = {
    DeviceType.CPU: "#A5D6A7",
    DeviceType.GPU: "#90CAF9",
    DeviceType.AI_ACCELERATOR: "#FFCC80",
    DeviceType.QPU: "#CE93D8",
}


class GanttChart:
    """Generates a Gantt chart of task execution across devices."""

    @staticmethod
    def render(
        tcg: TensorComputeGraph,
        metrics: MetricsCollector,
        output_path: str | Path = "results/gantt_chart.png",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        task_data = []
        device_ids = set()
        for tm in metrics.task_metrics.values():
            if tm.end_tick > 0:
                task_data.append(tm)
                device_ids.add(tm.device_id)

        if not task_data:
            print("No completed tasks to visualize.")
            return

        device_list = sorted(device_ids)
        device_y = {d: i for i, d in enumerate(device_list)}

        fig, ax = plt.subplots(figsize=(14, max(4, len(device_list) * 0.8)))

        for tm in task_data:
            node = tcg.get_node(tm.task_id)
            y = device_y[tm.device_id]
            width = tm.actual_time
            color = TASK_COLORS.get(node.task_type, "#999999")
            edgecolor = "red" if tm.was_reallocated else "black"
            linewidth = 2 if tm.was_reallocated else 0.5

            ax.barh(
                y, width, left=tm.start_tick, height=0.6,
                color=color, edgecolor=edgecolor, linewidth=linewidth,
                alpha=0.85,
            )
            # Label if bar is wide enough
            if width > 2:
                ax.text(
                    tm.start_tick + width / 2, y,
                    tm.task_id, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                )

        # Reallocation arrows
        for realloc in metrics.reallocation_events:
            if realloc["from_device"] in device_y and realloc["to_device"] in device_y:
                y_from = device_y[realloc["from_device"]]
                y_to = device_y[realloc["to_device"]]
                ax.annotate(
                    "", xy=(realloc["tick"], y_to),
                    xytext=(realloc["tick"], y_from),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                )

        ax.set_yticks(range(len(device_list)))
        ax.set_yticklabels(device_list)
        ax.set_xlabel("Simulation Time (ticks)")
        ax.set_ylabel("Device")
        ax.set_title(f"Task Execution Gantt Chart - {tcg.name}")
        ax.grid(axis="x", alpha=0.3)

        # Legend
        patches = [mpatches.Patch(color=c, label=tt.name) for tt, c in TASK_COLORS.items()]
        patches.append(mpatches.Patch(edgecolor="red", facecolor="none", linewidth=2, label="Reallocated"))
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gantt chart saved to {output_path}")
