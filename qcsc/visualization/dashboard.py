"""Summary dashboard combining multiple visualization panels."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

from qcsc.config import DeviceType
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.scheduler.metrics import MetricsCollector


DEVICE_COLORS = {
    "CPU": "#4CAF50",
    "GPU": "#2196F3",
    "AI_ACCELERATOR": "#FF9800",
    "QPU": "#9C27B0",
}


class Dashboard:
    """Multi-panel dashboard of simulation results."""

    @staticmethod
    def render(
        tcg: TensorComputeGraph,
        metrics: MetricsCollector,
        output_path: str | Path = "results/dashboard.png",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"QCSC Simulation Dashboard - {tcg.name}", fontsize=16, fontweight="bold")

        Dashboard._plot_utilization(axes[0, 0], metrics)
        Dashboard._plot_latency_distribution(axes[0, 1], metrics)
        Dashboard._plot_reallocation_timeline(axes[1, 0], metrics)
        Dashboard._plot_summary_stats(axes[1, 1], metrics)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Dashboard saved to {output_path}")

    @staticmethod
    def _plot_utilization(ax, metrics: MetricsCollector) -> None:
        """Device utilization over time."""
        if not metrics.utilization_over_time:
            ax.text(0.5, 0.5, "No utilization data", ha="center", va="center")
            ax.set_title("Device Utilization Over Time")
            return

        # Group by device type prefix
        ticks = [entry["tick"] for entry in metrics.utilization_over_time]
        device_ids = [k for k in metrics.utilization_over_time[0].keys() if k != "tick"]

        # Aggregate by device type
        type_data: dict[str, list[float]] = defaultdict(list)
        for entry in metrics.utilization_over_time:
            type_utils: dict[str, list[float]] = defaultdict(list)
            for dev_id in device_ids:
                if dev_id.startswith("cpu"):
                    type_utils["CPU"].append(entry.get(dev_id, 0))
                elif dev_id.startswith("gpu"):
                    type_utils["GPU"].append(entry.get(dev_id, 0))
                elif dev_id.startswith("ai_accel"):
                    type_utils["AI_ACCELERATOR"].append(entry.get(dev_id, 0))
                elif dev_id.startswith("qpu"):
                    type_utils["QPU"].append(entry.get(dev_id, 0))

            for dt_name, utils in type_utils.items():
                type_data[dt_name].append(np.mean(utils) if utils else 0)

        for dt_name, values in type_data.items():
            color = DEVICE_COLORS.get(dt_name, "#999999")
            ax.plot(ticks[:len(values)], values, label=dt_name, color=color, linewidth=2)

        ax.set_xlabel("Simulation Time (ticks)")
        ax.set_ylabel("Utilization")
        ax.set_title("Device Utilization Over Time")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    @staticmethod
    def _plot_latency_distribution(ax, metrics: MetricsCollector) -> None:
        """Box plot of task latencies per device type."""
        type_latencies: dict[str, list[float]] = defaultdict(list)
        for tm in metrics.task_metrics.values():
            if tm.actual_time > 0:
                type_latencies[tm.device_type.name].append(tm.actual_time)

        if not type_latencies:
            ax.text(0.5, 0.5, "No latency data", ha="center", va="center")
            ax.set_title("Task Latency Distribution")
            return

        labels = sorted(type_latencies.keys())
        data = [type_latencies[l] for l in labels]
        colors = [DEVICE_COLORS.get(l, "#999999") for l in labels]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Device Type")
        ax.set_ylabel("Execution Time (ticks)")
        ax.set_title("Task Latency Distribution by Device Type")
        ax.grid(axis="y", alpha=0.3)

    @staticmethod
    def _plot_reallocation_timeline(ax, metrics: MetricsCollector) -> None:
        """Scatter plot of reallocation events."""
        if not metrics.reallocation_events:
            ax.text(0.5, 0.5, "No reallocations occurred", ha="center", va="center",
                    fontsize=12, color="green")
            ax.set_title("Task Reallocation Timeline")
            return

        ticks = [r["tick"] for r in metrics.reallocation_events]
        tasks = [r["task_id"] for r in metrics.reallocation_events]
        task_set = sorted(set(tasks))
        task_y = {t: i for i, t in enumerate(task_set)}
        y_vals = [task_y[t] for t in tasks]

        ax.scatter(ticks, y_vals, c="red", s=100, marker="x", zorder=5)
        for i, r in enumerate(metrics.reallocation_events):
            ax.annotate(
                f"{r['from_device']}->{r['to_device']}",
                (ticks[i], y_vals[i]),
                fontsize=6, textcoords="offset points", xytext=(5, 5),
            )

        ax.set_yticks(range(len(task_set)))
        ax.set_yticklabels(task_set, fontsize=8)
        ax.set_xlabel("Simulation Time (ticks)")
        ax.set_title("Task Reallocation Timeline")
        ax.grid(alpha=0.3)

    @staticmethod
    def _plot_summary_stats(ax, metrics: MetricsCollector) -> None:
        """Summary statistics table."""
        ax.axis("off")
        summary = metrics.summary()

        rows = [
            ["Metric", "Value"],
            ["Makespan", f"{summary.get('makespan', 0):.1f} ticks"],
            ["Total Tasks", str(summary.get("total_tasks", 0))],
            ["Completed Tasks", str(summary.get("completed_tasks", 0))],
            ["Avg Task Latency", f"{summary.get('avg_task_latency', 0):.2f} ticks"],
            ["Reallocations", str(summary.get("reallocation_count", 0))],
            ["Avg Latency Ratio", f"{summary.get('avg_latency_ratio', 0):.2f}x"],
        ]

        per_dt = summary.get("per_device_type", {})
        for dt_name, stats in per_dt.items():
            rows.append([f"{dt_name} Tasks", str(stats["task_count"])])
            rows.append([f"{dt_name} Avg Time", f"{stats['avg_time']:.2f}"])

        table = ax.table(
            cellText=rows[1:],
            colLabels=rows[0],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style header
        for j in range(2):
            table[0, j].set_facecolor("#333333")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title("Summary Statistics", fontsize=12, pad=20)
