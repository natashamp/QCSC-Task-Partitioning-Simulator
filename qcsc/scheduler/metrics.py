"""Runtime metrics collection for the simulation."""

from dataclasses import dataclass, field
from typing import Optional

from qcsc.config import DeviceType


@dataclass
class TaskMetric:
    task_id: str
    device_id: str
    device_type: DeviceType
    estimated_time: float
    actual_time: float = 0.0
    start_tick: float = 0.0
    end_tick: float = 0.0
    was_reallocated: bool = False
    reallocation_count: int = 0


@dataclass
class DeviceSnapshot:
    tick: float
    device_id: str
    device_type: DeviceType
    utilization: float
    tasks_running: int
    tasks_completed: int


class MetricsCollector:
    """Collects and aggregates simulation metrics."""

    def __init__(self):
        self.task_metrics: dict[str, TaskMetric] = {}
        self.device_history: list[DeviceSnapshot] = []
        self.reallocation_events: list[dict] = []
        self.utilization_over_time: list[dict[str, float]] = []

    def record_task_start(
        self, task_id: str, device_id: str, device_type: DeviceType,
        estimated_time: float, tick: float
    ) -> None:
        self.task_metrics[task_id] = TaskMetric(
            task_id=task_id,
            device_id=device_id,
            device_type=device_type,
            estimated_time=estimated_time,
            start_tick=tick,
        )

    def record_task_complete(self, task_id: str, tick: float) -> None:
        if task_id in self.task_metrics:
            m = self.task_metrics[task_id]
            m.end_tick = tick
            m.actual_time = tick - m.start_tick

    def record_reallocation(
        self, task_id: str, from_device: str, to_device: str, tick: float, reason: str
    ) -> None:
        self.reallocation_events.append({
            "task_id": task_id,
            "from_device": from_device,
            "to_device": to_device,
            "tick": tick,
            "reason": reason,
        })
        if task_id in self.task_metrics:
            self.task_metrics[task_id].was_reallocated = True
            self.task_metrics[task_id].reallocation_count += 1

    def record_device_snapshot(self, devices, tick: float) -> None:
        snapshot = {}
        for device in devices:
            self.device_history.append(DeviceSnapshot(
                tick=tick,
                device_id=device.id,
                device_type=device.device_type,
                utilization=device.utilization,
                tasks_running=device.current_load,
                tasks_completed=device.total_tasks_executed,
            ))
            snapshot[device.id] = device.utilization
        self.utilization_over_time.append({"tick": tick, **snapshot})

    def summary(self) -> dict:
        if not self.task_metrics:
            return {"makespan": 0, "total_tasks": 0}

        completed = [m for m in self.task_metrics.values() if m.end_tick > 0]
        if not completed:
            return {"makespan": 0, "total_tasks": len(self.task_metrics)}

        makespan = max(m.end_tick for m in completed)
        total_actual = sum(m.actual_time for m in completed)
        avg_latency = total_actual / len(completed) if completed else 0

        # Per-device-type stats
        per_device_type: dict[str, list[float]] = {}
        for m in completed:
            key = m.device_type.name
            per_device_type.setdefault(key, []).append(m.actual_time)

        device_stats = {}
        for dt_name, times in per_device_type.items():
            device_stats[dt_name] = {
                "task_count": len(times),
                "avg_time": sum(times) / len(times),
                "total_time": sum(times),
            }

        latency_ratios = [
            m.actual_time / m.estimated_time
            for m in completed if m.estimated_time > 0
        ]

        return {
            "makespan": makespan,
            "total_tasks": len(self.task_metrics),
            "completed_tasks": len(completed),
            "avg_task_latency": avg_latency,
            "reallocation_count": len(self.reallocation_events),
            "avg_latency_ratio": sum(latency_ratios) / len(latency_ratios) if latency_ratios else 0,
            "per_device_type": device_stats,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"Makespan:            {s['makespan']:.1f} ticks")
        print(f"Total tasks:         {s['total_tasks']}")
        print(f"Completed tasks:     {s.get('completed_tasks', 0)}")
        print(f"Avg task latency:    {s.get('avg_task_latency', 0):.2f} ticks")
        print(f"Reallocations:       {s.get('reallocation_count', 0)}")
        print(f"Avg latency ratio:   {s.get('avg_latency_ratio', 0):.2f}x")

        if "per_device_type" in s:
            print("\nPer-device-type breakdown:")
            for dt_name, stats in s["per_device_type"].items():
                print(f"  {dt_name:16s}: {stats['task_count']} tasks, "
                      f"avg={stats['avg_time']:.2f}, total={stats['total_time']:.1f}")
        print("=" * 60)
