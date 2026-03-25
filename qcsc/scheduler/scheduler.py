"""Dynamic task scheduler with priority queue dispatching."""

from dataclasses import dataclass, field
from typing import Optional
import heapq

from qcsc.config import TaskStatus, SchedulerConfig
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.graph.node import TaskNode
from qcsc.hardware.device import Device
from qcsc.hardware.cluster import HardwareCluster
from qcsc.hardware.mapper import HardwareMapper
from .metrics import MetricsCollector


@dataclass(order=True)
class ScheduledTask:
    priority: float  # Negative urgency (heapq is min-heap)
    task_id: str = field(compare=False)
    device_id: str = field(compare=False)
    start_time: float = field(compare=False)
    estimated_end_time: float = field(compare=False)
    actual_end_time: float = field(default=0.0, compare=False)


class DynamicScheduler:
    """Priority-based dynamic scheduler for heterogeneous hardware."""

    def __init__(
        self,
        tcg: TensorComputeGraph,
        cluster: HardwareCluster,
        config: SchedulerConfig,
        metrics: MetricsCollector,
    ):
        self.tcg = tcg
        self.cluster = cluster
        self.config = config
        self.metrics = metrics
        self.mapper = HardwareMapper(cluster, config)
        self.running_tasks: dict[str, ScheduledTask] = {}
        self.completed_count: int = 0

    def schedule_ready_tasks(self, current_tick: float) -> list[ScheduledTask]:
        """Identify ready tasks, map to devices, and dispatch."""
        ready = self.tcg.get_ready_tasks()
        if not ready:
            return []

        assignments = self.mapper.map_tasks(ready, self.tcg)
        scheduled = []

        for task, device in assignments:
            estimated_time = device.estimated_time(task)

            # Dispatch
            task.status = TaskStatus.RUNNING
            task.assigned_device = device.id
            task.start_time = current_tick
            device.allocate(task)

            st = ScheduledTask(
                priority=-self.tcg.node_urgency(task.id),
                task_id=task.id,
                device_id=device.id,
                start_time=current_tick,
                estimated_end_time=current_tick + estimated_time,
            )
            self.running_tasks[task.id] = st
            scheduled.append(st)

            self.metrics.record_task_start(
                task.id, device.id, device.device_type,
                estimated_time, current_tick
            )

        return scheduled

    def complete_task(self, task_id: str, current_tick: float) -> list[str]:
        """Mark a task as completed and return newly ready task IDs."""
        if task_id not in self.running_tasks:
            return []

        st = self.running_tasks.pop(task_id)
        st.actual_end_time = current_tick
        node = self.tcg.get_node(task_id)
        node.status = TaskStatus.COMPLETED
        node.end_time = current_tick

        # Release device
        device = self._find_device(st.device_id)
        if device:
            device.release(node)

        self.metrics.record_task_complete(task_id, current_tick)
        self.completed_count += 1

        # Find newly ready successors
        new_ready = []
        for succ_id in self.tcg.get_successors(task_id):
            succ = self.tcg.get_node(succ_id)
            if succ.status == TaskStatus.PENDING:
                preds = self.tcg.get_predecessors(succ_id)
                if all(self.tcg.get_node(p).status == TaskStatus.COMPLETED for p in preds):
                    new_ready.append(succ_id)

        return new_ready

    def get_tasks_completing_at(self, tick: float) -> list[str]:
        """Get tasks whose estimated end time is at or before the given tick."""
        completing = []
        for task_id, st in self.running_tasks.items():
            if st.actual_end_time > 0 and st.actual_end_time <= tick:
                completing.append(task_id)
            elif st.estimated_end_time <= tick and st.actual_end_time == 0:
                completing.append(task_id)
        return completing

    def get_next_event_time(self) -> Optional[float]:
        """Get the earliest completion time among running tasks."""
        if not self.running_tasks:
            return None
        times = []
        for st in self.running_tasks.values():
            t = st.actual_end_time if st.actual_end_time > 0 else st.estimated_end_time
            times.append(t)
        return min(times) if times else None

    def reallocate_task(
        self, task_id: str, new_device: Device, current_tick: float
    ) -> Optional[ScheduledTask]:
        """Cancel a running task and restart it on a new device."""
        if task_id not in self.running_tasks:
            return None

        old_st = self.running_tasks.pop(task_id)
        node = self.tcg.get_node(task_id)

        # Release old device
        old_device = self._find_device(old_st.device_id)
        if old_device:
            old_device.release(node)

        # Re-assign
        estimated_time = new_device.estimated_time(node)
        node.status = TaskStatus.RUNNING
        node.assigned_device = new_device.id
        node.start_time = current_tick
        node.reallocation_count += 1
        node.last_reallocation_time = current_tick
        new_device.allocate(node)

        new_st = ScheduledTask(
            priority=-self.tcg.node_urgency(task_id),
            task_id=task_id,
            device_id=new_device.id,
            start_time=current_tick,
            estimated_end_time=current_tick + estimated_time,
        )
        self.running_tasks[task_id] = new_st

        self.metrics.record_reallocation(
            task_id, old_st.device_id, new_device.id, current_tick,
            reason="adaptive_feedback"
        )
        self.metrics.record_task_start(
            task_id, new_device.id, new_device.device_type,
            estimated_time, current_tick
        )

        return new_st

    def _find_device(self, device_id: str) -> Optional[Device]:
        for d in self.cluster.devices:
            if d.id == device_id:
                return d
        return None
