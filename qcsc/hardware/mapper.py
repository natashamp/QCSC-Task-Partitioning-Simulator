"""Hardware-aware task-to-device mapping algorithm."""

from typing import Optional

from qcsc.config import DeviceType, SchedulerConfig
from qcsc.graph.node import TaskNode
from qcsc.graph.tcg import TensorComputeGraph
from .device import Device
from .cluster import HardwareCluster


class HardwareMapper:
    """Maps tasks to devices using a weighted scoring function."""

    def __init__(self, cluster: HardwareCluster, config: SchedulerConfig):
        self.cluster = cluster
        self.config = config

    def score(self, task: TaskNode, device: Device) -> float:
        """Compute mapping score for a (task, device) pair.

        score = w1 * affinity + w2 * availability + w3 * transfer_fit + w4 * memory_fit
        """
        # Affinity score
        affinity = task.hardware_affinity.get(device.device_type, 0.0)

        # Availability score (inverse of load)
        availability = 1.0 - device.utilization

        # Transfer cost score (simplified: same-type transfer is free)
        transfer_score = 1.0  # Default: no penalty

        # Memory fit score
        if device.device_type == DeviceType.QPU:
            memory_fit = 1.0  # QPU doesn't use classical memory
        elif device.memory_total > 0:
            memory_fit = min(1.0, device.memory_available / max(task.memory_requirement, 0.1))
        else:
            memory_fit = 0.0

        score = (
            self.config.affinity_weight * affinity
            + self.config.load_weight * availability
            + self.config.transfer_weight * transfer_score
            + self.config.memory_weight * memory_fit
        )

        return score

    def map_task(self, task: TaskNode, tcg: TensorComputeGraph) -> Optional[Device]:
        """Find the best device for a given task."""
        best_device = None
        best_score = -1.0

        for device in self.cluster.get_all_available():
            if not device.can_accept(task):
                continue

            s = self.score(task, device)
            if s > best_score:
                best_score = s
                best_device = device

        return best_device

    def map_tasks(
        self, tasks: list[TaskNode], tcg: TensorComputeGraph
    ) -> list[tuple[TaskNode, Device]]:
        """Map a batch of ready tasks to devices, sorted by urgency."""
        # Sort by urgency (longest remaining path)
        tasks_with_urgency = []
        for task in tasks:
            urgency = tcg.node_urgency(task.id)
            tasks_with_urgency.append((urgency, task))
        tasks_with_urgency.sort(key=lambda x: -x[0])  # Most urgent first

        assignments = []
        for _, task in tasks_with_urgency:
            device = self.map_task(task, tcg)
            if device is not None:
                assignments.append((task, device))

        return assignments
