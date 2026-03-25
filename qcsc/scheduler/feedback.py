"""Adaptive feedback loop for dynamic task reallocation."""

from dataclasses import dataclass
from typing import Optional

from qcsc.config import DeviceType, SchedulerConfig, TaskStatus
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.hardware.cluster import HardwareCluster
from qcsc.hardware.device import Device
from .metrics import MetricsCollector


@dataclass
class ReallocationDecision:
    task_id: str
    from_device_id: str
    to_device: Device
    reason: str


class AdaptiveFeedbackLoop:
    """Monitors system metrics and triggers task reallocation to avoid bottlenecks."""

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
        self.last_check_tick: float = 0.0

    def should_check(self, current_tick: float) -> bool:
        return (current_tick - self.last_check_tick) >= self.config.feedback_interval

    def check_and_reallocate(
        self, running_tasks: dict, current_tick: float
    ) -> list[ReallocationDecision]:
        """Analyze system state and decide if any running tasks should be reallocated."""
        self.last_check_tick = current_tick
        decisions = []

        # 1. Check QPU latency variance
        decisions.extend(self._check_qpu_latency(running_tasks, current_tick))

        # 2. Check load imbalance
        decisions.extend(self._check_load_balance(running_tasks, current_tick))

        return decisions

    def _check_qpu_latency(
        self, running_tasks: dict, current_tick: float
    ) -> list[ReallocationDecision]:
        """Detect QPU tasks running much longer than expected and move to GPU fallback."""
        decisions = []

        for task_id, scheduled_task in list(running_tasks.items()):
            node = self.tcg.get_node(task_id)

            # Only check QPU tasks
            device = self._find_device(scheduled_task.device_id)
            if device is None or device.device_type != DeviceType.QPU:
                continue

            # Check cooldown and max reallocations
            if node.reallocation_count >= self.config.max_reallocations_per_task:
                continue
            if (current_tick - node.last_reallocation_time) < self.config.reallocation_cooldown:
                continue

            # Check if task is running significantly longer than estimated
            elapsed = current_tick - scheduled_task.start_time
            expected = scheduled_task.estimated_end_time - scheduled_task.start_time

            if expected > 0 and (elapsed / expected) > self.config.latency_variance_threshold:
                # Find GPU fallback (state-vector simulation)
                fallback = self.cluster.get_least_loaded(DeviceType.GPU)
                if fallback and fallback.can_accept(node):
                    decisions.append(ReallocationDecision(
                        task_id=task_id,
                        from_device_id=scheduled_task.device_id,
                        to_device=fallback,
                        reason=f"QPU latency {elapsed/expected:.1f}x expected, fallback to GPU",
                    ))

        return decisions

    def _check_load_balance(
        self, running_tasks: dict, current_tick: float
    ) -> list[ReallocationDecision]:
        """Detect device type overload and rebalance to underutilized devices."""
        decisions = []

        # Calculate average utilization per device type
        type_util: dict[DeviceType, float] = {}
        for dt in DeviceType:
            devices = self.cluster.get_devices_by_type(dt)
            if devices:
                type_util[dt] = sum(d.utilization for d in devices) / len(devices)
            else:
                type_util[dt] = 0.0

        # Find overloaded device types
        overloaded = {dt for dt, u in type_util.items() if u > self.config.utilization_threshold}
        underloaded = {dt for dt, u in type_util.items() if u < 0.5 and dt not in overloaded}

        if not overloaded or not underloaded:
            return decisions

        # Find tasks on overloaded devices that have reasonable affinity to underloaded devices
        for task_id, scheduled_task in list(running_tasks.items()):
            device = self._find_device(scheduled_task.device_id)
            if device is None or device.device_type not in overloaded:
                continue

            node = self.tcg.get_node(task_id)

            if node.reallocation_count >= self.config.max_reallocations_per_task:
                continue
            if (current_tick - node.last_reallocation_time) < self.config.reallocation_cooldown:
                continue

            # Find best underloaded target
            for target_dt in underloaded:
                affinity = node.hardware_affinity.get(target_dt, 0.0)
                if affinity >= 0.3:  # Minimum viable affinity
                    target = self.cluster.get_least_loaded(target_dt)
                    if target and target.can_accept(node):
                        decisions.append(ReallocationDecision(
                            task_id=task_id,
                            from_device_id=scheduled_task.device_id,
                            to_device=target,
                            reason=f"Load balance: {device.device_type.name} overloaded, "
                                   f"moving to {target_dt.name}",
                        ))
                        break  # Only one reallocation per task

        return decisions

    def _find_device(self, device_id: str) -> Optional[Device]:
        for d in self.cluster.devices:
            if d.id == device_id:
                return d
        return None
