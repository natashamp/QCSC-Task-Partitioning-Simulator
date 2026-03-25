"""Discrete-event simulation engine for QCSC workload scheduling."""

import heapq
import numpy as np
from typing import Optional

from qcsc.config import DeviceType, SchedulerConfig, TaskStatus
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.hardware.cluster import HardwareCluster
from qcsc.hardware.device import QPU
from qcsc.scheduler.scheduler import DynamicScheduler
from qcsc.scheduler.feedback import AdaptiveFeedbackLoop
from qcsc.scheduler.metrics import MetricsCollector
from .clock import SimulatedClock
from .events import (
    Event, EventType, TaskDispatchEvent, TaskCompleteEvent, ReallocationEvent,
)


class SimulationEngine:
    """Central discrete-event simulation loop."""

    def __init__(
        self,
        tcg: TensorComputeGraph,
        cluster: HardwareCluster,
        config: Optional[SchedulerConfig] = None,
        enable_feedback: bool = True,
        verbose: bool = True,
    ):
        self.tcg = tcg
        self.cluster = cluster
        self.config = config or SchedulerConfig()
        self.enable_feedback = enable_feedback
        self.verbose = verbose

        self.clock = SimulatedClock()
        self.metrics = MetricsCollector()
        self.scheduler = DynamicScheduler(tcg, cluster, self.config, self.metrics)
        self.feedback = AdaptiveFeedbackLoop(tcg, cluster, self.config, self.metrics)

        self.event_queue: list[Event] = []
        self.rng = np.random.default_rng(self.config.random_seed)
        self.max_ticks = 10000  # Safety limit

    def run(self) -> MetricsCollector:
        """Run the simulation to completion."""
        if self.verbose:
            print(f"\nStarting simulation: {self.tcg.name}")
            print(f"Tasks: {len(self.tcg)}, Feedback: {self.enable_feedback}")
            print(self.cluster.summary())
            print()

        # Initial scheduling
        self._schedule_and_dispatch()

        iteration = 0
        while not self.tcg.all_completed() and iteration < self.max_ticks:
            iteration += 1

            # Find next event time
            next_time = self.scheduler.get_next_event_time()
            if next_time is None:
                # No running tasks and not all completed - deadlock
                if self.verbose:
                    pending = [n.id for n in self.tcg.nodes.values()
                               if n.status == TaskStatus.PENDING]
                    print(f"WARNING: No running tasks. Pending: {pending}")
                break

            self.clock.advance_to(next_time)

            # Record device snapshots
            self.metrics.record_device_snapshot(self.cluster.devices, self.clock.current_tick)

            # Complete finished tasks
            self._process_completions()

            # QPU calibration cycles
            self._handle_qpu_calibration()

            # Adaptive feedback check
            if self.enable_feedback and self.feedback.should_check(self.clock.current_tick):
                self._process_feedback()

            # Schedule newly ready tasks
            self._schedule_and_dispatch()

        if self.verbose:
            self.metrics.print_summary()

        return self.metrics

    def _schedule_and_dispatch(self) -> None:
        """Schedule all currently ready tasks."""
        scheduled = self.scheduler.schedule_ready_tasks(self.clock.current_tick)

        for st in scheduled:
            # Apply stochastic latency for QPU tasks
            device = self._find_device(st.device_id)
            if device and device.device_type == DeviceType.QPU:
                variance = self.config.qpu_latency_variance
                multiplier = max(0.5, self.rng.normal(1.0, variance))
                original_duration = st.estimated_end_time - st.start_time
                actual_duration = original_duration * multiplier
                st.actual_end_time = st.start_time + actual_duration

                if self.verbose and abs(multiplier - 1.0) > 0.1:
                    print(f"  [t={self.clock.current_tick:6.1f}] QPU latency variance: "
                          f"task {st.task_id} {multiplier:.2f}x "
                          f"(est={original_duration:.1f}, actual={actual_duration:.1f})")
            else:
                st.actual_end_time = st.estimated_end_time

            if self.verbose:
                node = self.tcg.get_node(st.task_id)
                print(f"  [t={self.clock.current_tick:6.1f}] DISPATCH {st.task_id:20s} "
                      f"-> {st.device_id:12s} "
                      f"(type={node.task_type.name}, "
                      f"est_end={st.actual_end_time:.1f})")

    def _process_completions(self) -> None:
        """Complete all tasks finishing at current tick."""
        completing = []
        for task_id, st in list(self.scheduler.running_tasks.items()):
            end = st.actual_end_time if st.actual_end_time > 0 else st.estimated_end_time
            if end <= self.clock.current_tick:
                completing.append(task_id)

        for task_id in completing:
            new_ready = self.scheduler.complete_task(task_id, self.clock.current_tick)
            if self.verbose:
                print(f"  [t={self.clock.current_tick:6.1f}] COMPLETE {task_id}")

    def _process_feedback(self) -> None:
        """Run the adaptive feedback loop and process reallocations."""
        decisions = self.feedback.check_and_reallocate(
            self.scheduler.running_tasks, self.clock.current_tick
        )

        for decision in decisions:
            result = self.scheduler.reallocate_task(
                decision.task_id, decision.to_device, self.clock.current_tick
            )
            if result and self.verbose:
                print(f"  [t={self.clock.current_tick:6.1f}] REALLOCATE {decision.task_id} "
                      f"from {decision.from_device_id} -> {decision.to_device.id} "
                      f"({decision.reason})")

            # Apply stochastic latency if reallocated to QPU
            if result and decision.to_device.device_type == DeviceType.QPU:
                variance = self.config.qpu_latency_variance
                multiplier = max(0.5, self.rng.normal(1.0, variance))
                original_duration = result.estimated_end_time - result.start_time
                result.actual_end_time = result.start_time + original_duration * multiplier

    def _handle_qpu_calibration(self) -> None:
        """Check if any QPU needs calibration."""
        for device in self.cluster.get_devices_by_type(DeviceType.QPU):
            if isinstance(device, QPU):
                device.tick()
                if device.needs_calibration(self.clock.current_tick):
                    if self.verbose:
                        print(f"  [t={self.clock.current_tick:6.1f}] QPU CALIBRATION: {device.id}")
                    device.calibrate()

    def _find_device(self, device_id: str):
        for d in self.cluster.devices:
            if d.id == device_id:
                return d
        return None
