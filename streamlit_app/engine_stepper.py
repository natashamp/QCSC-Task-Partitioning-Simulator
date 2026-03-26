"""Stepping wrapper around the simulation engine for event-by-event replay."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from qcsc.config import DeviceType, SchedulerConfig, TaskStatus
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.hardware.cluster import HardwareCluster
from qcsc.hardware.device import QPU
from qcsc.hardware.mapper import HardwareMapper
from qcsc.scheduler.scheduler import DynamicScheduler
from qcsc.scheduler.feedback import AdaptiveFeedbackLoop
from qcsc.scheduler.metrics import MetricsCollector
from qcsc.simulation.clock import SimulatedClock


@dataclass
class SubEvent:
    event_type: str  # DISPATCH, COMPLETE, REALLOCATE, CALIBRATION, FEEDBACK_CHECK
    tick: float
    task_id: Optional[str] = None
    device_id: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class StepResult:
    step_number: int
    tick: float
    sub_events: list[SubEvent] = field(default_factory=list)
    is_finished: bool = False
    node_statuses: dict = field(default_factory=dict)
    device_states: list = field(default_factory=list)
    running_tasks: dict = field(default_factory=dict)
    newly_completed: list = field(default_factory=list)
    newly_dispatched: list = field(default_factory=list)
    score_breakdowns: dict = field(default_factory=dict)


class SteppableSimulation:
    """Wraps the simulation to expose one-step-at-a-time execution."""

    def __init__(
        self,
        tcg: TensorComputeGraph,
        cluster: HardwareCluster,
        config: SchedulerConfig,
        enable_feedback: bool = True,
    ):
        self.tcg = tcg
        self.cluster = cluster
        self.config = config
        self.enable_feedback = enable_feedback

        self.clock = SimulatedClock()
        self.metrics = MetricsCollector()
        self.scheduler = DynamicScheduler(tcg, cluster, config, self.metrics)
        self.feedback = AdaptiveFeedbackLoop(tcg, cluster, config, self.metrics)
        self.mapper = HardwareMapper(cluster, config)

        self.rng = np.random.default_rng(config.random_seed)
        self._step_count = 0
        self._initialized = False

    def initialize(self) -> StepResult:
        """Perform initial dispatch and return the first step result."""
        self._initialized = True
        sub_events = []
        dispatched = []
        score_breakdowns = {}

        scheduled = self.scheduler.schedule_ready_tasks(self.clock.current_tick)
        for st in scheduled:
            device = self._find_device(st.device_id)
            node = self.tcg.get_node(st.task_id)

            # Capture score breakdowns for all available devices
            task_scores = {}
            for d in self.cluster.devices:
                if d.can_accept(node):
                    task_scores[d.id] = self.mapper.score_breakdown(node, d)
            score_breakdowns[st.task_id] = task_scores

            # Apply QPU stochastic latency
            if device and device.device_type == DeviceType.QPU:
                multiplier = max(0.5, self.rng.normal(1.0, self.config.qpu_latency_variance))
                original = st.estimated_end_time - st.start_time
                st.actual_end_time = st.start_time + original * multiplier
            else:
                st.actual_end_time = st.estimated_end_time
                multiplier = 1.0

            sub_events.append(SubEvent(
                event_type="DISPATCH",
                tick=0.0,
                task_id=st.task_id,
                device_id=st.device_id,
                details={
                    "task_type": node.task_type.name,
                    "est_end": round(st.estimated_end_time, 2),
                    "actual_end": round(st.actual_end_time, 2),
                    "latency_multiplier": round(multiplier, 2),
                    "compute_cost": node.compute_cost,
                    "urgency": round(self.tcg.node_urgency(node.id), 1),
                },
            ))
            dispatched.append((st.task_id, st.device_id))

        result = self._build_result(
            step_number=0,
            sub_events=sub_events,
            newly_dispatched=dispatched,
            score_breakdowns=score_breakdowns,
        )
        return result

    def step(self) -> StepResult:
        """Execute one simulation step (advance to next event, process it)."""
        if not self._initialized:
            return self.initialize()

        self._step_count += 1
        sub_events = []
        completed = []
        dispatched = []
        score_breakdowns = {}

        # Find next event time
        next_time = self.scheduler.get_next_event_time()
        if next_time is None or self.tcg.all_completed():
            return self._build_result(
                step_number=self._step_count,
                sub_events=[],
                is_finished=True,
            )

        self.clock.advance_to(next_time)

        # Record device snapshots
        self.metrics.record_device_snapshot(self.cluster.devices, self.clock.current_tick)

        # Complete finished tasks
        completing_ids = []
        for task_id, st in list(self.scheduler.running_tasks.items()):
            end = st.actual_end_time if st.actual_end_time > 0 else st.estimated_end_time
            if end <= self.clock.current_tick:
                completing_ids.append(task_id)

        for task_id in completing_ids:
            node = self.tcg.get_node(task_id)
            st = self.scheduler.running_tasks.get(task_id)
            duration = self.clock.current_tick - st.start_time if st else 0
            estimated = st.estimated_end_time - st.start_time if st else 0

            new_ready = self.scheduler.complete_task(task_id, self.clock.current_tick)
            completed.append(task_id)

            sub_events.append(SubEvent(
                event_type="COMPLETE",
                tick=self.clock.current_tick,
                task_id=task_id,
                device_id=st.device_id if st else None,
                details={
                    "task_type": node.task_type.name,
                    "duration": round(duration, 2),
                    "estimated_duration": round(estimated, 2),
                    "new_ready": new_ready,
                    "successors": self.tcg.get_successors(task_id),
                },
            ))

        # QPU calibration
        for device in self.cluster.get_devices_by_type(DeviceType.QPU):
            if isinstance(device, QPU):
                device.tick()
                if device.needs_calibration(self.clock.current_tick):
                    device.calibrate()
                    sub_events.append(SubEvent(
                        event_type="CALIBRATION",
                        tick=self.clock.current_tick,
                        device_id=device.id,
                        details={"message": "QPU calibration cycle completed"},
                    ))

        # Adaptive feedback
        if self.enable_feedback and self.feedback.should_check(self.clock.current_tick):
            decisions = self.feedback.check_and_reallocate(
                self.scheduler.running_tasks, self.clock.current_tick
            )
            for decision in decisions:
                result = self.scheduler.reallocate_task(
                    decision.task_id, decision.to_device, self.clock.current_tick
                )
                if result:
                    if decision.to_device.device_type == DeviceType.QPU:
                        multiplier = max(0.5, self.rng.normal(1.0, self.config.qpu_latency_variance))
                        original = result.estimated_end_time - result.start_time
                        result.actual_end_time = result.start_time + original * multiplier
                    else:
                        result.actual_end_time = result.estimated_end_time

                    sub_events.append(SubEvent(
                        event_type="REALLOCATE",
                        tick=self.clock.current_tick,
                        task_id=decision.task_id,
                        device_id=decision.to_device.id,
                        details={
                            "from_device": decision.from_device_id,
                            "to_device": decision.to_device.id,
                            "reason": decision.reason,
                        },
                    ))

        # Schedule newly ready tasks
        scheduled = self.scheduler.schedule_ready_tasks(self.clock.current_tick)
        for st in scheduled:
            device = self._find_device(st.device_id)
            node = self.tcg.get_node(st.task_id)

            task_scores = {}
            for d in self.cluster.devices:
                if d.can_accept(node) or d.id == st.device_id:
                    task_scores[d.id] = self.mapper.score_breakdown(node, d)
            score_breakdowns[st.task_id] = task_scores

            if device and device.device_type == DeviceType.QPU:
                multiplier = max(0.5, self.rng.normal(1.0, self.config.qpu_latency_variance))
                original = st.estimated_end_time - st.start_time
                st.actual_end_time = st.start_time + original * multiplier
            else:
                st.actual_end_time = st.estimated_end_time
                multiplier = 1.0

            sub_events.append(SubEvent(
                event_type="DISPATCH",
                tick=self.clock.current_tick,
                task_id=st.task_id,
                device_id=st.device_id,
                details={
                    "task_type": node.task_type.name,
                    "est_end": round(st.estimated_end_time, 2),
                    "actual_end": round(st.actual_end_time, 2),
                    "latency_multiplier": round(multiplier, 2),
                    "compute_cost": node.compute_cost,
                    "urgency": round(self.tcg.node_urgency(node.id), 1),
                },
            ))
            dispatched.append((st.task_id, st.device_id))

        is_finished = self.tcg.all_completed()

        return self._build_result(
            step_number=self._step_count,
            sub_events=sub_events,
            newly_completed=completed,
            newly_dispatched=dispatched,
            score_breakdowns=score_breakdowns,
            is_finished=is_finished,
        )

    def _build_result(
        self,
        step_number: int,
        sub_events: list,
        newly_completed: list = None,
        newly_dispatched: list = None,
        score_breakdowns: dict = None,
        is_finished: bool = False,
    ) -> StepResult:
        node_statuses = {
            nid: node.status.name for nid, node in self.tcg.nodes.items()
        }
        device_states = []
        for d in self.cluster.devices:
            device_states.append({
                "id": d.id,
                "name": d.name,
                "type": d.device_type.name,
                "load": d.current_load,
                "capacity": d.capacity,
                "utilization": round(d.utilization, 3),
                "memory_used": round(d.memory_used, 1),
                "memory_total": round(d.memory_total, 1),
            })

        running = {}
        for tid, st in self.scheduler.running_tasks.items():
            running[tid] = {
                "device_id": st.device_id,
                "start": st.start_time,
                "est_end": st.estimated_end_time,
                "actual_end": st.actual_end_time,
            }

        return StepResult(
            step_number=step_number,
            tick=self.clock.current_tick,
            sub_events=sub_events,
            is_finished=is_finished,
            node_statuses=node_statuses,
            device_states=device_states,
            running_tasks=running,
            newly_completed=newly_completed or [],
            newly_dispatched=newly_dispatched or [],
            score_breakdowns=score_breakdowns or {},
        )

    def _find_device(self, device_id: str):
        for d in self.cluster.devices:
            if d.id == device_id:
                return d
        return None
