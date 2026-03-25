"""Simulated hardware device models."""

from dataclasses import dataclass, field
from typing import Optional

from qcsc.config import DeviceType
from qcsc.graph.node import TaskNode


@dataclass
class Device:
    id: str
    name: str
    device_type: DeviceType
    capacity: int = 4  # Max concurrent tasks
    speed_factor: float = 1.0  # Multiplier: execution_time = compute_cost / speed_factor
    memory_total: float = 1024.0  # MB
    memory_used: float = 0.0
    current_load: int = 0
    total_tasks_executed: int = 0

    @property
    def memory_available(self) -> float:
        return self.memory_total - self.memory_used

    @property
    def utilization(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0.0

    def can_accept(self, node: TaskNode) -> bool:
        return (
            self.current_load < self.capacity
            and self.memory_available >= node.memory_requirement
        )

    def estimated_time(self, node: TaskNode) -> float:
        return node.compute_cost / self.speed_factor

    def allocate(self, node: TaskNode) -> None:
        self.current_load += 1
        self.memory_used += node.memory_requirement

    def release(self, node: TaskNode) -> None:
        self.current_load = max(0, self.current_load - 1)
        self.memory_used = max(0.0, self.memory_used - node.memory_requirement)
        self.total_tasks_executed += 1


@dataclass
class CPU(Device):
    device_type: DeviceType = DeviceType.CPU
    capacity: int = 8
    speed_factor: float = 1.0
    cores: int = 16


@dataclass
class GPU(Device):
    device_type: DeviceType = DeviceType.GPU
    capacity: int = 4
    speed_factor: float = 5.0
    flops: float = 100.0  # TFLOPS
    memory_bandwidth: float = 900.0  # GB/s
    memory_total: float = 40960.0  # 40 GB


@dataclass
class AIAccelerator(Device):
    device_type: DeviceType = DeviceType.AI_ACCELERATOR
    capacity: int = 4
    speed_factor: float = 8.0
    ops_per_second: float = 275.0  # TOPS
    precision: str = "INT8"
    memory_total: float = 32768.0  # 32 GB


@dataclass
class QPU(Device):
    device_type: DeviceType = DeviceType.QPU
    capacity: int = 1  # Typically serial execution
    speed_factor: float = 3.0
    qubit_count: int = 127
    coherence_time_us: float = 100.0
    gate_fidelity: float = 0.999
    latency_variance: float = 0.3  # Stddev of latency multiplier
    memory_total: float = 0.0  # QPU doesn't use classical memory
    calibration_interval: int = 50  # Ticks between calibration cycles
    _ticks_since_calibration: int = 0

    def can_accept(self, node: TaskNode) -> bool:
        # QPU only checks load capacity, not classical memory
        return self.current_load < self.capacity

    def needs_calibration(self, current_tick: int) -> bool:
        return self._ticks_since_calibration >= self.calibration_interval

    def calibrate(self) -> float:
        """Perform calibration, returns downtime in ticks."""
        self._ticks_since_calibration = 0
        return 5.0  # Calibration takes 5 ticks

    def tick(self) -> None:
        self._ticks_since_calibration += 1
