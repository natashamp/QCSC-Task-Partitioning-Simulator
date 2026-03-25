"""Task node representation in the Tensor Compute Graph."""

from dataclasses import dataclass, field
from typing import Optional

from qcsc.config import TaskType, DeviceType, TaskStatus, DEFAULT_AFFINITY


@dataclass
class TaskNode:
    id: str
    name: str
    task_type: TaskType
    compute_cost: float
    memory_requirement: float = 10.0
    hardware_affinity: dict[DeviceType, float] = field(default_factory=dict)
    tensor_shape: Optional[tuple[int, ...]] = None
    qubit_count: Optional[int] = None
    circuit_depth: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_device: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    reallocation_count: int = 0
    last_reallocation_time: float = -100.0

    def __post_init__(self):
        if not self.hardware_affinity:
            self.hardware_affinity = dict(DEFAULT_AFFINITY[self.task_type])

    @property
    def is_quantum(self) -> bool:
        return self.task_type in (
            TaskType.QUANTUM_CIRCUIT,
            TaskType.COMBINATORIAL_OPT,
            TaskType.SAMPLING,
        )
