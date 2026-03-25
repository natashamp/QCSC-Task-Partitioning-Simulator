"""Global configuration and defaults for the QCSC simulator."""

from enum import Enum, auto
from dataclasses import dataclass, field


class TaskType(Enum):
    CLASSICAL_DATA_OP = auto()
    LINEAR_ALGEBRA = auto()
    QUANTUM_CIRCUIT = auto()
    PARAMETER_OPT = auto()
    NN_INFERENCE = auto()
    COMBINATORIAL_OPT = auto()
    SAMPLING = auto()


class DeviceType(Enum):
    CPU = auto()
    GPU = auto()
    AI_ACCELERATOR = auto()
    QPU = auto()


class TaskStatus(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    REALLOCATED = auto()


# Default hardware affinity scores: TaskType -> {DeviceType: score}
DEFAULT_AFFINITY = {
    TaskType.CLASSICAL_DATA_OP: {
        DeviceType.CPU: 0.9, DeviceType.GPU: 0.2,
        DeviceType.AI_ACCELERATOR: 0.1, DeviceType.QPU: 0.0,
    },
    TaskType.LINEAR_ALGEBRA: {
        DeviceType.CPU: 0.3, DeviceType.GPU: 0.95,
        DeviceType.AI_ACCELERATOR: 0.4, DeviceType.QPU: 0.0,
    },
    TaskType.QUANTUM_CIRCUIT: {
        DeviceType.CPU: 0.1, DeviceType.GPU: 0.6,
        DeviceType.AI_ACCELERATOR: 0.0, DeviceType.QPU: 0.95,
    },
    TaskType.PARAMETER_OPT: {
        DeviceType.CPU: 0.3, DeviceType.GPU: 0.4,
        DeviceType.AI_ACCELERATOR: 0.9, DeviceType.QPU: 0.0,
    },
    TaskType.NN_INFERENCE: {
        DeviceType.CPU: 0.2, DeviceType.GPU: 0.7,
        DeviceType.AI_ACCELERATOR: 0.95, DeviceType.QPU: 0.0,
    },
    TaskType.COMBINATORIAL_OPT: {
        DeviceType.CPU: 0.2, DeviceType.GPU: 0.3,
        DeviceType.AI_ACCELERATOR: 0.1, DeviceType.QPU: 0.95,
    },
    TaskType.SAMPLING: {
        DeviceType.CPU: 0.1, DeviceType.GPU: 0.5,
        DeviceType.AI_ACCELERATOR: 0.0, DeviceType.QPU: 0.9,
    },
}


@dataclass
class SchedulerConfig:
    """Configuration for the dynamic scheduler and mapper."""
    # Mapper weights
    affinity_weight: float = 0.4
    load_weight: float = 0.25
    transfer_weight: float = 0.2
    memory_weight: float = 0.15

    # Feedback loop
    feedback_interval: int = 5  # Check every N ticks
    latency_variance_threshold: float = 1.5  # Trigger reallocation if actual/expected > this
    utilization_threshold: float = 0.85  # High-load threshold
    reallocation_cooldown: int = 10  # Ticks before a task can be reallocated again
    max_reallocations_per_task: int = 2

    # Simulation
    qpu_latency_variance: float = 0.3  # Stddev of latency multiplier
    random_seed: int = 42

    # Cross-device transfer latency (source_type, target_type) -> latency multiplier
    transfer_latency: dict = field(default_factory=lambda: {
        (DeviceType.CPU, DeviceType.GPU): 0.5,
        (DeviceType.GPU, DeviceType.CPU): 0.5,
        (DeviceType.CPU, DeviceType.QPU): 2.0,
        (DeviceType.QPU, DeviceType.CPU): 2.0,
        (DeviceType.GPU, DeviceType.QPU): 2.5,
        (DeviceType.QPU, DeviceType.GPU): 2.5,
        (DeviceType.CPU, DeviceType.AI_ACCELERATOR): 0.3,
        (DeviceType.AI_ACCELERATOR, DeviceType.CPU): 0.3,
        (DeviceType.GPU, DeviceType.AI_ACCELERATOR): 0.4,
        (DeviceType.AI_ACCELERATOR, DeviceType.GPU): 0.4,
        (DeviceType.AI_ACCELERATOR, DeviceType.QPU): 2.5,
        (DeviceType.QPU, DeviceType.AI_ACCELERATOR): 2.5,
    })
