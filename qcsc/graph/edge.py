"""Data edge representation in the Tensor Compute Graph."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataEdge:
    source_id: str
    target_id: str
    data_size: float = 1.0  # MB
    tensor_shape: Optional[tuple[int, ...]] = None
    transfer_latency: float = 0.0  # Computed dynamically based on device assignments
