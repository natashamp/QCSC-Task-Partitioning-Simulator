"""Analyze computational cost, parallelism, and hardware affinity of a workflow."""

from dataclasses import dataclass, field
from collections import Counter

from qcsc.config import TaskType, DeviceType, DEFAULT_AFFINITY
from qcsc.graph.tcg import TensorComputeGraph
from qcsc.graph.node import TaskNode


@dataclass
class CostReport:
    total_cost: float = 0.0
    per_type_cost: dict[TaskType, float] = field(default_factory=dict)
    per_device_affinity_cost: dict[DeviceType, float] = field(default_factory=dict)
    task_count: int = 0
    quantum_task_count: int = 0
    classical_task_count: int = 0


@dataclass
class ParallelismReport:
    critical_path_length: float = 0.0
    critical_path_nodes: list[str] = field(default_factory=list)
    max_parallelism_width: int = 1
    parallelism_ratio: float = 1.0  # total_cost / critical_path_length
    total_tasks: int = 0


class CostAnalyzer:
    """Analyzes cost, parallelism, and hardware affinity of a TensorComputeGraph."""

    @staticmethod
    def analyze_cost(tcg: TensorComputeGraph) -> CostReport:
        report = CostReport()
        report.task_count = len(tcg)

        per_type: dict[TaskType, float] = {}
        per_device: dict[DeviceType, float] = {dt: 0.0 for dt in DeviceType}

        for node in tcg.nodes.values():
            report.total_cost += node.compute_cost
            per_type[node.task_type] = per_type.get(node.task_type, 0.0) + node.compute_cost

            if node.is_quantum:
                report.quantum_task_count += 1
            else:
                report.classical_task_count += 1

            # Weight cost by affinity to estimate per-device load
            for dt, affinity in node.hardware_affinity.items():
                per_device[dt] += node.compute_cost * affinity

        report.per_type_cost = per_type
        report.per_device_affinity_cost = per_device
        return report

    @staticmethod
    def analyze_parallelism(tcg: TensorComputeGraph) -> ParallelismReport:
        report = ParallelismReport()
        report.total_tasks = len(tcg)
        report.critical_path_length = tcg.critical_path_length()
        report.critical_path_nodes = tcg.critical_path()
        report.max_parallelism_width = tcg.parallelism_width()

        if report.critical_path_length > 0:
            total_cost = sum(n.compute_cost for n in tcg.nodes.values())
            report.parallelism_ratio = total_cost / report.critical_path_length

        return report

    @staticmethod
    def print_report(cost: CostReport, parallelism: ParallelismReport) -> None:
        print("\n" + "=" * 60)
        print("WORKFLOW ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nTotal tasks: {cost.task_count}")
        print(f"  Classical: {cost.classical_task_count}")
        print(f"  Quantum:   {cost.quantum_task_count}")
        print(f"Total compute cost: {cost.total_cost:.1f}")

        print("\nCost by task type:")
        for tt, c in sorted(cost.per_type_cost.items(), key=lambda x: -x[1]):
            print(f"  {tt.name:20s}: {c:8.1f}")

        print("\nWeighted affinity cost by device:")
        for dt, c in sorted(cost.per_device_affinity_cost.items(), key=lambda x: -x[1]):
            print(f"  {dt.name:16s}: {c:8.1f}")

        print(f"\nCritical path length: {parallelism.critical_path_length:.1f}")
        print(f"Critical path nodes:  {' -> '.join(parallelism.critical_path_nodes)}")
        print(f"Max parallelism width: {parallelism.max_parallelism_width}")
        print(f"Parallelism ratio:    {parallelism.parallelism_ratio:.2f}")
        print("=" * 60)
