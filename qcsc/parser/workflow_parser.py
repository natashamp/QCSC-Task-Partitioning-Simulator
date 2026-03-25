"""Parse workflow YAML definitions into a TensorComputeGraph."""

from pathlib import Path
import yaml

from qcsc.config import TaskType
from qcsc.graph.node import TaskNode
from qcsc.graph.edge import DataEdge
from qcsc.graph.tcg import TensorComputeGraph


TASK_TYPE_MAP = {
    "CLASSICAL_DATA_OP": TaskType.CLASSICAL_DATA_OP,
    "LINEAR_ALGEBRA": TaskType.LINEAR_ALGEBRA,
    "QUANTUM_CIRCUIT": TaskType.QUANTUM_CIRCUIT,
    "PARAMETER_OPT": TaskType.PARAMETER_OPT,
    "NN_INFERENCE": TaskType.NN_INFERENCE,
    "COMBINATORIAL_OPT": TaskType.COMBINATORIAL_OPT,
    "SAMPLING": TaskType.SAMPLING,
}


class WorkflowParser:
    """Parses a YAML workflow definition into a TensorComputeGraph."""

    @staticmethod
    def parse(yaml_path: str | Path) -> TensorComputeGraph:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        workflow = data["workflow"]
        tcg = TensorComputeGraph(name=workflow.get("name", "unnamed"))

        for task_def in workflow["tasks"]:
            task_type = TASK_TYPE_MAP[task_def["type"]]
            node = TaskNode(
                id=task_def["id"],
                name=task_def.get("name", task_def["id"]),
                task_type=task_type,
                compute_cost=float(task_def.get("cost", 1.0)),
                memory_requirement=float(task_def.get("memory", 10.0)),
                qubit_count=task_def.get("qubits"),
                circuit_depth=task_def.get("depth"),
                tensor_shape=tuple(task_def["tensor_shape"]) if "tensor_shape" in task_def else None,
            )
            tcg.add_node(node)

        # Second pass: create edges from depends_on
        for task_def in workflow["tasks"]:
            if "depends_on" in task_def:
                for dep_id in task_def["depends_on"]:
                    edge = DataEdge(
                        source_id=dep_id,
                        target_id=task_def["id"],
                        data_size=float(task_def.get("input_data_size", 1.0)),
                    )
                    tcg.add_edge(edge)

        return tcg
