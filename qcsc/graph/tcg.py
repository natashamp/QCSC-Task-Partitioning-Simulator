"""Tensor Compute Graph - DAG representation of the workflow."""

from typing import Optional
import networkx as nx

from qcsc.config import TaskStatus
from .node import TaskNode
from .edge import DataEdge


class TensorComputeGraph:
    """Directed acyclic graph representing a quantum-classical workflow."""

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._graph = nx.DiGraph()
        self._nodes: dict[str, TaskNode] = {}
        self._edges: list[DataEdge] = []

    def add_node(self, node: TaskNode) -> None:
        self._nodes[node.id] = node
        self._graph.add_node(node.id, weight=node.compute_cost)

    def add_edge(self, edge: DataEdge) -> None:
        self._edges.append(edge)
        self._graph.add_edge(
            edge.source_id, edge.target_id, weight=edge.data_size
        )

    def get_node(self, node_id: str) -> TaskNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> dict[str, TaskNode]:
        return self._nodes

    @property
    def edges(self) -> list[DataEdge]:
        return self._edges

    def topological_sort(self) -> list[str]:
        return list(nx.topological_sort(self._graph))

    def get_predecessors(self, node_id: str) -> list[str]:
        return list(self._graph.predecessors(node_id))

    def get_successors(self, node_id: str) -> list[str]:
        return list(self._graph.successors(node_id))

    def get_ready_tasks(self) -> list[TaskNode]:
        """Return tasks whose all predecessors are COMPLETED and that are PENDING."""
        ready = []
        for node_id, node in self._nodes.items():
            if node.status != TaskStatus.PENDING:
                continue
            preds = self.get_predecessors(node_id)
            if all(self._nodes[p].status == TaskStatus.COMPLETED for p in preds):
                ready.append(node)
        return ready

    def critical_path_length(self) -> float:
        """Compute the critical path length (longest weighted path)."""
        return nx.dag_longest_path_length(
            self._graph,
            weight="weight",
            default_weight=1.0,
        )

    def critical_path(self) -> list[str]:
        """Return node IDs on the critical path."""
        return nx.dag_longest_path(
            self._graph,
            weight="weight",
            default_weight=1.0,
        )

    def node_urgency(self, node_id: str) -> float:
        """Compute urgency as the longest path from this node to any sink."""
        longest = 0.0
        for sink in (n for n in self._graph.nodes() if self._graph.out_degree(n) == 0):
            try:
                for path in nx.all_simple_paths(self._graph, node_id, sink):
                    cost = sum(self._nodes[n].compute_cost for n in path)
                    longest = max(longest, cost)
            except nx.NetworkXError:
                continue
        return longest

    def parallelism_width(self) -> int:
        """Max number of tasks that can run concurrently (max antichain)."""
        try:
            antichain = nx.dag.dag_longest_path(self._graph)
            levels: dict[str, int] = {}
            for node_id in self.topological_sort():
                preds = self.get_predecessors(node_id)
                if not preds:
                    levels[node_id] = 0
                else:
                    levels[node_id] = max(levels[p] for p in preds) + 1
            from collections import Counter
            level_counts = Counter(levels.values())
            return max(level_counts.values()) if level_counts else 1
        except Exception:
            return 1

    def all_completed(self) -> bool:
        return all(n.status == TaskStatus.COMPLETED for n in self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)
