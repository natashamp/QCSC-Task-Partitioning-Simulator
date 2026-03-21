# QCSC Task Partitioning and Dynamic Scheduling Simulator

A discrete-event simulation system that models the execution of hybrid quantum-classical workflows on heterogeneous hardware clusters. It demonstrates how complex workloads mixing classical computing (CPU, GPU, AI accelerators) with quantum processing units (QPUs) can be scheduled, executed, and dynamically rebalanced to handle real-world quantum hardware variability.

## Project Overview

Quantum-Centric Supercomputing (QCSC) architectures combine classical and quantum processors to solve problems neither can tackle alone. This simulator lets you define hybrid workflows as directed acyclic graphs (DAGs), map tasks onto a modeled hardware cluster, and observe how different scheduling and feedback strategies affect execution performance.

The simulator includes:

- **YAML-defined workflows** representing real quantum algorithms (VQE, QAOA, hybrid ML)
- **Heterogeneous hardware modeling** with CPUs, GPUs, AI accelerators, and QPUs — each with realistic capacity, speed, and memory constraints
- **Affinity-based task mapping** that scores each task-device pairing on suitability, load, transfer cost, and memory fit
- **Stochastic QPU latency** modeling quantum hardware noise via configurable variance
- **An adaptive feedback loop** that detects bottlenecks at runtime and reallocates tasks (e.g., moving a stalled QPU task to a GPU for state-vector simulation)
- **Visualization outputs** including Gantt charts, DAG graphs, and a multi-panel performance dashboard

## Architecture

```
workflows/*.yaml
        |
        v
  WorkflowParser ──> TensorComputeGraph (DAG of TaskNodes + DataEdges)
        |
        v
   CostAnalyzer ──> Cost & parallelism reports
        |
        v
  HardwareCluster ──> Heterogeneous devices (CPU, GPU, AI Accel, QPU)
        |
        v
  SimulationEngine (discrete-event loop)
    ├── DynamicScheduler + HardwareMapper (priority-based dispatch)
    ├── AdaptiveFeedbackLoop (reallocation decisions)
    ├── SimulatedClock (tick-based time)
    └── MetricsCollector (runtime statistics)
        |
        v
  Visualization ──> Gantt chart, DAG graph, performance dashboard (PNGs)
```

## Module Breakdown

| Module | Purpose |
|---|---|
| `qcsc/graph/` | DAG representation — `TaskNode` (tasks with type, cost, affinity), `DataEdge` (data dependencies), `TensorComputeGraph` (critical path, urgency, parallelism analysis) |
| `qcsc/hardware/` | Hardware modeling — `Device` subclasses (CPU, GPU, AIAccelerator, QPU with coherence/fidelity), `HardwareCluster` (device management), `HardwareMapper` (weighted scoring for task placement) |
| `qcsc/parser/` | `WorkflowParser` (YAML to DAG), `CostAnalyzer` (compute cost breakdown, critical path, parallelism ratio) |
| `qcsc/scheduler/` | `DynamicScheduler` (task lifecycle and dispatch), `AdaptiveFeedbackLoop` (QPU latency fallback + load balancing), `MetricsCollector` (per-task and per-device statistics) |
| `qcsc/simulation/` | `SimulationEngine` (main event loop), `SimulatedClock`, event types (dispatch, complete, reallocation, calibration) |
| `qcsc/visualization/` | `GanttChart` (execution timeline), `GraphVisualizer` (DAG layout), `Dashboard` (utilization, latency, reallocation summary) |

## Included Workflows

| Workflow | File | Description |
|---|---|---|
| **VQE** | `workflows/vqe_workflow.yaml` | Variational Quantum Eigensolver — iterative quantum-classical optimization loop with two QPU execution rounds |
| **QAOA** | `workflows/qaoa_workflow.yaml` | Quantum Approximate Optimization Algorithm for Max-Cut — includes neural network parameter prediction and two QAOA depth rounds |
| **Hybrid ML** | `workflows/hybrid_ml_workflow.yaml` | Quantum-classical machine learning pipeline with three parallel branches (classical NN, quantum kernel, variational classifier) merging into a training phase |

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
pip install -r requirements.txt
```

### Running a Simulation

```bash
# Basic run
python main.py --workflow workflows/vqe_workflow.yaml

# With adaptive feedback and QPU latency variance
python main.py --workflow workflows/vqe_workflow.yaml --feedback --latency-variance 0.3

# Verbose output with custom output directory
python main.py --workflow workflows/qaoa_workflow.yaml --feedback --latency-variance 0.2 --output results/qaoa --verbose

# Reproducible run with a fixed seed
python main.py --workflow workflows/hybrid_ml_workflow.yaml --feedback --seed 42
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--workflow` | Path to workflow YAML file | *required* |
| `--feedback` | Enable adaptive feedback loop | disabled |
| `--latency-variance` | QPU latency variance (0.0 = deterministic) | `0.0` |
| `--output` | Output directory for visualizations | `results/` |
| `--seed` | Random seed for reproducibility | `None` |
| `--verbose` | Print detailed per-tick simulation logs | disabled |

### Output

The simulator generates three PNG files in the output directory:

- **`gantt_chart.png`** — Horizontal timeline showing when each task ran on which device. Reallocated tasks have red borders with arrows showing the migration.
- **`task_graph.png`** — The workflow DAG with nodes colored by assigned device type and edge widths proportional to data transfer size.
- **`dashboard.png`** — Four-panel summary: device utilization over time, task latency distribution, reallocation timeline, and a statistics table.

## End-to-End Workflow

1. **Parse** — `WorkflowParser` reads the YAML and builds a `TensorComputeGraph` (DAG of tasks and data edges)
2. **Analyze** — `CostAnalyzer` computes total cost, per-type breakdowns, critical path length, and parallelism width
3. **Build hardware** — `HardwareCluster.default_cluster()` creates 6 devices: 2 CPUs, 2 GPUs, 1 AI accelerator, 1 QPU
4. **Simulate** — The `SimulationEngine` runs a discrete-event loop:
   - Ready tasks (all predecessors complete) are dispatched to devices via `HardwareMapper`, prioritized by DAG urgency
   - QPU tasks receive stochastic latency multipliers sampled from a normal distribution
   - The clock advances to the next task completion; resources are released and successors become ready
   - If feedback is enabled, every N ticks the `AdaptiveFeedbackLoop` checks for QPU stalls (>1.5x estimated time triggers GPU fallback) and device overload (>85% utilization triggers rebalancing)
   - QPU calibration pauses occur periodically (every 50 ticks)
5. **Report** — `MetricsCollector` prints makespan, average latency ratio, per-device stats, and reallocation count
6. **Visualize** — Three PNG outputs are saved to the output directory

## What You Will Learn

This project is designed to teach core concepts at the intersection of systems architecture, scheduling theory, and quantum computing:

### Task Scheduling and DAG Analysis
- How to represent computational workflows as **directed acyclic graphs** and use topological ordering to respect dependencies
- **Critical path analysis** — identifying the longest chain of dependent tasks that determines the minimum possible execution time
- **Task urgency** and priority-based scheduling — why the order you dispatch tasks matters for overall makespan

### Heterogeneous Hardware Management
- How different processor types (CPU, GPU, QPU) have fundamentally different **performance characteristics** — capacity, throughput, memory, and reliability
- **Affinity-based mapping** — the idea that not every task runs equally well on every device, and how to formalize device suitability as a scoring function
- **Resource contention** — what happens when multiple tasks compete for limited device capacity, and how load-aware scheduling mitigates bottlenecks

### Quantum Computing Challenges
- Why **quantum hardware is inherently noisy and variable** — latency is stochastic, qubits decohere, and periodic calibration is required
- The concept of **quantum-classical hybrid algorithms** (VQE, QAOA) where quantum and classical processors must collaborate iteratively
- **Fallback strategies** — when a QPU task takes too long, it can sometimes be simulated classically on a GPU instead

### Adaptive and Dynamic Systems
- **Feedback loops** in scheduling — monitoring runtime performance and making reallocation decisions on the fly
- The tradeoffs of **dynamic rescheduling**: reallocation has a cost (lost progress, transfer overhead), so it must be triggered judiciously (cooldowns, thresholds, max reallocation limits)
- How **discrete-event simulation** works as an efficient technique for modeling complex systems without simulating every time unit

### Systems Thinking
- How to decompose a complex simulation into clean, modular components (graph, hardware, scheduler, simulation, visualization)
- The value of **configurable parameters** (affinity weights, feedback thresholds, variance levels) for exploring design tradeoffs
- Reading **Gantt charts and utilization dashboards** to diagnose performance bottlenecks in parallel systems
