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

### Interactive Learning App (Streamlit)

An interactive web-based learning tool that walks you through the simulation one event at a time with plain-English explanations of every scheduling decision.

```bash
python3 -m streamlit run streamlit_app/app.py
```

This opens a browser at `http://localhost:8501` with:

- **Step-by-step controls** — press "Next Step" to advance one event at a time, or use "Auto-Play" in Guided mode to watch the simulation unfold automatically
- **Live DAG view** — the workflow graph updates in real time as tasks transition from pending (gray) to running (orange) to completed (green)
- **Device panel** — utilization bars and current task assignments for each device (CPU, GPU, AI Accelerator, QPU)
- **Incremental Gantt chart** — the execution timeline builds bar-by-bar as tasks dispatch and complete
- **Event narration** — each event gets a plain-English explanation of what happened and why the scheduler made that decision
- **Score breakdown** — when a task is dispatched, a table and bar chart show how every device scored on affinity, availability, transfer cost, and memory fit
- **Concept panels** — educational explanations appear automatically when new concepts become relevant (DAG dependencies, parallelism, affinity scoring, quantum noise, feedback loops, makespan)

#### Sidebar Controls

| Control | Description |
|---|---|
| **Workflow** | Select from VQE, QAOA, or Hybrid ML workflows |
| **Adaptive Feedback Loop** | Toggle the feedback system on/off |
| **QPU Latency Variance** | Slider from 0.0 (deterministic) to 1.0 (high noise) |
| **Random Seed** | Set seed for reproducible runs |
| **Mode** | Interactive (manual stepping) or Guided (auto-play with configurable speed) |
| **Reset Simulation** | Restart from the beginning with current settings |

## Simulation Phases

The simulator executes in seven distinct phases, each handled by a dedicated module:

### Phase 1: Workflow Parsing

**Module:** `qcsc/parser/workflow_parser.py`

The `WorkflowParser` reads a YAML workflow definition and converts it into an in-memory directed acyclic graph (DAG) called a `TensorComputeGraph`. This happens in two passes:

- **Pass 1 — Build nodes:** Iterates over `workflow.tasks` in the YAML and creates a `TaskNode` for each entry. Each node captures the task's `id`, `name`, `type` (mapped from a string like `"QUANTUM_CIRCUIT"` to the `TaskType` enum), `compute_cost`, `memory_requirement`, and optional quantum-specific fields (`qubit_count`, `circuit_depth`, `tensor_shape`). Each node also receives a hardware affinity score for every device type from the `DEFAULT_AFFINITY` matrix in `config.py` — for example, a `QUANTUM_CIRCUIT` task gets 0.95 affinity for QPU and 0.6 for GPU.

- **Pass 2 — Build edges:** Iterates again and processes `depends_on` lists. For each dependency, a `DataEdge` is created from the predecessor to the current task, carrying a `data_size` (in MB) that represents the volume of data flowing between them. These edges define the execution order constraints in the DAG.

The result is a `TensorComputeGraph` backed by NetworkX, where nodes are tasks and edges are data dependencies.

### Phase 2: Cost and Parallelism Analysis

**Module:** `qcsc/parser/cost_analyzer.py`

Before any simulation begins, the `CostAnalyzer` examines the DAG to answer two questions: **how expensive is this workflow, and how much of it can run in parallel?** This phase sets expectations — if the critical path is long and parallelism is low, no amount of clever scheduling will dramatically reduce makespan. If affinity cost is skewed toward one device type, that device will be the bottleneck.

#### Cost Report

`CostAnalyzer.analyze_cost()` makes a single pass over every node in the graph and computes:

- **Total compute cost** — the sum of all task costs across the entire workflow (e.g., VQE totals 133.0)
- **Per-type cost breakdown** — how much work belongs to each task type (e.g., `QUANTUM_CIRCUIT: 60.0`, `LINEAR_ALGEBRA: 30.0`), revealing where the computational weight lives
- **Classical vs. quantum task count** — a simple split showing the balance of the workflow
- **Weighted affinity cost per device type** — for each device type, sums `task.compute_cost * task.affinity[device_type]` across all tasks. This estimates the load each device type *would* see if tasks were assigned purely by affinity. For example, a QPU shows high weighted cost because quantum tasks have 0.95 affinity for it, while classical tasks contribute 0.0. This reveals which devices will be under the most pressure before the simulation even runs.

#### Parallelism Report

`CostAnalyzer.analyze_parallelism()` analyzes the DAG structure to determine:

- **Critical path length** — the longest weighted path through the DAG (using `data_size` edge weights). This is the theoretical minimum makespan — even with infinite hardware, you cannot finish faster than this serial chain of dependent tasks.
- **Critical path nodes** — which specific tasks form that bottleneck chain (e.g., in VQE the critical path spans 14 of 16 tasks, passing through both QPU execution rounds)
- **Max parallelism width** — the widest "level" of the DAG, computed by assigning each node to a level based on its deepest predecessor, then counting the most populated level. In VQE this is 2 — at best you can run `qpu_execute_1` and `qpu_execute_2` concurrently.
- **Parallelism ratio** — `total_cost / critical_path_length`. A ratio of 1.0 means the workflow is fully serial. Higher ratios indicate more opportunity for parallel speedup. However, the parallelism width limits how much of that ratio you can actually exploit — a high ratio with a low width means plenty of work exists but the DAG structure constrains concurrency.

Both reports are printed to stdout before simulation begins, giving an upfront understanding of the workflow's structure, cost distribution, and potential bottlenecks.

### Phase 3: Hardware Cluster Construction

**Module:** `qcsc/hardware/cluster.py`, `qcsc/hardware/device.py`

`HardwareCluster.default_cluster()` creates a heterogeneous set of 6 devices with realistic properties:

| Device | Count | Capacity | Speed | Memory | Special Properties |
|---|---|---|---|---|---|
| **CPU** | 2 | 8 concurrent tasks | 1.0x | 16 GB | 16 cores |
| **GPU** | 2 | 4 concurrent tasks | 5.0x | 40 GB | 100 TFLOPS |
| **AI Accelerator** | 1 | 4 concurrent tasks | 8.0x | 32 GB | 275 TOPS |
| **QPU** | 1 | 1 task (serial) | 3.0x | N/A | 127 qubits, coherence time, gate fidelity, latency variance, calibration every 50 ticks |

Each device tracks its own `current_load`, `utilization`, and `memory_available`. The `can_accept()` method checks both capacity and memory before allowing a task to be assigned. The QPU is notably limited to one task at a time and requires periodic calibration pauses.

### Phase 4: Task Scheduling and Dispatch

**Modules:** `qcsc/scheduler/scheduler.py`, `qcsc/hardware/mapper.py`

The `DynamicScheduler` and `HardwareMapper` work together to assign ready tasks to devices:

1. **Identify ready tasks:** The `TensorComputeGraph.get_ready_tasks()` method finds all `PENDING` tasks whose predecessors are all `COMPLETED`.

2. **Sort by urgency:** Ready tasks are sorted by their **node urgency** — the longest remaining path from the task to any sink in the DAG. Tasks on the critical path get dispatched first, which minimizes overall makespan.

3. **Score task-device pairs:** For each ready task, the `HardwareMapper` evaluates every available device using a weighted scoring function:
   ```
   score = 0.4 * affinity + 0.25 * availability + 0.2 * transfer_fit + 0.15 * memory_fit
   ```
   - **Affinity** (0.4): How well-suited this device type is for this task type (from the affinity matrix)
   - **Availability** (0.25): Inverse of current device utilization — prefers less-loaded devices
   - **Transfer fit** (0.2): Data transfer cost between devices
   - **Memory fit** (0.15): Whether the device has enough free memory for the task

4. **Dispatch:** The task is assigned to the highest-scoring device. The device's load and memory are updated, the task status changes to `RUNNING`, and a `ScheduledTask` record is created with the estimated end time (`start_time + compute_cost / device.speed_factor`).

5. **QPU stochastic latency:** If the assigned device is a QPU, a latency multiplier is sampled from a normal distribution `N(1.0, variance)` (clamped to a minimum of 0.5x). This models the inherent unpredictability of quantum hardware — the actual end time may differ significantly from the estimate.

### Phase 5: Discrete-Event Simulation Loop

**Module:** `qcsc/simulation/engine.py`

The `SimulationEngine` drives the main simulation loop. Rather than simulating every time unit, it jumps directly to the next meaningful event:

1. **Initial dispatch:** All tasks with no dependencies are scheduled immediately at tick 0.

2. **Advance clock:** The clock jumps to the earliest `actual_end_time` among all running tasks.

3. **Record device snapshots:** At each tick, the utilization of every device is recorded for later visualization.

4. **Process completions:** All tasks finishing at the current tick are marked `COMPLETED`. Their devices are released (load decremented, memory freed). The scheduler then checks each completed task's successors — if all of a successor's predecessors are now complete, that successor becomes ready for dispatch.

5. **QPU calibration:** Every 50 ticks, the QPU undergoes a calibration cycle. During calibration, the QPU is temporarily unavailable, modeling real quantum hardware maintenance.

6. **Feedback check** (if enabled): The adaptive feedback loop runs at configurable intervals — see Phase 6.

7. **Schedule new tasks:** Any newly-ready tasks are dispatched using the same mapping and scoring process from Phase 4.

8. **Repeat** until all tasks are `COMPLETED` or a safety limit of 10,000 iterations is reached. The loop also detects deadlocks — if no tasks are running but some remain pending, it warns and exits.

### Phase 6: Adaptive Feedback and Reallocation

**Module:** `qcsc/scheduler/feedback.py`

When enabled (`--feedback` flag), the `AdaptiveFeedbackLoop` periodically inspects the system state and makes dynamic reallocation decisions. It runs every N ticks (configurable via `feedback_interval`, default 5) and applies two strategies:

**Strategy 1 — QPU Latency Fallback:**
- For each task currently running on the QPU, the feedback loop compares elapsed time to estimated time
- If `elapsed / expected > 1.5` (the `latency_variance_threshold`), the task is considered stalled
- The loop finds the least-loaded GPU as a fallback (state-vector simulation can approximate quantum circuit execution classically)
- If a GPU has capacity and memory, the task is cancelled on the QPU and restarted on the GPU

**Strategy 2 — Load Balancing:**
- Calculates average utilization per device type
- Identifies **overloaded** types (>85% utilization) and **underloaded** types (<50% utilization)
- For tasks running on overloaded devices, checks if any underloaded device type has at least 0.3 affinity for that task
- If so, migrates the task to the least-loaded device of the underloaded type

**Reallocation constraints** prevent thrashing:
- Each task can be reallocated at most 2 times (`max_reallocations_per_task`)
- A cooldown of 10 ticks must pass between reallocations of the same task (`reallocation_cooldown`)

When a task is reallocated, it loses all progress — it restarts from scratch on the new device. The `MetricsCollector` records every reallocation event with its reason, source device, and target device.

### Phase 7: Metrics Collection and Visualization

**Modules:** `qcsc/scheduler/metrics.py`, `qcsc/visualization/`

Throughout the simulation, the `MetricsCollector` records data at every event:
- **Per-task metrics:** device assignment, estimated vs. actual execution time, start/end ticks, reallocation count
- **Device snapshots:** utilization and task load at each tick
- **Reallocation events:** which task moved, from/to devices, reason, and timestamp

After simulation completes, a summary is printed:
- **Makespan** — total simulation time from first dispatch to last completion
- **Average task latency** — mean execution time across all tasks
- **Average latency ratio** — actual time / estimated time (>1.0 means tasks took longer than expected on average)
- **Reallocation count** — how many times tasks were moved between devices
- **Per-device-type breakdown** — task count, average time, and total time for CPU, GPU, AI accelerator, and QPU

Three visualizations are then generated:

- **Gantt Chart** (`gantt_chart.png`): Horizontal bars showing each task's execution span on its assigned device. Tasks are colored by type (7 colors). Reallocated tasks have red borders and thicker outlines. Red arrows connect the old and new device positions to show migration paths.

- **DAG Graph** (`task_graph.png`): The workflow DAG rendered with a hierarchical layered layout (topological generations). Nodes are colored by their final assigned device type (green=CPU, blue=GPU, orange=AI accelerator, purple=QPU, gray=unassigned). Edge widths are proportional to data transfer size between tasks.

- **Dashboard** (`dashboard.png`): A 2x2 multi-panel figure combining:
  1. **Device utilization over time** — line plot showing how busy each device type was throughout the simulation
  2. **Task latency distribution** — box plot comparing execution time spreads across device types
  3. **Reallocation timeline** — scatter plot of reallocation events with annotations showing the from/to device migration
  4. **Summary statistics table** — makespan, task counts, average latency, reallocation count, and per-device-type breakdowns

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
