#!/usr/bin/env python3
"""QCSC Hybrid Task Partitioning and Dynamic Scheduling Simulator - CLI Entry Point."""

import argparse
import sys
from pathlib import Path

from qcsc.config import SchedulerConfig
from qcsc.parser.workflow_parser import WorkflowParser
from qcsc.parser.cost_analyzer import CostAnalyzer
from qcsc.hardware.cluster import HardwareCluster
from qcsc.simulation.engine import SimulationEngine
from qcsc.visualization.gantt import GanttChart
from qcsc.visualization.graph_viz import GraphVisualizer
from qcsc.visualization.dashboard import Dashboard


def main():
    parser = argparse.ArgumentParser(
        description="QCSC Hybrid Task Partitioning and Dynamic Scheduling Simulator"
    )
    parser.add_argument(
        "--workflow", required=True,
        help="Path to workflow YAML definition"
    )
    parser.add_argument(
        "--feedback", action="store_true", default=True,
        help="Enable adaptive feedback loop (default: enabled)"
    )
    parser.add_argument(
        "--no-feedback", action="store_true",
        help="Disable adaptive feedback loop"
    )
    parser.add_argument(
        "--latency-variance", type=float, default=0.3,
        help="QPU latency variance stddev (default: 0.3)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for visualizations (default: results/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress tick-by-tick output"
    )

    args = parser.parse_args()

    enable_feedback = not args.no_feedback
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse workflow
    print(f"Parsing workflow: {args.workflow}")
    tcg = WorkflowParser.parse(args.workflow)
    print(f"  Loaded {len(tcg)} tasks in workflow '{tcg.name}'")

    # 2. Analyze cost and parallelism
    cost_report = CostAnalyzer.analyze_cost(tcg)
    parallelism_report = CostAnalyzer.analyze_parallelism(tcg)
    CostAnalyzer.print_report(cost_report, parallelism_report)

    # 3. Build hardware cluster
    cluster = HardwareCluster.default_cluster()

    # 4. Configure and run simulation
    config = SchedulerConfig(
        qpu_latency_variance=args.latency_variance,
        random_seed=args.seed,
    )

    engine = SimulationEngine(
        tcg=tcg,
        cluster=cluster,
        config=config,
        enable_feedback=enable_feedback,
        verbose=not args.quiet,
    )

    print(f"\nRunning simulation (feedback={'ON' if enable_feedback else 'OFF'})...")
    metrics = engine.run()

    # 5. Generate visualizations
    print("\nGenerating visualizations...")
    GanttChart.render(tcg, metrics, output_dir / "gantt_chart.png")
    GraphVisualizer.render(tcg, output_dir / "tcg_graph.png")
    Dashboard.render(tcg, metrics, output_dir / "dashboard.png")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
