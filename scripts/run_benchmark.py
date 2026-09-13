"""Run ApexPoker Comprehensive System and Performance Benchmark."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.benchmark.perf_benchmark import PerformanceBenchmark


def main():
    parser = argparse.ArgumentParser(description="Run Full ApexPoker System Benchmark")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to eval config")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--json-out", type=str, default="benchmark_report.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print(" APEXPOKER SYSTEM & PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    bench = PerformanceBenchmark(device=args.device)
    results = bench.run_all_benchmarks()
    results.print_summary()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Benchmark results successfully saved to: {args.json_out}")


if __name__ == "__main__":
    main()
