"""Unit tests for Phase 11 PerformanceBenchmark suite."""

import pytest
from src.benchmark.perf_benchmark import PerformanceBenchmark, BenchmarkResults


def test_performance_benchmark_dimensions():
    bench = PerformanceBenchmark(device="cpu")
    results = bench.run_all_benchmarks()

    assert isinstance(results, BenchmarkResults)
    assert results.hands_per_sec > 0.0
    assert results.decision_steps_per_sec > 0.0
    assert results.tournaments_per_sec > 0.0
    assert results.model_inference_per_sec > 0.0
    assert results.end_to_end_samples_per_sec > 0.0

    d = results.to_dict()
    assert len(d) == 5
    for k, v in d.items():
        assert v > 0.0
