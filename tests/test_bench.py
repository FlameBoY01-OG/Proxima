"""Smoke tests for the benchmark harness (small synthetic sweep)."""

import numpy as np

from bench.bench import make_synthetic, run_benchmark, write_csv
from bench.plot import render_curve_svg


def test_run_benchmark_shape_and_monotonicity():
    base, queries = make_synthetic(n=300, d=16, queries=40, seed=0)
    rows = run_benchmark(base, queries, k=10, metric="cosine",
                         M=12, ef_construction=100, ef_search_list=[10, 80])
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r["recall"] <= 1.0
        assert r["qps"] > 0
        assert r["avg_latency_ms"] >= 0
    # More search breadth should not reduce recall (allow tiny noise).
    assert rows[1]["recall"] >= rows[0]["recall"] - 0.05


def test_high_ef_reaches_strong_recall():
    base, queries = make_synthetic(n=500, d=24, queries=50, seed=1)
    rows = run_benchmark(base, queries, ef_search_list=[200])
    assert rows[0]["recall"] >= 0.85


def test_outputs_written(tmp_path):
    base, queries = make_synthetic(n=200, d=16, queries=20, seed=2)
    rows = run_benchmark(base, queries, ef_search_list=[10, 50])
    csv_path = tmp_path / "r.csv"
    svg_path = tmp_path / "r.svg"
    write_csv(rows, str(csv_path))
    render_curve_svg(rows, str(svg_path))
    assert csv_path.read_text().startswith("ef_search,recall,qps")
    assert svg_path.read_text(encoding="utf-8").lstrip().startswith("<svg")
