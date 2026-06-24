"""Recall@10 vs QPS benchmark for the hand-rolled HNSW index.

For each ef_search value we measure:
  - recall@10 : how often HNSW's top-10 matches the EXACT top-10 from the
                Phase-1 brute force (the ground truth)
  - QPS       : queries per second the index sustains at that setting

Run it:
    python -m bench.bench                         # synthetic data, default sweep
    python -m bench.bench --n 20000 --d 128       # bigger synthetic set
    python -m bench.bench --base sift_base.npy --query sift_query.npy   # real data

Datasets: synthetic Gaussian by default (reproducible, no download). To use a
real set (SIFT1M, a GloVe subset, ...), convert it to two .npy arrays — base
(n, d) and query (q, d) — and pass them with --base/--query.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from proxima.bruteforce import BruteForceIndex
from proxima.index.hnsw import HNSW

from .plot import render_curve_svg


def make_synthetic(n: int, d: int, queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal((queries, d)).astype(np.float32)
    return base, query


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int, metric: str) -> list[set]:
    """Exact top-k id sets for every query, via brute force."""
    bf = BruteForceIndex(base.shape[1], metric)
    bf.add_many(list(range(len(base))), base)
    return [{i for i, _ in bf.search(q, k)} for q in queries]


def run_benchmark(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
    M: int = 16,
    ef_construction: int = 200,
    ef_search_list: list[int] | None = None,
    seed: int = 0,
) -> list[dict]:
    ef_search_list = ef_search_list or [10, 20, 40, 80, 160, 320]
    truth = ground_truth(base, queries, k, metric)

    hn = HNSW(base.shape[1], metric, M=M, ef_construction=ef_construction, seed=seed)
    for i in range(len(base)):
        hn.add(i, base[i])

    rows: list[dict] = []
    for ef in ef_search_list:
        hits = 0
        t0 = time.perf_counter()
        for qi, q in enumerate(queries):
            got = {i for i, _ in hn.search(q, k, ef_search=ef)}
            hits += len(got & truth[qi])
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "ef_search": ef,
                "recall": hits / (k * len(queries)),
                "qps": len(queries) / elapsed,
                "avg_latency_ms": (elapsed / len(queries)) * 1000.0,
            }
        )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ef_search", "recall", "qps", "avg_latency_ms"])
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict], n: int, q: int, d: int) -> None:
    print(f"\nHNSW benchmark - {n} vectors, {d}-d, {q} queries, recall@10 vs brute force\n")
    print(f"{'ef_search':>10} {'recall@10':>11} {'QPS':>10} {'latency(ms)':>12}")
    print("-" * 46)
    for r in rows:
        print(f"{r['ef_search']:>10} {r['recall']:>11.3f} {r['qps']:>10.0f} {r['avg_latency_ms']:>12.3f}")
    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HNSW recall@10 vs QPS benchmark.")
    p.add_argument("--n", type=int, default=5000, help="synthetic base size")
    p.add_argument("--d", type=int, default=64, help="synthetic dimensionality")
    p.add_argument("--queries", type=int, default=500, help="number of queries")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metric", default="cosine", choices=["cosine", "l2", "dot"])
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", help="optional .npy base vectors (n, d)")
    p.add_argument("--query", help="optional .npy query vectors (q, d)")
    p.add_argument("--csv", default="bench/results.csv")
    p.add_argument("--svg", default="docs/recall_qps.svg")
    args = p.parse_args(argv)

    if args.base and args.query:
        base = np.load(args.base).astype(np.float32)
        queries = np.load(args.query).astype(np.float32)[: args.queries]
        print(f"loaded base {base.shape} / queries {queries.shape} from disk")
    else:
        base, queries = make_synthetic(args.n, args.d, args.queries, args.seed)

    rows = run_benchmark(
        base, queries, k=args.k, metric=args.metric, M=args.M,
        ef_construction=args.ef_construction, ef_search_list=args.ef_search, seed=args.seed,
    )
    print_table(rows, len(base), len(queries), base.shape[1])
    for out in (args.csv, args.svg):
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
    write_csv(rows, args.csv)
    render_curve_svg(rows, args.svg,
                     title=f"Recall@10 vs QPS — {len(base)} vecs, {base.shape[1]}d ({args.metric})")
    print(f"wrote {args.csv} and {args.svg}")


if __name__ == "__main__":
    main()
