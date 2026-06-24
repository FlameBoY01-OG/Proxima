"""Tests for the hand-rolled HNSW index.

The headline test grades HNSW against the Phase-1 brute force: recall@10 must
clear a threshold on random data.
"""

import numpy as np
import pytest

from proxima.bruteforce import BruteForceIndex
from proxima.index.hnsw import HNSW


def _random_data(n, d, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def test_empty_index_search_returns_empty():
    assert HNSW(dim=4).search(np.zeros(4), k=5) == []


def test_single_node():
    idx = HNSW(dim=3, metric="l2", seed=1)
    idx.add(99, [1.0, 2.0, 3.0])
    assert idx.search([1.0, 2.0, 3.0], k=1) == [(99, pytest.approx(0.0))]


def test_nearest_to_a_stored_point_is_itself():
    data = _random_data(300, 16, seed=2)
    idx = HNSW(dim=16, metric="cosine", seed=2)
    for i, v in enumerate(data):
        idx.add(i, v)
    for i in range(0, 300, 25):
        assert idx.search(data[i], k=1)[0][0] == i


def test_duplicate_id_rejected():
    idx = HNSW(dim=2, seed=0)
    idx.add(1, [0.0, 1.0])
    with pytest.raises(ValueError):
        idx.add(1, [1.0, 0.0])


def test_wrong_dim_rejected():
    idx = HNSW(dim=3, seed=0)
    with pytest.raises(ValueError):
        idx.add(1, [1.0, 2.0])


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_recall_against_brute_force(metric):
    """HNSW must approximately match the exact answer."""
    n, d, k = 1000, 32, 10
    data = _random_data(n, d, seed=7)
    queries = _random_data(50, d, seed=99)

    bf = BruteForceIndex(d, metric)
    hn = HNSW(d, metric, M=16, ef_construction=200, ef_search=50, seed=123)
    for i, v in enumerate(data):
        bf.add(i, v)
        hn.add(i, v)

    hits = 0
    for q in queries:
        truth = {i for i, _ in bf.search(q, k)}
        got = {i for i, _ in hn.search(q, k)}
        hits += len(truth & got)
    recall = hits / (k * len(queries))
    assert recall >= 0.90, f"recall@{k}={recall:.3f} below threshold ({metric})"


def test_higher_ef_search_does_not_reduce_recall():
    """ef_search is the recall/latency dial: more breadth shouldn't hurt recall."""
    n, d, k = 600, 24, 10
    data = _random_data(n, d, seed=3)
    queries = _random_data(20, d, seed=4)
    bf = BruteForceIndex(d, "cosine")
    hn = HNSW(d, "cosine", M=12, ef_construction=150, seed=5)
    for i, v in enumerate(data):
        bf.add(i, v); hn.add(i, v)

    def recall_at(ef):
        hits = 0
        for q in queries:
            truth = {i for i, _ in bf.search(q, k)}
            got = {i for i, _ in hn.search(q, k, ef_search=ef)}
            hits += len(truth & got)
        return hits / (k * len(queries))

    assert recall_at(100) >= recall_at(10) - 0.05  # allow tiny noise, expect >=


def test_save_and_load_roundtrip(tmp_path):
    data = _random_data(200, 16, seed=8)
    hn = HNSW(16, "cosine", seed=11)
    for i, v in enumerate(data):
        hn.add(i, v)

    path = str(tmp_path / "graph.hnsw")
    hn.save(path)
    loaded = HNSW.load(path)

    assert len(loaded) == len(hn)
    # Identical results before and after a save/load cycle.
    for q in data[:10]:
        assert loaded.search(q, k=5) == hn.search(q, k=5)
