"""Hardening pass: error paths and invariants across the stack.

These tests pin down behaviour at the edges (missing collections, bad dims,
ordering guarantees) so future changes can't silently regress them.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from proxima.api.server import create_app
from proxima.bruteforce import BruteForceIndex
from proxima.distance import cosine_distance
from proxima.index.hnsw import HNSW
from proxima.store import Store


# ---- store error paths ----------------------------------------------------

def test_store_load_all_missing_collection_raises(tmp_path):
    with Store(str(tmp_path / "s.db")) as s:
        with pytest.raises(KeyError):
            s.load_all("ghost")


def test_store_get_missing_returns_none(tmp_path):
    with Store(str(tmp_path / "s.db")) as s:
        s.create_collection("c", dim=2, metric="l2")
        assert s.get("c", 123) is None


def test_store_iter_missing_collection_is_empty(tmp_path):
    with Store(str(tmp_path / "s.db")) as s:
        assert list(s.iter_vectors("ghost")) == []


# ---- distance edge cases --------------------------------------------------

def test_cosine_both_zero_vectors_finite():
    d = cosine_distance(np.zeros(3), np.zeros((1, 3)))
    assert np.isfinite(d[0])


# ---- index invariants -----------------------------------------------------

def test_bruteforce_distances_sorted_ascending():
    idx = BruteForceIndex(2, "l2")
    idx.add_many([1, 2, 3, 4], np.array([[0, 0], [3, 0], [1, 0], [2, 0]], dtype=np.float32))
    dists = [d for _, d in idx.search([0.0, 0.0], k=4)]
    assert dists == sorted(dists)


def test_hnsw_distances_sorted_ascending():
    rng = np.random.default_rng(0)
    hn = HNSW(16, "cosine", seed=0)
    for i in range(100):
        hn.add(i, rng.standard_normal(16))
    dists = [d for _, d in hn.search(rng.standard_normal(16), k=10)]
    assert dists == sorted(dists)


def test_hnsw_contains():
    hn = HNSW(2, "l2", seed=0)
    hn.add(7, [1.0, 2.0])
    assert 7 in hn and 8 not in hn


# ---- API error paths ------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "api.db"), seed=0))


def test_api_upsert_wrong_dim_is_400(client):
    client.post("/collections", json={"name": "c", "dim": 2, "metric": "l2"})
    r = client.post("/collections/c/points", json={"points": [{"id": 1, "vector": [1, 2, 3]}]})
    assert r.status_code == 400


def test_api_metrics_missing_collection_is_404(client):
    assert client.get("/collections/ghost/metrics").status_code == 404


def test_api_projection_missing_collection_is_404(client):
    assert client.get("/collections/ghost/projection").status_code == 404


def test_api_create_collection_rejects_nonpositive_dim(client):
    # Pydantic Field(gt=0) -> 422 before our handler runs.
    assert client.post("/collections", json={"name": "c", "dim": 0, "metric": "l2"}).status_code == 422


def test_api_search_k_must_be_positive(client):
    client.post("/collections", json={"name": "c", "dim": 2, "metric": "l2"})
    client.post("/collections/c/points", json={"points": [{"id": 1, "vector": [0, 0]}]})
    assert client.post("/collections/c/search", json={"query": [0, 0], "k": 0}).status_code == 422
