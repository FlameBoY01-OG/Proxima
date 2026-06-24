"""Tests for the FastAPI service, driven through FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from proxima.api.server import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "api.db"), seed=0)
    with TestClient(app) as c:
        yield c


def _make_collection(client, name="anime", dim=2, metric="l2"):
    return client.post("/collections", json={"name": name, "dim": dim, "metric": metric})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_collection(client):
    r = _make_collection(client)
    assert r.status_code == 201
    assert r.json() == {"name": "anime", "dim": 2, "metric": "l2"}


def test_create_duplicate_conflicts(client):
    _make_collection(client)
    assert _make_collection(client).status_code == 409


def test_create_bad_metric_rejected(client):
    r = client.post("/collections", json={"name": "c", "dim": 2, "metric": "manhattan"})
    assert r.status_code == 400


def test_upsert_and_search(client):
    _make_collection(client)
    points = [
        {"id": 1, "vector": [0.0, 0.0], "metadata": {"genre": "action"}},
        {"id": 2, "vector": [1.0, 0.0], "metadata": {"genre": "drama"}},
        {"id": 3, "vector": [5.0, 0.0], "metadata": {"genre": "action"}},
    ]
    r = client.post("/collections/anime/points", json={"points": points})
    assert r.status_code == 200
    assert r.json() == {"upserted": 3, "count": 3}

    r = client.post("/collections/anime/search", json={"query": [0.1, 0.0], "k": 2})
    body = r.json()
    ids = [hit["id"] for hit in body["results"]]
    assert ids == [1, 2]
    assert body["results"][0]["metadata"] == {"genre": "action"}
    assert "took_ms" in body


def test_search_with_filter(client):
    _make_collection(client)
    points = [
        {"id": 1, "vector": [0.0, 0.0], "metadata": {"genre": "action"}},
        {"id": 2, "vector": [0.2, 0.0], "metadata": {"genre": "drama"}},
        {"id": 3, "vector": [1.0, 0.0], "metadata": {"genre": "action"}},
    ]
    client.post("/collections/anime/points", json={"points": points})
    r = client.post("/collections/anime/search",
                    json={"query": [0.0, 0.0], "k": 2, "filter": {"genre": "action"}})
    ids = [hit["id"] for hit in r.json()["results"]]
    assert ids == [1, 3]


def test_search_dim_mismatch(client):
    _make_collection(client, dim=2)
    r = client.post("/collections/anime/search", json={"query": [1.0, 2.0, 3.0]})
    assert r.status_code == 400


def test_search_missing_collection(client):
    r = client.post("/collections/ghost/search", json={"query": [1.0, 2.0]})
    assert r.status_code == 404


def test_upsert_replace_same_id(client):
    _make_collection(client)
    client.post("/collections/anime/points",
                json={"points": [{"id": 1, "vector": [1.0, 1.0]}]})
    client.post("/collections/anime/points",
                json={"points": [{"id": 1, "vector": [9.0, 9.0], "metadata": {"v": 2}}]})
    r = client.post("/collections/anime/search", json={"query": [9.0, 9.0], "k": 1})
    body = r.json()
    assert body["results"][0]["id"] == 1
    assert body["results"][0]["metadata"] == {"v": 2}


def test_delete_point(client):
    _make_collection(client)
    client.post("/collections/anime/points",
                json={"points": [{"id": 1, "vector": [0.0, 0.0]},
                                 {"id": 2, "vector": [1.0, 1.0]}]})
    r = client.delete("/collections/anime/points/1")
    assert r.status_code == 200
    assert r.json() == {"deleted": 1, "count": 1}
    # gone now
    assert client.delete("/collections/anime/points/1").status_code == 404


def test_list_collections_with_counts(client):
    _make_collection(client, name="a", dim=2)
    _make_collection(client, name="b", dim=3, metric="cosine")
    client.post("/collections/a/points", json={"points": [{"id": 1, "vector": [0.0, 0.0]}]})
    cols = {c["name"]: c for c in client.get("/collections").json()["collections"]}
    assert set(cols) == {"a", "b"}
    assert cols["a"]["count"] == 1
    assert cols["b"]["metric"] == "cosine"


def test_drop_collection(client):
    _make_collection(client)
    assert client.delete("/collections/anime").status_code == 200
    assert client.delete("/collections/anime").status_code == 404


def test_metrics_endpoint(client):
    _make_collection(client, dim=4, metric="cosine")
    points = [{"id": i, "vector": [float(i), 0.0, 0.0, 1.0]} for i in range(20)]
    client.post("/collections/anime/points", json={"points": points})
    body = client.get("/collections/anime/metrics").json()
    assert body["vector_count"] == 20
    assert 0.0 <= body["recall_at_10"] <= 1.0
    assert body["avg_latency_ms"] >= 0.0
    assert body["qps"] > 0
