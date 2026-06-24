"""Tests for the demo dataset + seed/reset, via both the library and the API."""

import pytest
from fastapi.testclient import TestClient

from proxima import demo
from proxima.api.server import create_app
from proxima.db import Database


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "demo.db")


def test_dataset_is_well_formed():
    rows = demo.build_dataset()
    assert len(rows) == len(demo.DEMO_TITLES)
    ids = [r[0] for r in rows]
    assert ids == list(range(1, len(rows) + 1))      # ids 1..N, no gaps
    for _, vec, meta in rows:
        assert vec.shape == (demo.DIM,)
        assert set(meta) == {"title", "genre", "year", "studio"}


def test_seed_then_count(db_path):
    with Database(db_path, seed=0) as db:
        count = demo.seed(db)
        assert count == len(demo.DEMO_TITLES)
        assert db.store.count(demo.DEMO_COLLECTION) == count


def test_extra_per_genre_scales_dataset(db_path):
    n_genres = len({t["genre"] for t in demo.DEMO_TITLES})
    rows = demo.build_dataset(extra_per_genre=10)
    assert len(rows) == len(demo.DEMO_TITLES) + 10 * n_genres
    # ids stay contiguous 1..N and synthetic points carry full metadata.
    assert [r[0] for r in rows] == list(range(1, len(rows) + 1))
    assert all(set(r[2]) == {"title", "genre", "year", "studio"} for r in rows)

    with Database(db_path, seed=0) as db:
        count = demo.seed(db, extra_per_genre=10)
        assert count == len(demo.DEMO_TITLES) + 10 * n_genres


def test_reset_empties_but_keeps_collection(db_path):
    with Database(db_path, seed=0) as db:
        demo.seed(db)
        removed = demo.reset(db)
        assert removed == len(demo.DEMO_TITLES)
        assert db.store.count(demo.DEMO_COLLECTION) == 0
        # Definition survives a reset (dim/metric remembered).
        assert db.store.get_collection(demo.DEMO_COLLECTION) == (demo.DIM, demo.METRIC)


def test_reseed_is_idempotent(db_path):
    with Database(db_path, seed=0) as db:
        demo.seed(db)
        count = demo.seed(db)  # second seed must not duplicate ids
        assert count == len(demo.DEMO_TITLES)
        assert db.store.count(demo.DEMO_COLLECTION) == len(demo.DEMO_TITLES)


def test_clusters_are_separable(db_path):
    """A title's nearest neighbours should mostly share its genre."""
    with Database(db_path, seed=0) as db:
        demo.seed(db)
        col = db.get_collection(demo.DEMO_COLLECTION)
        rows = demo.build_dataset()
        same_genre_hits = 0
        checked = 0
        for vid, vec, meta in rows[:20]:
            results = col.search(vec, k=4)  # itself + 3 neighbours
            neighbour_genres = [col._metadata[r[0]]["genre"] for r in results if r[0] != vid]
            same_genre_hits += sum(g == meta["genre"] for g in neighbour_genres)
            checked += len(neighbour_genres)
        # With tight clusters, the vast majority of neighbours are same-genre.
        assert same_genre_hits / checked >= 0.8


def test_api_seed_and_reset(db_path):
    app = create_app(db_path, seed=0)
    with TestClient(app) as client:
        r = client.post("/demo/seed")
        assert r.status_code == 200
        body = r.json()
        assert body["collection"] == demo.DEMO_COLLECTION
        assert body["count"] == len(demo.DEMO_TITLES)

        # The seeded collection is now searchable through the normal endpoint.
        rows = demo.build_dataset()
        query = rows[0][1].tolist()
        s = client.post(f"/collections/{demo.DEMO_COLLECTION}/search",
                        json={"query": query, "k": 3})
        assert s.status_code == 200
        assert len(s.json()["results"]) == 3

        r = client.post("/demo/reset")
        assert r.json()["removed"] == len(demo.DEMO_TITLES)
        cols = {c["name"]: c for c in client.get("/collections").json()["collections"]}
        assert cols[demo.DEMO_COLLECTION]["count"] == 0
