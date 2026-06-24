"""Tests for Collection (index + store) and Database (many collections)."""

import numpy as np
import pytest

from proxima.collection import Collection
from proxima.db import Database
from proxima.store import Store


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "proxima.db")


def test_collection_add_and_search(db_path):
    with Store(db_path) as store:
        col = Collection.create("anime", store, dim=2, metric="l2", seed=0)
        col.add(1, [0.0, 0.0], {"genre": "action"})
        col.add(2, [1.0, 0.0], {"genre": "drama"})
        col.add(3, [5.0, 0.0], {"genre": "action"})
        results = col.search([0.1, 0.0], k=2)
        assert [r[0] for r in results] == [1, 2]


def test_filtered_search_post_filters_on_metadata(db_path):
    with Store(db_path) as store:
        col = Collection.create("anime", store, dim=2, metric="l2", seed=0)
        col.add(1, [0.0, 0.0], {"genre": "action"})
        col.add(2, [0.2, 0.0], {"genre": "drama"})   # closest, but filtered out
        col.add(3, [1.0, 0.0], {"genre": "action"})
        results = col.search([0.0, 0.0], k=2, filter={"genre": "action"})
        ids = [r[0] for r in results]
        assert ids == [1, 3]          # only action titles, nearest first
        assert 2 not in ids


def test_filter_accepts_membership_list(db_path):
    with Store(db_path) as store:
        col = Collection.create("c", store, dim=2, metric="l2", seed=0)
        col.add(1, [0.0, 0.0], {"genre": "action"})
        col.add(2, [1.0, 0.0], {"genre": "drama"})
        col.add(3, [2.0, 0.0], {"genre": "scifi"})
        results = col.search([0.0, 0.0], k=3, filter={"genre": ["action", "scifi"]})
        assert {r[0] for r in results} == {1, 3}


def test_rebuild_from_store_reconstructs_index(db_path):
    # Write through one Collection, then open a fresh one that rebuilds from SQLite.
    with Store(db_path) as store:
        col = Collection.create("c", store, dim=3, metric="cosine", seed=1)
        rng = np.random.default_rng(0)
        for i in range(50):
            col.add(i, rng.standard_normal(3), {"i": i})
        q = rng.standard_normal(3)
        before = col.search(q, k=5)

    # New process simulation: reopen the file and rebuild the index from scratch.
    with Store(db_path) as store2:
        reopened = Collection.open("c", store2, seed=1)
        assert len(reopened) == 50
        after = reopened.search(q, k=5)
        assert [r[0] for r in after] == [r[0] for r in before]
        # Metadata cache rebuilt too, so filtering still works.
        filtered = reopened.search(q, k=5, filter={"i": before[0][0]})
        assert filtered[0][0] == before[0][0]


def test_database_manages_multiple_collections(db_path):
    with Database(db_path, seed=0) as db:
        db.create_collection("a", dim=2, metric="l2")
        db.create_collection("b", dim=3, metric="cosine")
        db.add("a", 1, [0.0, 0.0], {"x": 1})
        db.add("b", 1, [1.0, 0.0, 0.0], {"x": 2})
        assert set(db.list_collections()) == {"a", "b"}
        assert db.search("a", [0.0, 0.0], k=1)[0][0] == 1


def test_database_lazy_loads_after_reopen(db_path):
    with Database(db_path, seed=2) as db:
        db.create_collection("c", dim=2, metric="l2")
        db.add("c", 1, [0.0, 0.0])
        db.add("c", 2, [9.0, 9.0])

    with Database(db_path, seed=2) as db2:
        # Never created in this process — must load + rebuild from SQLite.
        assert db2.list_collections() == ["c"]
        assert db2.search("c", [0.1, 0.1], k=1)[0][0] == 1


def test_drop_collection_evicts_everywhere(db_path):
    with Database(db_path, seed=0) as db:
        db.create_collection("c", dim=2, metric="l2")
        db.add("c", 1, [0.0, 0.0])
        db.drop_collection("c")
        assert db.list_collections() == []
        with pytest.raises(KeyError):
            db.get_collection("c")
