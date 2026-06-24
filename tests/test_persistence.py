"""Tests for graph persistence: load a saved HNSW on startup instead of rebuilding.

The graph is a CACHE; SQLite stays authoritative. A saved graph is used only if
its content fingerprint still matches the store.
"""

import numpy as np
import pytest

from proxima.db import Database
from proxima.store import Store


@pytest.fixture
def paths(tmp_path):
    return str(tmp_path / "p.db"), str(tmp_path / "graphs")


def _seed(db, n=60, dim=8):
    db.create_collection("c", dim=dim, metric="cosine")
    rng = np.random.default_rng(0)
    for i in range(n):
        db.add("c", i, rng.standard_normal(dim), {"i": i})
    return rng


def test_persist_then_reopen_loads_from_graph(paths):
    db_path, graph_dir = paths
    with Database(db_path, graph_dir=graph_dir, seed=1) as db:
        rng = _seed(db)
        db.persist("c")
        q = rng.standard_normal(8)
        before = db.search("c", q, k=5)

    # Fresh process: must load the cached graph, not rebuild.
    with Database(db_path, graph_dir=graph_dir, seed=1) as db2:
        col = db2.get_collection("c")
        assert col.loaded_from_graph is True
        after = col.search(q, k=5)
        assert [r[0] for r in after] == [r[0] for r in before]


def test_stale_graph_is_ignored_and_rebuilt(paths):
    db_path, graph_dir = paths
    with Database(db_path, graph_dir=graph_dir, seed=1) as db:
        _seed(db, n=60)
        db.persist("c")

    # Mutate the store behind the cache's back (add a 61st vector directly).
    with Store(db_path) as store:
        store.upsert("c", 999, np.ones(8, dtype=np.float32), {"i": 999})

    with Database(db_path, graph_dir=graph_dir, seed=1) as db2:
        col = db2.get_collection("c")
        assert col.loaded_from_graph is False   # fingerprint mismatch -> rebuild
        assert len(col) == 61


def test_metadata_edit_does_not_invalidate_graph(paths):
    db_path, graph_dir = paths
    with Database(db_path, graph_dir=graph_dir, seed=1) as db:
        _seed(db, n=40)
        db.persist("c")

    # Change ONLY metadata for id 0 (same vector). Graph must stay valid.
    with Store(db_path) as store:
        vec, _ = store.get("c", 0)
        store.upsert("c", 0, vec, {"i": 0, "note": "edited"})

    with Database(db_path, graph_dir=graph_dir, seed=1) as db2:
        col = db2.get_collection("c")
        assert col.loaded_from_graph is True            # vectors unchanged
        assert col._metadata[0] == {"i": 0, "note": "edited"}  # cache reflects edit


def test_persistence_disabled_without_graph_dir(paths):
    db_path, _ = paths
    with Database(db_path, seed=1) as db:   # no graph_dir
        _seed(db, n=20)
        assert db.persist("c") is False     # nothing to cache to

    with Database(db_path, seed=1) as db2:
        col = db2.get_collection("c")
        assert col.loaded_from_graph is False   # always rebuilds from SQLite


def test_drop_removes_graph_files(paths):
    db_path, graph_dir = paths
    import os
    with Database(db_path, graph_dir=graph_dir, seed=1) as db:
        _seed(db, n=10)
        db.persist("c")
        graph_file = db._graph_path("c")
        assert os.path.exists(graph_file)
        db.drop_collection("c")
        assert not os.path.exists(graph_file)
        assert not os.path.exists(graph_file + ".fp")


def test_add_replace_keeps_store_and_index_consistent(paths):
    """The add-consistency fix: re-adding an id must not desync store vs index."""
    db_path, graph_dir = paths
    with Database(db_path, graph_dir=graph_dir, seed=1) as db:
        db.create_collection("c", dim=2, metric="l2")
        db.add("c", 1, [1.0, 1.0], {"v": 1})
        db.add("c", 1, [9.0, 9.0], {"v": 2})   # replace, not a crash
        assert len(db.get_collection("c")) == 1
        assert db.store.count("c") == 1
        res = db.search("c", [9.0, 9.0], k=1)
        assert res[0][0] == 1
        assert db.get_collection("c")._metadata[1] == {"v": 2}
