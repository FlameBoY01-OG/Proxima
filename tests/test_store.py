"""Tests for the SQLite persistence layer.

The headline test is durability: data written, then read back from a freshly
reopened DB file, must be byte-for-byte intact.
"""

import json

import numpy as np
import pytest

from proxima.store import Store


@pytest.fixture
def db_path(tmp_path):
    # A real file (not :memory:) so we can close and reopen it to prove durability.
    return str(tmp_path / "test.db")


def test_create_and_list_collections(db_path):
    with Store(db_path) as s:
        s.create_collection("anime", dim=4, metric="cosine")
        s.create_collection("movies", dim=8, metric="l2")
        assert s.list_collections() == ["anime", "movies"]
        assert s.get_collection("anime") == (4, "cosine")
        assert s.get_collection("missing") is None


def test_create_collection_idempotent_but_guards_redefinition(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=4, metric="cosine")
        s.create_collection("c", dim=4, metric="cosine")  # same -> no-op
        with pytest.raises(ValueError):
            s.create_collection("c", dim=8, metric="cosine")  # conflict


def test_upsert_get_roundtrip(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=3, metric="cosine")
        vec = np.array([1.5, -2.0, 3.25], dtype=np.float32)
        s.upsert("c", 1, vec, {"genre": "shonen", "year": 1999})
        got_vec, got_meta = s.get("c", 1)
        np.testing.assert_array_equal(got_vec, vec)
        assert got_meta == {"genre": "shonen", "year": 1999}


def test_upsert_replaces_existing_id(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=2, metric="l2")
        s.upsert("c", 7, [1.0, 1.0], {"v": 1})
        s.upsert("c", 7, [9.0, 9.0], {"v": 2})
        assert s.count("c") == 1  # replaced, not duplicated
        vec, meta = s.get("c", 7)
        np.testing.assert_array_equal(vec, np.array([9.0, 9.0], dtype=np.float32))
        assert meta == {"v": 2}


def test_dim_mismatch_raises(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=3, metric="cosine")
        with pytest.raises(ValueError):
            s.upsert("c", 1, [1.0, 2.0])  # only 2 dims


def test_upsert_into_missing_collection_raises(db_path):
    with Store(db_path) as s:
        with pytest.raises(KeyError):
            s.upsert("nope", 1, [1.0, 2.0])


def test_delete(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=2, metric="l2")
        s.upsert("c", 1, [1.0, 2.0])
        assert s.delete("c", 1) is True
        assert s.get("c", 1) is None
        assert s.delete("c", 1) is False  # already gone


def test_load_all_matrix_and_metadata(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=2, metric="l2")
        s.upsert("c", 1, [0.0, 0.0], {"a": 1})
        s.upsert("c", 2, [1.0, 1.0], {"a": 2})
        s.upsert("c", 3, [2.0, 2.0], {"a": 3})
        ids, matrix, metas = s.load_all("c")
        assert ids == [1, 2, 3]
        assert matrix.shape == (3, 2)
        assert matrix.dtype == np.float32
        assert [m["a"] for m in metas] == [1, 2, 3]


def test_load_all_empty_collection(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=5, metric="cosine")
        ids, matrix, metas = s.load_all("c")
        assert ids == [] and metas == []
        assert matrix.shape == (0, 5)


def test_drop_collection_cascades_to_vectors(db_path):
    with Store(db_path) as s:
        s.create_collection("c", dim=2, metric="l2")
        s.upsert("c", 1, [1.0, 2.0])
        s.drop_collection("c")
        assert s.get_collection("c") is None
        # The vector row must be gone too (FK ON DELETE CASCADE).
        leftover = s._conn.execute(
            "SELECT COUNT(*) AS n FROM vectors WHERE collection = 'c'"
        ).fetchone()["n"]
        assert leftover == 0


def test_durability_survives_reopen(db_path):
    # ---- session 1: write, then close the connection entirely ----
    s1 = Store(db_path)
    s1.create_collection("anime", dim=4, metric="cosine")
    original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    s1.upsert("anime", 42, original, {"title": "Cowboy Bebop", "year": 1998})
    s1.close()

    # ---- session 2: open the SAME FILE fresh and read it back ----
    s2 = Store(db_path)
    assert s2.get_collection("anime") == (4, "cosine")
    vec, meta = s2.get("anime", 42)
    np.testing.assert_array_equal(vec, original)  # byte-for-byte intact
    assert meta == {"title": "Cowboy Bebop", "year": 1998}
    s2.close()


def test_metadata_stored_as_json_text(db_path):
    # Prove metadata really is JSON in the column (the schema-flexible choice).
    with Store(db_path) as s:
        s.create_collection("c", dim=1, metric="l2")
        s.upsert("c", 1, [1.0], {"nested": {"x": [1, 2, 3]}})
        raw = s._conn.execute(
            "SELECT metadata FROM vectors WHERE id = 1"
        ).fetchone()["metadata"]
        assert isinstance(raw, str)
        assert json.loads(raw) == {"nested": {"x": [1, 2, 3]}}
