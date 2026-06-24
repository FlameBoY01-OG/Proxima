"""Concurrency: the shared SQLite connection must be safe under parallel access.

FastAPI runs sync endpoints in a threadpool, so several requests can touch the
one connection at once (the UI fires projection + metrics together right after
seed/clear). Without serialization this raised sqlite errors intermittently —
the root cause behind the UI's occasional "Failed to fetch". These tests hammer
the store from many threads and assert nothing blows up.
"""

import concurrent.futures

import numpy as np

from proxima.db import Database


def test_store_survives_concurrent_access(tmp_path):
    db = Database(str(tmp_path / "c.db"), seed=0)
    db.create_collection("c", dim=8, metric="cosine")
    rng = np.random.default_rng(0)
    for i in range(120):
        db.add("c", i, rng.standard_normal(8), {"i": i})

    q = np.ones(8, dtype=np.float32)  # fixed query — no shared RNG across threads
    errors: list[Exception] = []

    def work(_n: int) -> None:
        try:
            for _ in range(25):
                db.store.count("c")
                db.store.load_all("c")          # full read through the connection
                list(db.store.iter_vectors("c"))
                db.search("c", q, k=5)          # read-only HNSW search
        except Exception as e:  # noqa: BLE001 - we want to surface ANY failure
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(8)))

    assert errors == [], f"concurrent access raised: {errors[:3]}"


def test_concurrent_reads_return_consistent_counts(tmp_path):
    db = Database(str(tmp_path / "c.db"), seed=0)
    db.create_collection("c", dim=4, metric="l2")
    for i in range(50):
        db.add("c", i, [float(i), 0.0, 0.0, 1.0], {"i": i})

    def count(_n: int) -> int:
        return db.store.count("c")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        counts = list(ex.map(count, range(64)))

    assert all(c == 50 for c in counts)
