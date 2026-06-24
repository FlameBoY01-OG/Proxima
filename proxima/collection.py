"""Collection — owns one HNSW index plus its dim + metric.

A Collection ties together the two halves of the system:
  - SQLite (the durable source of truth) via the Store
  - the in-memory HNSW index (a derived structure) that actually answers queries

Every write goes to BOTH: SQLite for durability, the index for searchability.
On startup the index is rebuilt from SQLite (or loaded from a saved graph),
because the index is disposable and the database is authoritative.

METADATA FILTERING — post-filter, and we say so.
We search the index first, then drop results whose metadata fails the
predicate. The honest tradeoff: a very selective filter can starve the result
set (the k nearest might all be filtered out), so we over-fetch a wider
candidate set to compensate. The alternative, pre-filtering (restrict the
candidate set *before* graph search), needs the filter pushed into the index
and is harder to do well on a graph — a deliberate scope choice.
"""

from __future__ import annotations

import time

import numpy as np

from .bruteforce import BruteForceIndex
from .index.hnsw import HNSW
from .store import Store

# When a filter is active we ask the index for this many times k candidates,
# so post-filtering still has enough survivors to return a full k.
_FILTER_OVERFETCH = 10


def _matches(metadata: dict, predicate: dict) -> bool:
    """True if metadata satisfies every key in `predicate`.

    A predicate value may be a scalar (exact match) or a list/set (membership),
    which is exactly what UI toggles produce, e.g. {"genre": ["action", "scifi"]}.
    """
    for key, want in predicate.items():
        have = metadata.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class Collection:
    def __init__(self, name: str, store: Store, dim: int, metric: str, **hnsw_params):
        self.name = name
        self.store = store
        self.dim = dim
        self.metric = metric
        self.index = HNSW(dim, metric, **hnsw_params)
        # In-memory metadata cache so post-filtering doesn't hit SQLite per
        # candidate. Kept in sync on add()/build; rebuilt from the store on load.
        self._metadata: dict = {}

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    def create(cls, name, store: Store, dim, metric="cosine", **hnsw_params) -> "Collection":
        """Register the collection in SQLite and return a fresh Collection."""
        store.create_collection(name, dim, metric)
        return cls(name, store, dim, metric, **hnsw_params)

    @classmethod
    def open(cls, name, store: Store, **hnsw_params) -> "Collection":
        """Open an existing collection and rebuild its index from SQLite."""
        meta = store.get_collection(name)
        if meta is None:
            raise KeyError(f"collection {name!r} does not exist")
        dim, metric = meta
        col = cls(name, store, dim, metric, **hnsw_params)
        col.build_from_store()
        return col

    def build_from_store(self) -> None:
        """Rebuild the in-memory index from the durable store (source of truth)."""
        ids, matrix, metas = self.store.load_all(self.name)
        self.index = HNSW(self.dim, self.metric,
                          M=self.index.M,
                          ef_construction=self.index.ef_construction,
                          ef_search=self.index.ef_search)
        self._metadata = {}
        for i, vid in enumerate(ids):
            self.index.add(vid, matrix[i])
            self._metadata[vid] = metas[i]

    # ---- writes -----------------------------------------------------------

    def add(self, id: int, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Persist a vector to SQLite and insert it into the live index."""
        self.store.upsert(self.name, id, vector, metadata)
        self.index.add(id, vector)        # raises if id already indexed
        self._metadata[id] = metadata or {}

    def delete(self, id: int) -> bool:
        """Delete a vector from the store, then rebuild the index.

        Our HNSW is append-only (deleting a node from a proximity graph and
        repairing its neighbours' edges is fiddly and error-prone). Since the
        index is a *derived* structure and SQLite is authoritative, the simplest
        correct approach is: remove the row, then rebuild the index from the
        store. O(n) rebuild, but always consistent — fine at demo scale, and an
        honest tradeoff to state in an interview.
        """
        removed = self.store.delete(self.name, id)
        if removed:
            self.build_from_store()
        return removed

    def __len__(self) -> int:
        return len(self.index)

    # ---- search -----------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[object, float]]:
        """k nearest (id, distance). With `filter`, post-filter on metadata."""
        if filter is None:
            return self.index.search(query, k=k, ef_search=ef_search)

        # Over-fetch, then keep only the survivors that match the predicate.
        fetch = min(len(self.index), max(k * _FILTER_OVERFETCH, k))
        candidates = self.index.search(query, k=fetch, ef_search=ef_search)
        out = [
            (vid, dist)
            for vid, dist in candidates
            if _matches(self._metadata.get(vid, {}), filter)
        ]
        return out[:k]

    # ---- metrics ----------------------------------------------------------

    def metrics(self, sample: int = 50, k: int = 10, seed: int = 0) -> dict:
        """Measure recall@k, latency, and QPS of the live index right now.

        We grade the index honestly: load the collection's own vectors as a
        brute-force ground truth, then query the HNSW index with a random sample
        of stored vectors and compare. (Using stored points as probes is a quick
        self-consistency check, not a held-out benchmark — that's Phase 6's job.)
        """
        ids, matrix, _ = self.store.load_all(self.name)
        n = len(ids)
        if n == 0:
            return {"vector_count": 0, "recall_at_10": None,
                    "avg_latency_ms": None, "qps": None}

        bf = BruteForceIndex(self.dim, self.metric)
        bf.add_many(ids, matrix)

        rng = np.random.default_rng(seed)
        probes = rng.choice(n, size=min(sample, n), replace=False)
        kk = min(k, n)

        hits = 0
        t0 = time.perf_counter()
        for p in probes:
            got = {vid for vid, _ in self.index.search(matrix[p], k=kk)}
            truth = {vid for vid, _ in bf.search(matrix[p], k=kk)}
            hits += len(got & truth)
        elapsed = time.perf_counter() - t0
        m = len(probes)

        return {
            "vector_count": n,
            "recall_at_10": hits / (kk * m),
            "avg_latency_ms": (elapsed / m) * 1000.0,
            "qps": m / elapsed if elapsed > 0 else None,
        }
