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

import hashlib
import os
import time

import numpy as np

from .bruteforce import BruteForceIndex
from .index.hnsw import HNSW
from .store import Store

# When a filter is active we ask the index for this many times k candidates,
# so post-filtering still has enough survivors to return a full k.
_FILTER_OVERFETCH = 10


def _graph_fingerprint(store: Store, name: str) -> str:
    """A hash of a collection's (id, vector) contents — NOT its metadata.

    The HNSW graph depends only on the vectors, so this fingerprint decides
    whether a saved graph is still valid for the current data. Metadata is
    deliberately excluded: editing a title or studio must NOT invalidate the
    graph. iter_vectors orders by id, so the hash is order-stable.
    """
    h = hashlib.sha256()
    for vid, vec, _meta in store.iter_vectors(name):
        h.update(int(vid).to_bytes(8, "little", signed=True))
        h.update(vec.tobytes())
    return h.hexdigest()


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
    def __init__(self, name: str, store: Store, dim: int, metric: str,
                 graph_path: str | None = None, **hnsw_params):
        self.name = name
        self.store = store
        self.dim = dim
        self.metric = metric
        self.graph_path = graph_path       # where to cache the serialized graph (or None)
        self.loaded_from_graph = False     # observability: did open() skip the rebuild?
        self.index = HNSW(dim, metric, **hnsw_params)
        # In-memory metadata cache so post-filtering doesn't hit SQLite per
        # candidate. Kept in sync on add()/build; rebuilt from the store on load.
        self._metadata: dict = {}

    def __contains__(self, id) -> bool:
        return id in self._metadata

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    def create(cls, name, store: Store, dim, metric="cosine",
               graph_path: str | None = None, **hnsw_params) -> "Collection":
        """Register the collection in SQLite and return a fresh Collection."""
        store.create_collection(name, dim, metric)
        return cls(name, store, dim, metric, graph_path=graph_path, **hnsw_params)

    @classmethod
    def open(cls, name, store: Store, graph_path: str | None = None, **hnsw_params) -> "Collection":
        """Open an existing collection: load the saved graph if valid, else rebuild.

        The graph is a cache; SQLite is the source of truth. We load the saved
        graph only if its fingerprint still matches the store's vectors —
        otherwise we rebuild and refresh the cache.
        """
        meta = store.get_collection(name)
        if meta is None:
            raise KeyError(f"collection {name!r} does not exist")
        dim, metric = meta
        col = cls(name, store, dim, metric, graph_path=graph_path, **hnsw_params)
        if not col._try_load_graph():
            col.build_from_store()
            col.save_graph()  # cache for next startup (no-op when graph_path is None)
        return col

    def build_from_store(self) -> None:
        """Rebuild the in-memory index from the durable store (source of truth)."""
        ids, matrix, metas = self.store.load_all(self.name)
        self.index = HNSW(self.dim, self.metric,
                          M=self.index.M,
                          ef_construction=self.index.ef_construction,
                          ef_search=self.index.ef_search,
                          seed=self.index.seed)   # preserve seed -> deterministic rebuild
        self._metadata = {}
        for i, vid in enumerate(ids):
            self.index.add(vid, matrix[i])
            self._metadata[vid] = metas[i]
        self.loaded_from_graph = False

    # ---- graph persistence ------------------------------------------------

    def save_graph(self) -> bool:
        """Serialize the graph + a content fingerprint. No-op if graph_path is None."""
        if not self.graph_path:
            return False
        parent = os.path.dirname(self.graph_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.index.save(self.graph_path)
        with open(self.graph_path + ".fp", "w", encoding="utf-8") as f:
            f.write(_graph_fingerprint(self.store, self.name))
        return True

    def _try_load_graph(self) -> bool:
        """Load the saved graph iff it exists and its fingerprint matches the store."""
        if not self.graph_path:
            return False
        fp_path = self.graph_path + ".fp"
        if not (os.path.exists(self.graph_path) and os.path.exists(fp_path)):
            return False
        with open(fp_path, encoding="utf-8") as f:
            saved_fp = f.read().strip()
        if saved_fp != _graph_fingerprint(self.store, self.name):
            return False  # data changed since the graph was saved -> rebuild
        self.index = HNSW.load(self.graph_path)
        # The graph doesn't carry metadata; rebuild that cache from the store.
        self._metadata = {vid: meta for vid, _vec, meta in self.store.iter_vectors(self.name)}
        self.loaded_from_graph = True
        return True

    # ---- writes -----------------------------------------------------------

    def add(self, id: int, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Upsert a vector into SQLite and the live index, consistently.

        If the id already exists we replace it (drop, then re-add) so the store
        and the index never disagree — the index is append-only, so an in-place
        overwrite isn't available.
        """
        if id in self:
            self.delete(id)
        self.store.upsert(self.name, id, vector, metadata)
        self.index.add(id, vector)
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
