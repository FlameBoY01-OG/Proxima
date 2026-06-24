"""Top-level database — manages many named collections over one SQLite file.

This is the object the API (Phase 4) talks to. It owns the Store and a cache of
live Collection objects. Collections are loaded lazily: the first time one is
accessed, its index is rebuilt from SQLite. That realizes the core commitment —
the database file is authoritative; the in-memory indexes are derived and
reconstructed on demand.
"""

from __future__ import annotations

import os

import numpy as np

from .collection import Collection
from .store import Store


class Database:
    def __init__(self, path: str = "proxima.db", graph_dir: str | None = None,
                 **default_hnsw_params) -> None:
        """`graph_dir`, if set, enables caching each collection's HNSW graph to
        disk so startup can load it instead of rebuilding (the graph stays a
        cache — SQLite remains authoritative and validates it via fingerprint).
        """
        self.store = Store(path)
        self.graph_dir = graph_dir
        self._default_params = default_hnsw_params
        self._collections: dict[str, Collection] = {}  # name -> live Collection

    def _graph_path(self, name: str) -> str | None:
        return os.path.join(self.graph_dir, f"{name}.hnsw") if self.graph_dir else None

    # ---- collection management -------------------------------------------

    def create_collection(self, name: str, dim: int, metric: str = "cosine", **hnsw_params) -> Collection:
        params = {**self._default_params, **hnsw_params}
        col = Collection.create(name, self.store, dim, metric,
                                graph_path=self._graph_path(name), **params)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Return a live Collection, loading + rebuilding its index if needed."""
        if name not in self._collections:
            self._collections[name] = Collection.open(
                name, self.store, graph_path=self._graph_path(name), **self._default_params
            )
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return self.store.list_collections()

    def drop_collection(self, name: str) -> None:
        self.store.drop_collection(name)            # cascades to vectors in SQLite
        self._collections.pop(name, None)           # evict the live index
        self._remove_graph_files(name)

    def clear_collection(self, name: str) -> int:
        """Empty a collection's vectors (keep its definition) and reset its index."""
        removed = self.store.clear_collection(name)
        if name in self._collections:
            col = self._collections[name]
            col.build_from_store()       # rebuild -> empty index
            col.save_graph()             # refresh the cache to match (no-op if disabled)
        else:
            self._remove_graph_files(name)
        return removed

    def persist(self, name: str) -> bool:
        """Explicitly cache a collection's current graph to disk. False if disabled."""
        return self.get_collection(name).save_graph()

    def _remove_graph_files(self, name: str) -> None:
        path = self._graph_path(name)
        if not path:
            return
        for p in (path, path + ".fp"):
            if os.path.exists(p):
                os.remove(p)

    # ---- convenience pass-throughs ---------------------------------------

    def add(self, collection: str, id: int, vector: np.ndarray, metadata: dict | None = None) -> None:
        self.get_collection(collection).add(id, vector, metadata)

    def search(self, collection: str, query: np.ndarray, k: int = 10, **kwargs):
        return self.get_collection(collection).search(query, k=k, **kwargs)

    def delete(self, collection: str, id: int) -> bool:
        return self.get_collection(collection).delete(id)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
