"""Top-level database — manages many named collections over one SQLite file.

This is the object the API (Phase 4) talks to. It owns the Store and a cache of
live Collection objects. Collections are loaded lazily: the first time one is
accessed, its index is rebuilt from SQLite. That realizes the core commitment —
the database file is authoritative; the in-memory indexes are derived and
reconstructed on demand.
"""

from __future__ import annotations

import numpy as np

from .collection import Collection
from .store import Store


class Database:
    def __init__(self, path: str = "proxima.db", **default_hnsw_params) -> None:
        self.store = Store(path)
        self._default_params = default_hnsw_params
        self._collections: dict[str, Collection] = {}  # name -> live Collection

    # ---- collection management -------------------------------------------

    def create_collection(self, name: str, dim: int, metric: str = "cosine", **hnsw_params) -> Collection:
        params = {**self._default_params, **hnsw_params}
        col = Collection.create(name, self.store, dim, metric, **params)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Return a live Collection, loading + rebuilding its index if needed."""
        if name not in self._collections:
            self._collections[name] = Collection.open(name, self.store, **self._default_params)
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return self.store.list_collections()

    def drop_collection(self, name: str) -> None:
        self.store.drop_collection(name)            # cascades to vectors in SQLite
        self._collections.pop(name, None)           # evict the live index

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
