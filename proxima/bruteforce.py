"""Brute-force (linear-scan) nearest-neighbour index — the ground truth.

This deliberately does the dumbest possible thing: to answer a query it
computes the distance to EVERY stored vector and returns the k smallest. That
is O(n * d) per query — hopeless at scale — but it is *exactly correct*.

We keep it forever for two reasons:
  1. In Phase 3, HNSW (which is approximate) is graded on how often its results
     match this exact answer. That ratio is "recall@k". Without an exact
     reference there is nothing to measure recall against.
  2. It's the simplest possible illustration of what an index is *for* — every
     optimization later exists to avoid this full scan.
"""

from __future__ import annotations

import numpy as np

from .distance import METRICS, distance


class BruteForceIndex:
    """An exact top-k index backed by a growing list of vectors."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
        self.dim = dim
        self.metric = metric
        # Parallel lists: _ids[i] is the external id of the vector in _vectors[i].
        # We keep them as Python lists while inserting (cheap appends) and only
        # stack into one numpy matrix at search time.
        self._ids: list[int] = []
        self._vectors: list[np.ndarray] = []
        # Cached (n, d) matrix; invalidated to None whenever we mutate.
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, id: int, vector: np.ndarray) -> None:
        """Add one vector under an integer id."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"vector dim {vec.shape[0]} != index dim {self.dim}")
        self._ids.append(id)
        self._vectors.append(vec)
        self._matrix = None  # invalidate the cached matrix

    def add_many(self, ids: list[int], vectors: np.ndarray) -> None:
        """Add a batch. `vectors` is an (n, d) array aligned with `ids`."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) array, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        for i, id in enumerate(ids):
            self._ids.append(id)
            self._vectors.append(vectors[i])
        self._matrix = None

    def _stacked(self) -> np.ndarray:
        """Return (and cache) the vectors as one (n, d) matrix for vectorized scoring."""
        if self._matrix is None:
            self._matrix = np.stack(self._vectors) if self._vectors else np.empty((0, self.dim), dtype=np.float32)
        return self._matrix

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Return the k nearest (id, distance) pairs, sorted nearest-first.

        Note: distance follows the convention in distance.py — smaller = nearer.
        """
        if len(self) == 0:
            return []
        k = min(k, len(self))
        matrix = self._stacked()
        dists = distance(self.metric, query, matrix)

        # argpartition finds the k smallest in O(n) without fully sorting the
        # rest — we only need the top-k, not a total ordering of everything.
        # It leaves those k unsorted, so we sort just that small slice afterward.
        topk_idx = np.argpartition(dists, k - 1)[:k]
        topk_idx = topk_idx[np.argsort(dists[topk_idx])]

        return [(self._ids[i], float(dists[i])) for i in topk_idx]
