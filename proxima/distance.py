"""Distance / similarity functions over numpy arrays.

DESIGN DECISION — everything is a *distance* where **smaller means nearer**:

    cosine  ->  1 - cosine_similarity   (range 0..2)
    l2      ->  Euclidean distance       (range 0..inf)
    dot     ->  -(dot product)           (so a bigger dot => smaller distance)

Why unify like this? Nearest-neighbour search only ever needs to ask "which of
these is closest?". If every metric answers in the same direction (keep the
smallest), then the index code (brute force now, HNSW in Phase 3) never has to
branch on the metric — it just keeps minima. The conversion lives here, in one
place, instead of being smeared across the search code.

Everything is vectorized: we score ONE query (shape (d,)) against MANY vectors
(shape (n, d)) in a single numpy call, returning an array of n distances. That
batching is what makes even the "slow" brute-force baseline fast enough to be
useful as ground truth.
"""

from __future__ import annotations

import numpy as np

# The metrics we support. Kept as a plain tuple so callers can validate cheaply.
METRICS = ("cosine", "l2", "dot")

# Guards a denominator against divide-by-zero (e.g. a zero vector has norm 0).
_EPS = 1e-12


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    """Coerce input to a 2-D float32 (n, d) matrix.

    Accepts a single vector (1-D) and promotes it to a 1-row matrix so the
    vectorized code below has a single shape to reason about.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def _as_query(query: np.ndarray, dim: int) -> np.ndarray:
    """Coerce the query to a 1-D float32 (d,) vector and check its dimension."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.shape[0] != dim:
        raise ValueError(f"query dim {q.shape[0]} != matrix dim {dim}")
    return q


def l2_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean distance from `query` to each row of `matrix`. Smaller = nearer."""
    matrix = _as_matrix(matrix)
    q = _as_query(query, matrix.shape[1])
    # matrix - q broadcasts q across every row; norm over axis=1 collapses the
    # per-dimension differences into one distance per row.
    return np.linalg.norm(matrix - q, axis=1)


def dot_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Negated inner product. Bigger dot product => smaller (nearer) distance.

    Maximum-inner-product search isn't a true metric space, but negating the
    dot product gives us the same "keep the smallest" ordering as the others.
    """
    matrix = _as_matrix(matrix)
    q = _as_query(query, matrix.shape[1])
    return -(matrix @ q)


def cosine_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """1 - cosine similarity. 0 = identical direction, 1 = orthogonal, 2 = opposite."""
    matrix = _as_matrix(matrix)
    q = _as_query(query, matrix.shape[1])
    # Cosine similarity = dot / (|a||b|). We compute the norms explicitly and
    # clamp them away from zero so a zero-vector doesn't blow up to NaN/inf.
    row_norms = np.linalg.norm(matrix, axis=1)
    q_norm = np.linalg.norm(q)
    sims = (matrix @ q) / (row_norms * q_norm + _EPS)
    return 1.0 - sims


# Lookup table: metric name -> the distance function above. The index code
# resolves the function once and stays metric-agnostic from then on.
_DISTANCES = {
    "cosine": cosine_distance,
    "l2": l2_distance,
    "dot": dot_distance,
}


def distance(metric: str, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Dispatch to the chosen metric's distance function (smaller = nearer)."""
    if metric not in _DISTANCES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return _DISTANCES[metric](query, matrix)
