"""2D projection of high-dimensional vectors for the UI map.

Vectors live in many dimensions (16 in the demo); a screen has two. To draw the
"vector space" we squash everything down to 2D with PCA — Principal Component
Analysis finds the two directions along which the data varies most, and projects
onto them. It keeps as much of the spread as a 2D picture can, so clusters that
are separated in the original space tend to stay separated on screen.

We let scikit-learn own the linear algebra (a deliberate library choice — PCA is
standard, well-tested math, not the part of this project worth hand-rolling).
The fit is deterministic for a given dataset, so the map doesn't jitter between
calls; it only re-orients when the data itself changes (seed / clear).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def project_pca(matrix: np.ndarray) -> np.ndarray:
    """Project an (n, d) matrix to (n, 2) with PCA.

    Gracefully handles tiny/degenerate inputs (n < 2 or d < 2) by padding the
    missing axis with zeros, so the UI always gets 2 coordinates per point.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    n = matrix.shape[0]
    if n < 2:
        # 0 or 1 points: no variance to analyse. Place them at the origin.
        return np.zeros((n, 2), dtype=np.float32)

    d = matrix.shape[1]
    n_components = min(2, n, d)
    pca = PCA(n_components=n_components, svd_solver="full")
    coords = pca.fit_transform(matrix)

    if n_components < 2:
        # Pad with a zero column so callers always see exactly 2 dims.
        coords = np.hstack([coords, np.zeros((n, 2 - n_components), dtype=coords.dtype)])
    return coords.astype(np.float32)
