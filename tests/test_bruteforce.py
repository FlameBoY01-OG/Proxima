"""Tests for the brute-force index, on small sets with obvious answers."""

import numpy as np
import pytest

from proxima.bruteforce import BruteForceIndex


def _build(metric="l2"):
    """Four points on a line at x = 0, 1, 2, 3 (ids 10..13)."""
    idx = BruteForceIndex(dim=2, metric=metric)
    idx.add(10, [0.0, 0.0])
    idx.add(11, [1.0, 0.0])
    idx.add(12, [2.0, 0.0])
    idx.add(13, [3.0, 0.0])
    return idx


def test_len_tracks_inserts():
    idx = _build()
    assert len(idx) == 4


def test_topk_order_l2():
    idx = _build("l2")
    # Query near x=0 should rank ids by increasing x.
    results = idx.search([0.1, 0.0], k=3)
    ids = [r[0] for r in results]
    assert ids == [10, 11, 12]
    # Distances must come back ascending (nearest first).
    dists = [r[1] for r in results]
    assert dists == sorted(dists)


def test_nearest_to_a_stored_point_is_itself():
    idx = _build("l2")
    results = idx.search([2.0, 0.0], k=1)
    assert results[0][0] == 12
    assert results[0][1] == pytest.approx(0.0)


def test_k_larger_than_n_is_clamped():
    idx = _build("l2")
    results = idx.search([0.0, 0.0], k=100)
    assert len(results) == 4  # clamped to the number of vectors


def test_empty_index_returns_empty():
    idx = BruteForceIndex(dim=3, metric="cosine")
    assert idx.search([1.0, 2.0, 3.0], k=5) == []


def test_cosine_ranks_by_direction_not_distance():
    idx = BruteForceIndex(dim=2, metric="cosine")
    idx.add(1, [10.0, 0.0])   # same direction as query, far away
    idx.add(2, [0.0, 1.0])    # orthogonal, closer in raw distance
    results = idx.search([1.0, 0.0], k=2)
    assert results[0][0] == 1  # cosine ignores magnitude -> same direction wins


def test_add_many_matches_individual_adds():
    a = BruteForceIndex(dim=2, metric="l2")
    a.add_many([1, 2, 3], np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    b = BruteForceIndex(dim=2, metric="l2")
    b.add(1, [0.0, 0.0]); b.add(2, [1.0, 1.0]); b.add(3, [2.0, 2.0])
    assert a.search([0.0, 0.0], k=3) == b.search([0.0, 0.0], k=3)


def test_wrong_dim_raises():
    idx = BruteForceIndex(dim=2)
    with pytest.raises(ValueError):
        idx.add(1, [1.0, 2.0, 3.0])
