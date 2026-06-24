"""Tests for distance.py, all on tiny vectors you can verify by hand."""

import numpy as np
import pytest

from proxima.distance import (
    METRICS,
    cosine_distance,
    distance,
    dot_distance,
    l2_distance,
)


def test_cosine_identical_is_zero():
    # Same direction => cosine similarity 1 => distance 0.
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v)[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_orthogonal_is_one():
    # Perpendicular vectors => similarity 0 => distance 1.
    q = np.array([1.0, 0.0])
    m = np.array([[0.0, 1.0]])
    assert cosine_distance(q, m)[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_opposite_is_two():
    # Opposite direction => similarity -1 => distance 2.
    q = np.array([1.0, 0.0])
    m = np.array([[-1.0, 0.0]])
    assert cosine_distance(q, m)[0] == pytest.approx(2.0, abs=1e-6)


def test_cosine_ignores_magnitude():
    # Cosine only cares about direction: scaling a vector changes nothing.
    q = np.array([1.0, 1.0])
    m = np.array([[2.0, 2.0], [10.0, 10.0]])
    d = cosine_distance(q, m)
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[1] == pytest.approx(0.0, abs=1e-6)


def test_cosine_zero_vector_does_not_explode():
    # A zero vector has no direction; the epsilon guard must keep it finite.
    q = np.array([1.0, 0.0])
    m = np.array([[0.0, 0.0]])
    d = cosine_distance(q, m)
    assert np.isfinite(d[0])


def test_l2_known_distances():
    # (0,0) -> (3,4) is the classic 3-4-5 triangle => distance 5.
    q = np.array([0.0, 0.0])
    m = np.array([[3.0, 4.0], [0.0, 0.0]])
    d = l2_distance(q, m)
    assert d[0] == pytest.approx(5.0)
    assert d[1] == pytest.approx(0.0)


def test_dot_distance_is_negated_inner_product():
    q = np.array([1.0, 2.0])
    m = np.array([[3.0, 4.0]])  # dot = 1*3 + 2*4 = 11
    assert dot_distance(q, m)[0] == pytest.approx(-11.0)


def test_dot_orders_larger_products_as_nearer():
    # Bigger dot product must come out as the SMALLER distance.
    q = np.array([1.0, 0.0])
    m = np.array([[5.0, 0.0], [1.0, 0.0]])
    d = dot_distance(q, m)
    assert d[0] < d[1]  # the larger-dot row is "nearer"


def test_dispatch_matches_direct_calls():
    q = np.array([1.0, 2.0, 3.0])
    m = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    for metric, fn in (("cosine", cosine_distance), ("l2", l2_distance), ("dot", dot_distance)):
        np.testing.assert_allclose(distance(metric, q, m), fn(q, m))


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        distance("manhattan", np.array([1.0]), np.array([[1.0]]))


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        l2_distance(np.array([1.0, 2.0]), np.array([[1.0, 2.0, 3.0]]))


def test_metrics_constant_is_complete():
    assert set(METRICS) == {"cosine", "l2", "dot"}
