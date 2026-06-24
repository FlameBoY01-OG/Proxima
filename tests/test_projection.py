"""Tests for the 2D projection helper and the projection/search_by_id endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from proxima import demo
from proxima.api.server import create_app
from proxima.projection import project_pca


def test_project_shape_and_emptiness():
    assert project_pca(np.empty((0, 16))).shape == (0, 2)
    coords = project_pca(np.random.default_rng(0).standard_normal((30, 16)))
    assert coords.shape == (30, 2)


def test_project_degenerate_inputs_are_padded():
    # Single point, or 1-D data: still must yield 2 columns.
    assert project_pca(np.ones((1, 16))).shape == (1, 2)
    assert project_pca(np.random.default_rng(1).standard_normal((10, 1))).shape == (10, 2)


def test_project_is_deterministic():
    data = np.random.default_rng(2).standard_normal((40, 16))
    np.testing.assert_array_equal(project_pca(data), project_pca(data))


def test_projection_keeps_clusters_separated():
    """Two well-separated blobs in 16-D should stay apart in 2-D."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((25, 16)) + 10.0   # blob A shifted far in every dim
    b = rng.standard_normal((25, 16)) - 10.0   # blob B shifted the other way
    coords = project_pca(np.vstack([a, b]))
    centroid_a = coords[:25].mean(axis=0)
    centroid_b = coords[25:].mean(axis=0)
    within_a = np.linalg.norm(coords[:25] - centroid_a, axis=1).mean()
    between = np.linalg.norm(centroid_a - centroid_b)
    assert between > 5 * within_a  # clusters far apart relative to their spread


@pytest.fixture
def seeded_client(tmp_path):
    app = create_app(str(tmp_path / "proj.db"), seed=0)
    client = TestClient(app)
    client.post("/demo/seed")
    return client


def test_projection_endpoint(seeded_client):
    body = seeded_client.get(f"/collections/{demo.DEMO_COLLECTION}/projection").json()
    pts = body["points"]
    assert len(pts) == len(demo.DEMO_TITLES)
    p = pts[0]
    assert {"id", "x", "y", "metadata"} <= set(p)
    assert "genre" in p["metadata"]


def test_search_by_id_endpoint(seeded_client):
    r = seeded_client.post(f"/collections/{demo.DEMO_COLLECTION}/search_by_id",
                           json={"id": 1, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query_id"] == 1
    assert body["results"][0]["id"] == 1          # a point is its own nearest
    assert len(body["results"]) == 5


def test_search_by_id_missing(seeded_client):
    r = seeded_client.post(f"/collections/{demo.DEMO_COLLECTION}/search_by_id",
                           json={"id": 99999, "k": 5})
    assert r.status_code == 404
