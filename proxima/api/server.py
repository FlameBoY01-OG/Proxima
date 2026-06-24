"""FastAPI service — the HTTP surface over the database.

A thin, honest translation layer: parse/validate JSON with Pydantic, call into
the Database, shape the response. No business logic lives here — it all sits in
collection.py / hnsw.py / store.py, which keeps the API easy to test and the
core reusable from the CLI (Phase 4.5) and bench (Phase 6).

We expose an app *factory* (`create_app(db_path)`) so tests can point each app
at a throwaway SQLite file, while `app = create_app()` at module scope is what
uvicorn serves in production.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import demo
from ..db import Database
from ..distance import METRICS
from ..projection import project_pca


# ---- request / response models (Pydantic validates the JSON for us) -------

class CreateCollection(BaseModel):
    name: str
    dim: int = Field(gt=0)
    metric: str = "cosine"
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50


class Point(BaseModel):
    id: int
    vector: list[float]
    metadata: dict = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    points: list[Point]


class SearchRequest(BaseModel):
    query: list[float]
    k: int = Field(default=10, gt=0)
    ef_search: int | None = None
    filter: dict | None = None


class SearchHit(BaseModel):
    id: int
    distance: float
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchHit]
    took_ms: float


class SearchByIdRequest(BaseModel):
    id: int
    k: int = Field(default=10, gt=0)
    ef_search: int | None = None
    filter: dict | None = None


class SearchByIdResponse(BaseModel):
    query_id: int
    results: list[SearchHit]
    took_ms: float


class ProjectedPoint(BaseModel):
    id: int
    x: float
    y: float
    metadata: dict


class ProjectionResponse(BaseModel):
    points: list[ProjectedPoint]


# ---- app factory ----------------------------------------------------------

def create_app(db_path: str = "proxima.db", **db_kwargs) -> FastAPI:
    app = FastAPI(title="Proxima", version="0.4.0",
                  description="A hand-rolled vector database.")

    # The UI (Phase 5) runs on a different localhost port, so allow CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.db = Database(db_path, **db_kwargs)

    def get_db() -> Database:
        return app.state.db

    # ---- health -----------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ---- collections ------------------------------------------------------

    @app.post("/collections", status_code=201)
    def create_collection(req: CreateCollection, db: Database = Depends(get_db)):
        if req.metric not in METRICS:
            raise HTTPException(400, f"unknown metric {req.metric!r}; expected {METRICS}")
        if db.store.get_collection(req.name) is not None:
            raise HTTPException(409, f"collection {req.name!r} already exists")
        db.create_collection(req.name, req.dim, req.metric,
                             M=req.M, ef_construction=req.ef_construction,
                             ef_search=req.ef_search)
        return {"name": req.name, "dim": req.dim, "metric": req.metric}

    @app.get("/collections")
    def list_collections(db: Database = Depends(get_db)):
        out = []
        for name in db.list_collections():
            dim, metric = db.store.get_collection(name)
            out.append({"name": name, "dim": dim, "metric": metric,
                        "count": db.store.count(name)})
        return {"collections": out}

    @app.delete("/collections/{name}")
    def drop_collection(name: str, db: Database = Depends(get_db)):
        if db.store.get_collection(name) is None:
            raise HTTPException(404, f"collection {name!r} not found")
        db.drop_collection(name)
        return {"dropped": name}

    # ---- vectors ----------------------------------------------------------

    @app.post("/collections/{name}/points")
    def upsert_points(name: str, req: UpsertRequest, db: Database = Depends(get_db)):
        col = _require(db, name)
        upserted = 0
        for p in req.points:
            try:
                # If the id already lives in the in-memory index, treat it as a
                # replace: drop it from the store (rebuilds index), then re-add.
                if p.id in col.index._id_to_idx:
                    col.delete(p.id)
                col.add(p.id, np.asarray(p.vector, dtype=np.float32), p.metadata)
            except ValueError as e:
                raise HTTPException(400, str(e))
            upserted += 1
        return {"upserted": upserted, "count": len(col)}

    @app.delete("/collections/{name}/points/{point_id}")
    def delete_point(name: str, point_id: int, db: Database = Depends(get_db)):
        col = _require(db, name)
        if not col.delete(point_id):
            raise HTTPException(404, f"point {point_id} not found in {name!r}")
        return {"deleted": point_id, "count": len(col)}

    # ---- search -----------------------------------------------------------

    @app.post("/collections/{name}/search", response_model=SearchResponse)
    def search(name: str, req: SearchRequest, db: Database = Depends(get_db)):
        col = _require(db, name)
        q = np.asarray(req.query, dtype=np.float32)
        if q.shape[0] != col.dim:
            raise HTTPException(400, f"query dim {q.shape[0]} != collection dim {col.dim}")
        t0 = time.perf_counter()
        hits = col.search(q, k=req.k, ef_search=req.ef_search, filter=req.filter)
        took_ms = (time.perf_counter() - t0) * 1000.0
        results = [
            SearchHit(id=vid, distance=dist, metadata=col._metadata.get(vid, {}))
            for vid, dist in hits
        ]
        return SearchResponse(results=results, took_ms=took_ms)

    @app.post("/collections/{name}/search_by_id", response_model=SearchByIdResponse)
    def search_by_id(name: str, req: SearchByIdRequest, db: Database = Depends(get_db)):
        """Find neighbours of an existing point — 'show me titles like this one'.

        The query vector never leaves the server: we look it up by id, search,
        and return the neighbours. The UI uses this for click-to-search.
        """
        col = _require(db, name)
        row = col.store.get(name, req.id)
        if row is None:
            raise HTTPException(404, f"point {req.id} not found in {name!r}")
        vector, _meta = row
        t0 = time.perf_counter()
        hits = col.search(vector, k=req.k, ef_search=req.ef_search, filter=req.filter)
        took_ms = (time.perf_counter() - t0) * 1000.0
        results = [
            SearchHit(id=vid, distance=dist, metadata=col._metadata.get(vid, {}))
            for vid, dist in hits
        ]
        return SearchByIdResponse(query_id=req.id, results=results, took_ms=took_ms)

    # ---- 2D projection for the map ----------------------------------------

    @app.get("/collections/{name}/projection", response_model=ProjectionResponse)
    def projection(name: str, db: Database = Depends(get_db)):
        """PCA-project every vector to 2D so the UI can draw the vector space."""
        _require(db, name)
        ids, matrix, metas = db.store.load_all(name)
        coords = project_pca(matrix)
        points = [
            ProjectedPoint(id=vid, x=float(xy[0]), y=float(xy[1]), metadata=meta)
            for vid, xy, meta in zip(ids, coords, metas)
        ]
        return ProjectionResponse(points=points)

    # ---- metrics ----------------------------------------------------------

    @app.get("/collections/{name}/metrics")
    def metrics(name: str, db: Database = Depends(get_db)):
        col = _require(db, name)
        return col.metrics()

    # ---- demo data (drives the UI's seed / clear buttons) -----------------

    @app.post("/demo/seed")
    def demo_seed(db: Database = Depends(get_db)):
        count = demo.seed(db)
        return {"collection": demo.DEMO_COLLECTION, "count": count}

    @app.post("/demo/reset")
    def demo_reset(db: Database = Depends(get_db)):
        removed = demo.reset(db)
        return {"collection": demo.DEMO_COLLECTION, "removed": removed}

    # ---- shared helper ----------------------------------------------------

    def _require(db: Database, name: str):
        if db.store.get_collection(name) is None:
            raise HTTPException(404, f"collection {name!r} not found")
        return db.get_collection(name)

    return app


# What uvicorn serves: `uvicorn proxima.api.server:app --reload`
app = create_app()
