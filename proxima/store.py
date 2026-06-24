"""SQLite persistence layer — the source of truth.

DESIGN DECISION — delegate durability to SQLite.
Instead of hand-rolling a write-ahead log, on-disk segments, and crash
recovery, we store everything in SQLite: a proven, embedded, ACID database.
A commit is atomic and durable; if the process is killed mid-write, SQLite
recovers a consistent state on the next open. That lets us spend our actual
engineering effort on the HNSW index (Phase 3), and gives a clean interview
answer: "I engineered the index by hand and let a proven DB own durability."

ENCODING DECISIONS:
  - Vectors are stored as raw float32 BLOBs (numpy .tobytes() / .frombuffer()).
    Compact, exact round-trip, no parsing. d floats == 4*d bytes.
  - Metadata is a JSON text column. Schema-flexible: every vector can carry an
    arbitrary {genre, year, studio, ...} dict without ALTER TABLE migrations.

SCHEMA:
  collections(name PK, dim, metric)
  vectors(collection, id, vector BLOB, metadata JSON)  PK (collection, id)
          with FK collection -> collections(name) ON DELETE CASCADE
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Iterator, Optional

import numpy as np

# All vectors are stored/loaded as float32 — a deliberate, fixed choice so the
# BLOB byte-width is predictable (4 bytes/dim) and consistent with the index.
_DTYPE = np.float32


class Store:
    """A durable store of collections and their vectors, backed by one SQLite file."""

    def __init__(self, path: str = "proxima.db") -> None:
        """Open (creating if needed) the SQLite database at `path`.

        Pass ":memory:" for an ephemeral in-process DB (used by fast tests).
        """
        self.path = path
        # FastAPI runs our sync endpoints in a threadpool, so several requests
        # can hit this one connection at once. A sqlite3 connection is NOT safe
        # for concurrent cross-thread use, so we (a) allow cross-thread access
        # and (b) serialize every operation with a re-entrant lock. The lock is
        # re-entrant because some methods call others (upsert -> get_collection).
        # This is cheap: DB ops are fast and the index, not the DB, is the hot
        # path — so serializing access trades negligible latency for correctness.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        # Return rows as sqlite3.Row so we can index columns by name.
        self._conn.row_factory = sqlite3.Row
        # Enforce foreign keys (off by default in SQLite) so dropping a
        # collection cascades to delete its vectors.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    # ---- schema -----------------------------------------------------------

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name   TEXT PRIMARY KEY,
                dim    INTEGER NOT NULL,
                metric TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT    NOT NULL,
                id         INTEGER NOT NULL,
                vector     BLOB    NOT NULL,
                metadata   TEXT    NOT NULL DEFAULT '{}',
                PRIMARY KEY (collection, id),
                FOREIGN KEY (collection) REFERENCES collections(name)
                    ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    # ---- collections ------------------------------------------------------

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        """Register a collection. Idempotent if the (dim, metric) match."""
        with self._lock:
            existing = self.get_collection(name)
            if existing is not None:
                if existing != (dim, metric):
                    raise ValueError(
                        f"collection {name!r} already exists as {existing}, "
                        f"cannot redefine as {(dim, metric)}"
                    )
                return
            self._conn.execute(
                "INSERT INTO collections (name, dim, metric) VALUES (?, ?, ?)",
                (name, dim, metric),
            )
            self._conn.commit()

    def get_collection(self, name: str) -> Optional[tuple[int, str]]:
        """Return (dim, metric) for a collection, or None if it doesn't exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT dim, metric FROM collections WHERE name = ?", (name,)
            ).fetchone()
        return (row["dim"], row["metric"]) if row else None

    def list_collections(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM collections ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    def drop_collection(self, name: str) -> None:
        """Delete a collection and (via FK cascade) all of its vectors."""
        with self._lock:
            self._conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            self._conn.commit()

    def clear_collection(self, name: str) -> int:
        """Delete all vectors in a collection but KEEP its definition.

        Used by demo reset: the UI's "Clear all" should empty the map without
        forgetting the collection's dim/metric. Returns the number removed.
        """
        with self._lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE collection = ?", (name,))
            self._conn.commit()
            return cur.rowcount

    # ---- vectors ----------------------------------------------------------

    def upsert(
        self,
        collection: str,
        id: int,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert or replace one vector + its metadata.

        Validates the vector's dimension against the collection's declared dim.
        """
        with self._lock:
            meta = self._require_collection(collection)
            dim = meta[0]
            vec = np.asarray(vector, dtype=_DTYPE).reshape(-1)
            if vec.shape[0] != dim:
                raise ValueError(f"vector dim {vec.shape[0]} != collection dim {dim}")

            blob = vec.tobytes()  # raw float32 bytes
            meta_json = json.dumps(metadata or {})
            # INSERT OR REPLACE gives us upsert semantics keyed on (collection, id).
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors (collection, id, vector, metadata) "
                "VALUES (?, ?, ?, ?)",
                (collection, id, blob, meta_json),
            )
            self._conn.commit()

    def get(self, collection: str, id: int) -> Optional[tuple[np.ndarray, dict]]:
        """Return (vector, metadata) for one id, or None if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT vector, metadata FROM vectors WHERE collection = ? AND id = ?",
                (collection, id),
            ).fetchone()
        if row is None:
            return None
        return self._decode(row["vector"]), json.loads(row["metadata"])

    def delete(self, collection: str, id: int) -> bool:
        """Delete one vector. Returns True if a row was actually removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM vectors WHERE collection = ? AND id = ?", (collection, id)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count(self, collection: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM vectors WHERE collection = ?", (collection,)
            ).fetchone()
        return row["n"]

    def iter_vectors(self, collection: str) -> Iterator[tuple[int, np.ndarray, dict]]:
        """Stream (id, vector, metadata) rows for a collection, ordered by id.

        We fetch all rows under the lock (brief) and decode them outside it, so
        the connection lock is never held across the caller's iteration.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vector, metadata FROM vectors WHERE collection = ? ORDER BY id",
                (collection,),
            ).fetchall()
        for row in rows:
            yield row["id"], self._decode(row["vector"]), json.loads(row["metadata"])

    def load_all(self, collection: str) -> tuple[list[int], np.ndarray, list[dict]]:
        """Load an entire collection into memory for index building.

        Returns (ids, matrix (n, d) float32, metadatas). On startup the index is
        rebuilt from exactly this — SQLite is the source of truth, the in-memory
        index is a derived structure.
        """
        dim = self._require_collection(collection)[0]
        ids: list[int] = []
        metas: list[dict] = []
        vecs: list[np.ndarray] = []
        for id, vec, meta in self.iter_vectors(collection):
            ids.append(id)
            vecs.append(vec)
            metas.append(meta)
        matrix = np.stack(vecs) if vecs else np.empty((0, dim), dtype=_DTYPE)
        return ids, matrix, metas

    # ---- helpers ----------------------------------------------------------

    def _require_collection(self, name: str) -> tuple[int, str]:
        meta = self.get_collection(name)
        if meta is None:
            raise KeyError(f"collection {name!r} does not exist")
        return meta

    @staticmethod
    def _decode(blob: bytes) -> np.ndarray:
        # frombuffer returns a read-only view over the BLOB bytes; copy() so the
        # caller gets a normal writable array decoupled from the DB buffer.
        return np.frombuffer(blob, dtype=_DTYPE).copy()

    def close(self) -> None:
        self._conn.close()

    # Allow `with Store(path) as s:` for tidy open/close in tests and scripts.
    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
