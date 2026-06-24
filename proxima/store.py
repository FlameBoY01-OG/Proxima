"""SQLite persistence layer (Phase 2) — the source of truth.

One row per vector: (id, collection, vector BLOB, metadata JSON). Vectors are
stored as float32 bytes; metadata as JSON text. SQLite gives us durability and
crash-safety for free, so we don't hand-roll a WAL / segments / recovery.
"""
