"""Collection (Phase 3) — owns one HNSW index + its dim + metric.

Builds/maintains the index from SQLite on startup, and exposes search and
metadata-filtered search. Filtering is post-filter (search index, then
intersect with a metadata predicate) — we'll note the recall tradeoff vs
pre-filtering.
"""
