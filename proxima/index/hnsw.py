"""Hand-rolled HNSW index — THE centerpiece (Phase 3).

Hierarchical Navigable Small World graph, implemented from scratch:
  - multi-layer graph with probabilistic layer assignment
  - greedy search descending from the top layer
  - bounded candidate heap (ef_search) trading recall vs latency
  - configurable M / ef_construction / ef_search

HARD RULE: no faiss / hnswlib / annoy / scann here. This is the from-scratch
signal interviewers probe. (hnswlib may appear in bench/ ONLY, as a baseline.)
"""
