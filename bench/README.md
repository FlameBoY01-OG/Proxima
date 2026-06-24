# bench/ (Phase 6)

Recall@10 vs QPS harness. Sweeps `ef_search` against brute-force ground truth
on SIFT1M (or a GloVe subset) and emits a table / curve.

Optional: an isolated `hnswlib` baseline for sanity — **ask the owner before
adding it**, and keep it strictly out of the live index.

Datasets download into `bench/data/` (gitignored).
