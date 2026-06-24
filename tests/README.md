# tests/

One pytest file per module, added as each phase lands:

- `test_distance.py` (Phase 1)
- `test_store.py` (Phase 2)
- `test_hnsw.py` (Phase 3 — recall@10 vs brute-force ground truth)
- `test_api.py` (Phase 4 — via FastAPI TestClient)
- `test_demo.py` (Phase 4.5 — seed → count, reset → empty)

Run from the repo root:

    pytest
