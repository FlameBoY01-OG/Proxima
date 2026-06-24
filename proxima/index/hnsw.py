"""Hand-rolled HNSW index — THE centerpiece.

HNSW = Hierarchical Navigable Small World. A layered proximity graph:

  layer 2:        (A)---------(F)            <- few nodes, long hops
  layer 1:    (A)--(C)----(F)--(H)           <- more nodes
  layer 0:  (A)(B)(C)(D)(E)(F)(G)(H)(I)...    <- EVERY node, short hops

A search enters at the top and, on each layer, greedily walks toward the query
until it can't get closer; then it drops a layer and continues from where it
landed. Upper layers make big jumps across the space (cheap, coarse); lower
layers refine (precise). The result: we touch O(log n)-ish nodes instead of all
n. It is *approximate* — that's the whole point, and why we measure recall
against the brute-force ground truth from Phase 1.

THE KNOBS (recall vs latency):
  M               neighbours kept per node per layer (layer 0 gets 2*M)
  ef_construction breadth of the search while *building* (bigger = better graph)
  ef_search       breadth of the search while *querying* (bigger = higher recall)

HARD RULE: no faiss / hnswlib / annoy / scann here. Hand-rolled on purpose.
"""

from __future__ import annotations

import heapq
import math
import pickle
import random

import numpy as np

from .. import distance as _dist


class HNSW:
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.metric = metric
        self.M = M
        # Layer 0 is the densest and most important layer, so it gets more
        # connections. The standard choice is 2*M.
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # mL normalizes the level distribution. 1/ln(M) makes the expected
        # number of layers ~log_M(n), which is what gives the log-ish search.
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0
        # Seedable RNG so tests are deterministic. Layer heights are the only
        # randomness in the whole structure. We keep `seed` so a rebuild can
        # reproduce the same graph.
        self.seed = seed
        self._rng = random.Random(seed)

        # ---- storage (internal index = position in these parallel lists) ----
        self._vectors: list[np.ndarray] = []     # internal idx -> vector
        self._ids: list = []                      # internal idx -> external id
        self._id_to_idx: dict = {}                # external id -> internal idx
        self._node_level: list[int] = []          # internal idx -> its top layer
        # _graph[level] = {node: [neighbour nodes]}; one dict per layer.
        self._graph: list[dict[int, list[int]]] = []
        self._entry: int | None = None            # internal idx of the top entry point
        self._max_level: int = -1

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, external_id) -> bool:
        return external_id in self._id_to_idx

    # ---- distance helpers -------------------------------------------------

    def _distance_many(self, query: np.ndarray, nodes: list[int]) -> np.ndarray:
        """Distance from `query` to each node in `nodes` (one vectorized call)."""
        mat = np.stack([self._vectors[n] for n in nodes])
        return _dist.distance(self.metric, query, mat)

    # ---- level assignment -------------------------------------------------

    def _random_level(self) -> int:
        """Draw a node's top layer from an exponentially decaying distribution.

        floor(-ln(U) * mL): most nodes get level 0, a few get 1, fewer get 2...
        This is what makes upper layers sparse 'express lanes' without any
        global bookkeeping — each node decides its own height on insert.
        """
        return int(-math.log(self._rng.random()) * self.mL)

    def _max_conn(self, level: int) -> int:
        return self.M0 if level == 0 else self.M

    # ---- the core graph traversal ----------------------------------------

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """Greedy best-first search within ONE layer.

        Returns up to `ef` nearest (distance, node) pairs found on this layer,
        starting from `entry_points`.

        Two heaps do the work:
          - `candidates`: a min-heap of frontier nodes, nearest popped first
            (where to explore next).
          - `W`: a max-heap (negated distance) holding the best `ef` results so
            far, farthest on top — so we can both check the current worst and
            evict it in O(log ef). This bound is what keeps the search cheap.
        """
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        W: list[tuple[float, int]] = []
        for d, n in zip(self._distance_many(query, entry_points), entry_points):
            d = float(d)
            heapq.heappush(candidates, (d, n))
            heapq.heappush(W, (-d, n))

        while candidates:
            c_dist, c = heapq.heappop(candidates)
            farthest = -W[0][0]
            # If the closest unexplored candidate is already farther than the
            # worst result we're keeping, nothing better remains. Stop early.
            if c_dist > farthest:
                break

            neighbours = [n for n in self._graph[level].get(c, ()) if n not in visited]
            if not neighbours:
                continue
            for n in neighbours:
                visited.add(n)
            for d, n in zip(self._distance_many(query, neighbours), neighbours):
                d = float(d)
                farthest = -W[0][0]
                # Accept a neighbour if it beats our current worst, or if we
                # haven't yet filled the ef-sized result set.
                if d < farthest or len(W) < ef:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(W, (-d, n))
                    if len(W) > ef:
                        heapq.heappop(W)  # evict the farthest, keep ef best

        return [(-neg_d, n) for neg_d, n in W]

    def _select_neighbours(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Pick the `m` closest of a candidate set (simple selection).

        We use the simple 'keep the m nearest' rule. HNSW also has a diversity
        heuristic (prefer neighbours that aren't already mutually close, to
        avoid clustered, poorly-connected hubs); the simple rule is easier to
        defend and reason about, and recalls well at our scale. Noted as a
        tradeoff rather than hidden.
        """
        return [n for _, n in sorted(candidates)[:m]]

    # ---- insertion --------------------------------------------------------

    def add(self, external_id, vector: np.ndarray) -> None:
        """Insert one vector under `external_id`."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"vector dim {vec.shape[0]} != index dim {self.dim}")
        if external_id in self._id_to_idx:
            # Keep it simple: this index is append-only and rebuilt from SQLite
            # (the source of truth) when vectors change. No in-place update.
            raise ValueError(f"id {external_id!r} already present in index")

        idx = len(self._ids)
        self._ids.append(external_id)
        self._id_to_idx[external_id] = idx
        self._vectors.append(vec)
        level = self._random_level()
        self._node_level.append(level)

        # Make sure we have a dict for every layer up to this node's height,
        # and register the node (with no neighbours yet) on each of its layers.
        while len(self._graph) <= level:
            self._graph.append({})
        for lc in range(level + 1):
            self._graph[lc][idx] = []

        # First node ever: it becomes the entry point and we're done.
        if self._entry is None:
            self._entry = idx
            self._max_level = level
            return

        # Phase 1: from the top down to just above this node's level, greedily
        # descend with ef=1 to find a good entry point near the query.
        ep = [self._entry]
        for lc in range(self._max_level, level, -1):
            ep = [n for _, n in self._search_layer(vec, ep, ef=1, level=lc)]

        # Phase 2: from this node's level down to 0, do a wide (ef_construction)
        # search, pick neighbours, and wire up bidirectional edges.
        for lc in range(min(level, self._max_level), -1, -1):
            W = self._search_layer(vec, ep, self.ef_construction, lc)
            m = self._max_conn(lc)
            neighbours = self._select_neighbours(W, m)
            self._graph[lc][idx] = list(neighbours)
            for n in neighbours:
                self._graph[lc][n].append(idx)
                # Adding an edge may push n over its budget; re-select to prune
                # n back down to its m nearest neighbours.
                if len(self._graph[lc][n]) > self._max_conn(lc):
                    n_vec = self._vectors[n]
                    pairs = [
                        (float(d), nb)
                        for d, nb in zip(
                            self._distance_many(n_vec, self._graph[lc][n]),
                            self._graph[lc][n],
                        )
                    ]
                    self._graph[lc][n] = self._select_neighbours(pairs, self._max_conn(lc))
            # The nodes we found here seed the search on the next layer down.
            ep = [n for _, n in W]

        # If this node is taller than anything before, it's the new entry point.
        if level > self._max_level:
            self._max_level = level
            self._entry = idx

    # ---- search -----------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int = 10, ef_search: int | None = None
    ) -> list[tuple[object, float]]:
        """Return the k nearest (external_id, distance), nearest first."""
        if self._entry is None:
            return []
        vec = np.asarray(query, dtype=np.float32).reshape(-1)
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)  # never search narrower than the number we must return

        # Descend the express lanes greedily (ef=1) to land near the query...
        ep = [self._entry]
        for lc in range(self._max_level, 0, -1):
            ep = [n for _, n in self._search_layer(vec, ep, ef=1, level=lc)]
        # ...then do the one wide search at layer 0 where every node lives.
        W = self._search_layer(vec, ep, ef, level=0)
        W.sort(key=lambda dn: dn[0])
        return [(self._ids[n], d) for d, n in W[:k]]

    # ---- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the graph to disk so startup can skip a full rebuild.

        We pickle our own state dict. Pickle is fine here because this is a
        trusted artifact we produced; you would NOT unpickle untrusted input.
        The collection can always fall back to rebuilding from SQLite.
        """
        state = {
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self.seed,
            "vectors": self._vectors,
            "ids": self._ids,
            "id_to_idx": self._id_to_idx,
            "node_level": self._node_level,
            "graph": self._graph,
            "entry": self._entry,
            "max_level": self._max_level,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "HNSW":
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls(
            dim=state["dim"],
            metric=state["metric"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
            seed=state.get("seed"),
        )
        idx._vectors = state["vectors"]
        idx._ids = state["ids"]
        idx._id_to_idx = state["id_to_idx"]
        idx._node_level = state["node_level"]
        idx._graph = state["graph"]
        idx._entry = state["entry"]
        idx._max_level = state["max_level"]
        return idx
