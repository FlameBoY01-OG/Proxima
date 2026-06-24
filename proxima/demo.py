"""Demo dataset + shared seed/reset logic.

Goal: make the 2D vector-space map in the UI *land*. We want clearly separated
clusters so search visibly happens "inside" a group.

HOW THE EMBEDDINGS ARE MADE — structured pseudo-embeddings, not real text
embeddings. Each genre is given one fixed random "center" vector; every title
is its genre's center plus a little noise. So titles of the same genre sit close
together and different genres sit apart. When the UI projects these to 2D with
PCA, you get tidy, well-separated blobs. (Real sentence embeddings would be more
authentic but noisier and would need a model download — out of scope here. The
generation is seeded, so the map is reproducible.)

This module is the single source of truth for the demo, used by BOTH the CLI
(scripts/seed.py) and the API (/demo/seed, /demo/reset).
"""

from __future__ import annotations

import numpy as np

from .db import Database

DEMO_COLLECTION = "anime"
DIM = 16
METRIC = "cosine"
_GENRE_SEED = 7          # seeds the per-genre center vectors (kept fixed)
_NOISE = 0.30            # how tight each cluster is (smaller = tighter)

# Curated sample: ~55 titles. metadata is {title, genre, year, studio}.
# (Details are approximate — this is demo data, not a catalog of record.)
DEMO_TITLES: list[dict] = [
    # --- action ---
    {"title": "Naruto", "genre": "action", "year": 2002, "studio": "Pierrot"},
    {"title": "Bleach", "genre": "action", "year": 2004, "studio": "Pierrot"},
    {"title": "Attack on Titan", "genre": "action", "year": 2013, "studio": "WIT"},
    {"title": "Demon Slayer", "genre": "action", "year": 2019, "studio": "ufotable"},
    {"title": "Jujutsu Kaisen", "genre": "action", "year": 2020, "studio": "MAPPA"},
    {"title": "My Hero Academia", "genre": "action", "year": 2016, "studio": "Bones"},
    {"title": "Hunter x Hunter", "genre": "action", "year": 2011, "studio": "Madhouse"},
    {"title": "One Punch Man", "genre": "action", "year": 2015, "studio": "Madhouse"},
    {"title": "Fullmetal Alchemist: Brotherhood", "genre": "action", "year": 2009, "studio": "Bones"},
    {"title": "Chainsaw Man", "genre": "action", "year": 2022, "studio": "MAPPA"},
    # --- romance ---
    {"title": "Your Name", "genre": "romance", "year": 2016, "studio": "CoMix Wave"},
    {"title": "Toradora!", "genre": "romance", "year": 2008, "studio": "J.C.Staff"},
    {"title": "Clannad", "genre": "romance", "year": 2007, "studio": "Kyoto Animation"},
    {"title": "Fruits Basket", "genre": "romance", "year": 2019, "studio": "TMS"},
    {"title": "Horimiya", "genre": "romance", "year": 2021, "studio": "CloverWorks"},
    {"title": "Kaguya-sama: Love is War", "genre": "romance", "year": 2019, "studio": "A-1"},
    {"title": "Nana", "genre": "romance", "year": 2006, "studio": "Madhouse"},
    # --- scifi ---
    {"title": "Steins;Gate", "genre": "scifi", "year": 2011, "studio": "White Fox"},
    {"title": "Cowboy Bebop", "genre": "scifi", "year": 1998, "studio": "Sunrise"},
    {"title": "Ghost in the Shell", "genre": "scifi", "year": 1995, "studio": "Production I.G"},
    {"title": "Psycho-Pass", "genre": "scifi", "year": 2012, "studio": "Production I.G"},
    {"title": "Akira", "genre": "scifi", "year": 1988, "studio": "TMS"},
    {"title": "Neon Genesis Evangelion", "genre": "scifi", "year": 1995, "studio": "Gainax"},
    {"title": "Dr. Stone", "genre": "scifi", "year": 2019, "studio": "TMS"},
    # --- fantasy ---
    {"title": "Frieren", "genre": "fantasy", "year": 2023, "studio": "Madhouse"},
    {"title": "Re:Zero", "genre": "fantasy", "year": 2016, "studio": "White Fox"},
    {"title": "Made in Abyss", "genre": "fantasy", "year": 2017, "studio": "Kinema Citrus"},
    {"title": "Mushoku Tensei", "genre": "fantasy", "year": 2021, "studio": "Studio Bind"},
    {"title": "Reincarnated as a Slime", "genre": "fantasy", "year": 2018, "studio": "8bit"},
    {"title": "The Seven Deadly Sins", "genre": "fantasy", "year": 2014, "studio": "A-1"},
    {"title": "Magi", "genre": "fantasy", "year": 2012, "studio": "A-1"},
    # --- slice_of_life ---
    {"title": "K-On!", "genre": "slice_of_life", "year": 2009, "studio": "Kyoto Animation"},
    {"title": "Barakamon", "genre": "slice_of_life", "year": 2014, "studio": "Kinema Citrus"},
    {"title": "March Comes in Like a Lion", "genre": "slice_of_life", "year": 2016, "studio": "Shaft"},
    {"title": "Violet Evergarden", "genre": "slice_of_life", "year": 2018, "studio": "Kyoto Animation"},
    {"title": "A Silent Voice", "genre": "slice_of_life", "year": 2016, "studio": "Kyoto Animation"},
    {"title": "Non Non Biyori", "genre": "slice_of_life", "year": 2013, "studio": "Silver Link"},
    # --- mystery ---
    {"title": "Death Note", "genre": "mystery", "year": 2006, "studio": "Madhouse"},
    {"title": "Monster", "genre": "mystery", "year": 2004, "studio": "Madhouse"},
    {"title": "Erased", "genre": "mystery", "year": 2016, "studio": "A-1"},
    {"title": "The Promised Neverland", "genre": "mystery", "year": 2019, "studio": "CloverWorks"},
    {"title": "Bungo Stray Dogs", "genre": "mystery", "year": 2016, "studio": "Bones"},
    {"title": "Hyouka", "genre": "mystery", "year": 2012, "studio": "Kyoto Animation"},
    # --- sports ---
    {"title": "Haikyuu!!", "genre": "sports", "year": 2014, "studio": "Production I.G"},
    {"title": "Kuroko's Basketball", "genre": "sports", "year": 2012, "studio": "Production I.G"},
    {"title": "Ace of Diamond", "genre": "sports", "year": 2013, "studio": "Madhouse"},
    {"title": "Free!", "genre": "sports", "year": 2013, "studio": "Kyoto Animation"},
    {"title": "Yuri on Ice", "genre": "sports", "year": 2016, "studio": "MAPPA"},
    {"title": "Run with the Wind", "genre": "sports", "year": 2018, "studio": "Production I.G"},
    # --- horror ---
    {"title": "Tokyo Ghoul", "genre": "horror", "year": 2014, "studio": "Pierrot"},
    {"title": "Parasyte", "genre": "horror", "year": 2014, "studio": "Madhouse"},
    {"title": "Another", "genre": "horror", "year": 2012, "studio": "P.A. Works"},
    {"title": "Hellsing Ultimate", "genre": "horror", "year": 2006, "studio": "Madhouse"},
    {"title": "Higurashi", "genre": "horror", "year": 2006, "studio": "Studio Deen"},
    {"title": "Junji Ito Collection", "genre": "horror", "year": 2018, "studio": "Studio Deen"},
]


def _genre_centers() -> dict[str, np.ndarray]:
    """One fixed random center vector per genre (sorted for determinism)."""
    genres = sorted({t["genre"] for t in DEMO_TITLES})
    rng = np.random.default_rng(_GENRE_SEED)
    return {g: rng.standard_normal(DIM).astype(np.float32) for g in genres}


def build_dataset() -> list[tuple[int, np.ndarray, dict]]:
    """Return [(id, vector, metadata), ...] for the whole demo set.

    ids are 1..N in list order. Each vector = its genre center + seeded noise,
    so reruns produce identical clusters.
    """
    centers = _genre_centers()
    rows: list[tuple[int, np.ndarray, dict]] = []
    for i, t in enumerate(DEMO_TITLES, start=1):
        # Seed noise off the id so each title is reproducible and distinct.
        noise = np.random.default_rng(1000 + i).standard_normal(DIM).astype(np.float32)
        vec = centers[t["genre"]] + _NOISE * noise
        meta = {"title": t["title"], "genre": t["genre"],
                "year": t["year"], "studio": t["studio"]}
        rows.append((i, vec, meta))
    return rows


def seed(db: Database, collection: str = DEMO_COLLECTION) -> int:
    """Insert the demo dataset. Idempotent: re-seeding clears first.

    Returns the resulting vector count.
    """
    if db.store.get_collection(collection) is None:
        db.create_collection(collection, dim=DIM, metric=METRIC)
    else:
        db.clear_collection(collection)  # wipe so re-seed doesn't duplicate ids
    col = db.get_collection(collection)
    for vid, vec, meta in build_dataset():
        col.add(vid, vec, meta)
    return len(col)


def reset(db: Database, collection: str = DEMO_COLLECTION) -> int:
    """Clear the demo collection's vectors (keep its definition). Returns count removed."""
    if db.store.get_collection(collection) is None:
        return 0
    return db.clear_collection(collection)
