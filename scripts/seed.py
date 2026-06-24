"""CLI for demo data — thin wrapper around proxima/demo.py.

    python -m scripts.seed --seed     # insert the demo dataset
    python -m scripts.seed --reset    # clear it
    python -m scripts.seed --seed --db mydata.db --collection anime

The CLI and the API (/demo/seed, /demo/reset) call the SAME functions in
demo.py, so there is exactly one definition of what "the demo data" is.
"""

from __future__ import annotations

import argparse

from proxima import demo
from proxima.db import Database


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed or reset Proxima demo data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", action="store_true", help="insert the demo dataset")
    group.add_argument("--reset", action="store_true", help="clear the demo dataset")
    parser.add_argument("--db", default="proxima.db", help="SQLite file (default: proxima.db)")
    parser.add_argument("--collection", default=demo.DEMO_COLLECTION,
                        help=f"collection name (default: {demo.DEMO_COLLECTION})")
    parser.add_argument("--extra-per-genre", type=int, default=0,
                        help="add N synthetic titles per genre for scale testing")
    args = parser.parse_args(argv)

    with Database(args.db) as db:
        if args.seed:
            count = demo.seed(db, args.collection, extra_per_genre=args.extra_per_genre)
            print(f"seeded {count} vectors into '{args.collection}' ({args.db})")
        else:
            removed = demo.reset(db, args.collection)
            print(f"reset '{args.collection}' - removed {removed} vectors ({args.db})")


if __name__ == "__main__":
    main()
