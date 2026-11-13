"""Run python -m ml.demo_collaborative to inspect item-based recommendations."""

import argparse
from pathlib import Path
import sys

from ml.collaborative import ItemBasedCollaborativeRecommender
from ml.data_loader import DEFAULT_DATA_DIR, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-common-users", type=int, default=3)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        data = load_data(args.data_dir)
        movies = data["movies"]
        model = ItemBasedCollaborativeRecommender(
            movies, data["ratings"], min_common_users=args.min_common_users
        )
        for title in ("Toy Story (1995)", "Jumanji (1995)"):
            matches = movies.loc[movies["title"].eq(title)].sort_values("movieId")
            if matches.empty:
                raise ValueError(f"Movie title not found: {title}")
            movie_id = int(matches.iloc[0]["movieId"])
            print(f"\n{title} (movieId={movie_id}, min_common_users={model.min_common_users}) — top 10:")
            print(model.recommend(movie_id).to_string(
                index=False, float_format=lambda score: f"{score:.6f}"
            ))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Recommendations failed: {exc}\nSee README.md for MovieLens setup.\n")


if __name__ == "__main__":
    main()
