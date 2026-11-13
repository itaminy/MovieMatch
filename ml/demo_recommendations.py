"""Run from the project root: python -m ml.demo_recommendations."""

import argparse
import sys
from pathlib import Path

from ml.content_based import ContentBasedRecommender
from ml.data_loader import DEFAULT_DATA_DIR, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    # Windows redirected stdout may use a legacy code page without movie accents.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        movies = load_data(args.data_dir)["movies"]
        model = ContentBasedRecommender(movies)
        for title in ("Toy Story (1995)", "Jumanji (1995)"):
            matches = movies.loc[movies["title"].eq(title)].sort_values("movieId")
            if matches.empty:
                raise ValueError(f"Movie title not found: {title}")
            movie_id = int(matches.iloc[0]["movieId"])
            print(f"\n{title} (movieId={movie_id}) — top 10:")
            print(model.recommend(movie_id, top_n=10).to_string(
                index=False, float_format=lambda score: f"{score:.6f}"
            ))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Recommendations failed: {exc}\nSee README.md for MovieLens setup.\n")


if __name__ == "__main__":
    main()
