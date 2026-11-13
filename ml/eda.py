"""MovieLens analysis: python -m ml.eda [--data-dir PATH] [--output-dir PATH]."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data_loader import DEFAULT_DATA_DIR, PROJECT_ROOT, load_data


def _movie_statistics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stats = data["ratings"].dropna(subset=["movieId"]).groupby("movieId")["rating"].agg(
        rating_count="count", mean_rating="mean"
    )
    movies = data["movies"].dropna(subset=["movieId"]).drop_duplicates("movieId")
    table = movies[["movieId", "title"]].merge(stats, on="movieId", how="outer")
    table["rating_count"] = table["rating_count"].fillna(0).astype(int)
    return table.sort_values(["rating_count", "movieId"], ascending=[False, True])


def _genres(movies: pd.DataFrame) -> pd.Series:
    genres = movies.drop_duplicates("movieId")["genres"].dropna().str.split("|").explode().str.strip()
    counts = genres[~genres.isin(["", "(no genres listed)"])].value_counts()
    return counts.rename_axis("genre").reset_index(name="count").sort_values(
        ["count", "genre"], ascending=[False, True]
    ).set_index("genre")["count"]


def analyze(data: dict[str, pd.DataFrame]) -> dict:
    """Describe raw rows, preserving duplicates and missing values in counts."""
    movies, ratings = data["movies"], data["ratings"]
    movie_stats = _movie_statistics(data)
    users = pd.concat([ratings["userId"], data["tags"]["userId"]])
    summary = {
        "counts": {
            "movies": int(movies["movieId"].nunique()),
            "users": int(users.nunique()),
            "rating_users": int(ratings["userId"].nunique()),
            "ratings": len(ratings),
            "valid_rating_values": int(ratings["rating"].count()),
            "tags": len(data["tags"]),
        },
        "tables": {name: {
            "rows": len(frame),
            "missing": {col: int(value) for col, value in frame.isna().sum().items()},
            "duplicate_rows": int(frame.duplicated().sum()),
        } for name, frame in data.items()},
        "duplicate_keys": {
            "movies_movieId": int(movies.duplicated(["movieId"]).sum()),
            "ratings_userId_movieId": int(ratings.duplicated(["userId", "movieId"]).sum()),
            "links_movieId": int(data["links"].duplicated(["movieId"]).sum()),
        },
        "mean_rating": None if ratings["rating"].dropna().empty else float(ratings["rating"].mean()),
        "mean_rating_by_user": [
            {"userId": int(uid), "mean_rating": None if pd.isna(mean) else float(mean)}
            for uid, mean in ratings.groupby("userId")["rating"].mean().items()
        ],
        "rating_distribution": {str(value): int(count) for value, count in
                                ratings["rating"].value_counts().sort_index().items()},
        "genres": {str(genre): int(count) for genre, count in _genres(movies).items()},
        "movie_statistics": json.loads(movie_stats.to_json(orient="records")),
        "top_10_movies": json.loads(movie_stats.head(10).to_json(orient="records")),
        "notes": "Raw rows retained; means skip missing ratings. Users include ratings and tags. "
                 "Genres count distinct movies; no-genres marker excluded. Duplicate movie IDs "
                 "use the first metadata row. Unknown rated movies remain in movie statistics.",
    }
    return summary


def run_eda(data_dir: str | Path = DEFAULT_DATA_DIR,
            output_dir: str | Path = PROJECT_ROOT / "reports") -> dict:
    data = load_data(data_dir)
    summary = analyze(data)
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(filename, title, xlabel, ylabel, draw):
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            draw(ax)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            fig.tight_layout()
            fig.savefig(figures / filename, dpi=150)
        finally:
            plt.close(fig)

    rating_values = data["ratings"]["rating"].dropna().to_numpy(dtype=float)
    save("rating_distribution.png", "Rating distribution", "Rating", "Number of ratings",
         lambda ax: ax.hist(rating_values, bins=np.arange(0.25, 5.76, 0.5)))
    movie_stats = _movie_statistics(data)
    save("ratings_per_movie.png", "Ratings per movie (including unrated movies)",
         "Number of ratings", "Number of movies",
         lambda ax: ax.hist(movie_stats["rating_count"].to_numpy(), bins=40))
    genres = _genres(data["movies"]).head(10).iloc[::-1]
    save("top_10_genres.png", "Top 10 genres", "Number of movies", "Genre",
         lambda ax: ax.barh(genres.index.tolist(), genres.to_numpy()))
    top = movie_stats.head(10).iloc[::-1]
    labels = [f"{row.title if pd.notna(row.title) else 'Unknown'} [id={row.movieId}]"
              for row in top.itertuples()]
    save("top_10_movies.png", "Top 10 movies by rating count", "Number of ratings", "Movie",
         lambda ax: ax.barh(labels, top["rating_count"].to_numpy()))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    try:
        summary = run_eda(args.data_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"EDA failed: {exc}\nDownload and extract the real MovieLens dataset; see README.md.\n")
    print(json.dumps(summary["counts"], indent=2))
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
