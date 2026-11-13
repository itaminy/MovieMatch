"""Item-based cosine similarity on a sparse movie-by-user rating matrix."""

from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ml.data_loader import DEFAULT_DATA_DIR, load_data


def _positive_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


class ItemBasedCollaborativeRecommender:
    """Compare raw rating vectors; missing ratings are implicit sparse zeros.

    Repeated user/movie pairs are averaged. Incomplete rating rows are skipped;
    malformed non-null values and references to unknown movies are rejected.
    min_common_users counts distinct users, not rating rows.
    """

    def __init__(self, movies: pd.DataFrame | None = None,
                 ratings: pd.DataFrame | None = None, *,
                 min_common_users: int = 3,
                 data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        if not _positive_integer(min_common_users):
            raise ValueError("min_common_users must be a positive integer")
        self.min_common_users = int(min_common_users)
        if movies is None and ratings is None:
            data = load_data(data_dir)
            movies, ratings = data["movies"], data["ratings"]
        elif movies is None or ratings is None:
            raise ValueError("Provide both movies and ratings, or neither")
        for name, frame, columns in (
            ("movies", movies, ["movieId", "title", "genres"]),
            ("ratings", ratings, ["movieId", "userId", "rating"]),
        ):
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(f"{name}: missing columns: {', '.join(sorted(missing))}")
        self._movies = movies[["movieId", "title", "genres"]].copy().reset_index(drop=True)
        if not all(_positive_integer(value) for value in self._movies["movieId"]):
            raise ValueError("movies.movieId must contain positive integers without missing values")
        if self._movies["movieId"].duplicated().any():
            raise ValueError("movies.movieId must be unique")
        self._positions = {int(mid): pos for pos, mid in enumerate(self._movies["movieId"])}
        valid = ratings[["movieId", "userId", "rating"]].copy()
        for column in ("movieId", "userId"):
            if not all(_positive_integer(value) for value in valid[column].dropna()):
                raise ValueError(f"ratings.{column} must contain positive integers")
        if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
               or not np.isfinite(value) or value not in np.arange(0.5, 5.5, 0.5)
               for value in valid["rating"].dropna()):
            raise ValueError("ratings.rating must be a finite half-star value between 0.5 and 5")
        valid = valid.dropna(subset=["movieId", "userId", "rating"])
        if not valid["movieId"].isin(self._positions).all():
            raise ValueError("ratings contains unknown movieId values")
        valid = valid.groupby(["movieId", "userId"], as_index=False)["rating"].mean()
        user_codes, users = pd.factorize(valid["userId"], sort=True)
        rows = valid["movieId"].map(self._positions).to_numpy(dtype=int)
        self._ratings = csr_matrix(
            (valid["rating"].to_numpy(dtype=float), (rows, user_codes)),
            shape=(len(self._movies), len(users)),
        )
        # Integer counts avoid overflow for hundreds of common users.
        self._observed = self._ratings.copy().astype(np.int64)
        self._observed.data[:] = 1

    def _empty_result(self) -> pd.DataFrame:
        result = self._movies.iloc[:0].copy()
        result["similarity_score"] = pd.Series(dtype=float)
        return result

    def recommend(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return positive-similarity items with enough common distinct users.

        Unknown IDs and invalid arguments raise ValueError. Unrated queries
        or queries without eligible neighbors return an empty result.
        """
        if not _positive_integer(top_n):
            raise ValueError("top_n must be a positive integer")
        if not _positive_integer(movie_id):
            raise ValueError("movie_id must be a positive integer")
        if movie_id not in self._positions:
            raise ValueError(f"Unknown movie_id: {movie_id}")
        position = self._positions[movie_id]
        query = self._ratings[position:position + 1]
        if query.nnz == 0:
            return self._empty_result()
        # Both products have shape (1, number_of_movies), never (movies, movies).
        common = (self._observed[position:position + 1] @ self._observed.T).tocsr()
        eligible = common.indices[common.data >= self.min_common_users]
        eligible = eligible[eligible != position]
        if eligible.size == 0:
            return self._empty_result()
        scores = cosine_similarity(query, self._ratings[eligible], dense_output=False).tocsr()
        positive = scores.data > 0
        positions = eligible[scores.indices[positive]]
        result = self._movies.iloc[positions].copy()
        result["similarity_score"] = np.clip(scores.data[positive], 0.0, 1.0)
        return result.sort_values(
            ["similarity_score", "movieId"], ascending=[False, True], kind="stable"
        ).head(int(top_n)).reset_index(drop=True)
