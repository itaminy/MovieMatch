"""Weighted hybrid over a common, complete movie candidate pool."""

from numbers import Real
from pathlib import Path

import numpy as np

from ml.collaborative import ItemBasedCollaborativeRecommender, _positive_integer
from ml.content_based import ContentBasedRecommender
from ml.data_loader import DEFAULT_DATA_DIR, load_data


def validate_alpha(alpha):
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must be a finite number between 0 and 1")


def combine_scores(content_scores, collaborative_scores, alpha):
    """Identity normalization: both nonnegative cosine scores already use [0, 1]."""
    validate_alpha(alpha)
    return (1 - alpha) * content_scores + alpha * collaborative_scores


class HybridRecommender:
    """alpha=0 is Content-Based; alpha=1 is Collaborative (no automatic fallback)."""

    def __init__(self, movies=None, ratings=None, *, alpha=0.5, min_common_users=3,
                 data_dir: str | Path = DEFAULT_DATA_DIR):
        validate_alpha(alpha)
        if movies is None and ratings is None:
            data = load_data(data_dir)
            movies, ratings = data["movies"], data["ratings"]
        elif movies is None or ratings is None:
            raise ValueError("Provide both movies and ratings, or neither")
        self.alpha = float(alpha)
        self.content_model = ContentBasedRecommender(movies)
        self.collaborative_model = ItemBasedCollaborativeRecommender(
            movies, ratings, min_common_users=min_common_users)
        self._movies = movies[["movieId", "title", "genres"]].copy().reset_index(drop=True)
        self._positions = {int(mid): pos for pos, mid in enumerate(self._movies["movieId"])}

    def recommend(self, movie_id, top_n=10):
        validate_alpha(self.alpha)
        if not _positive_integer(top_n):
            raise ValueError("top_n must be a positive integer")
        if not _positive_integer(movie_id) or movie_id not in self._positions:
            raise ValueError(f"Unknown or invalid movie_id: {movie_id}")
        # Preserve the component's exact endpoint behavior, including CB zero scores.
        if self.alpha == 0:
            return self.content_model.recommend(movie_id, top_n)
        if self.alpha == 1:
            return self.collaborative_model.recommend(movie_id, top_n)
        count = max(1, len(self._movies))
        content = self.content_model.recommend(movie_id, top_n=count)
        collaborative = self.collaborative_model.recommend(movie_id, top_n=count)
        # Alignment on the full catalog gives absent candidates a zero contribution.
        ids = self._movies["movieId"]
        cb = ids.map(content.set_index("movieId")["similarity_score"]).fillna(0).to_numpy(dtype=float)
        cf = ids.map(collaborative.set_index("movieId")["similarity_score"]).fillna(0).to_numpy(dtype=float)
        scores = combine_scores(cb, cf, self.alpha)
        result = self._movies.copy()
        result["similarity_score"] = scores
        result = result[(result["movieId"] != movie_id) & (result["similarity_score"] > 0)]
        return result.sort_values(["similarity_score", "movieId"], ascending=[False, True],
                                  kind="stable").head(int(top_n)).reset_index(drop=True)
