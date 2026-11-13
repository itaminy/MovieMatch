"""One catalog and fitted model pair per application lifespan.

API serving uses ALL local ratings. It is separate from the train-only models
used by ml.evaluation; API responses must not be used to reproduce test metrics.
"""

from copy import copy
import json

from ml.data_loader import DEFAULT_DATA_DIR, load_data
from ml.hybrid import HybridRecommender


def _records(frame):
    # pandas nullable scalars and missing metadata become standard JSON values.
    return json.loads(frame.to_json(orient="records", double_precision=15))


class RecommendationService:
    def __init__(self, movies, ratings):
        self._movies = movies[["movieId", "title", "genres"]].sort_values("movieId").reset_index(drop=True).copy()
        self._hybrid = HybridRecommender(self._movies, ratings, alpha=1.0, min_common_users=3)
        self._by_id = {row["movieId"]: row for row in _records(self._movies)}

    @classmethod
    def from_directory(cls, data_dir=DEFAULT_DATA_DIR):
        data = load_data(data_dir)
        return cls(data["movies"], data["ratings"])

    def search_movies(self, query=None, limit=20):
        frame = self._movies
        if query:
            mask = frame["title"].astype("string").str.casefold().str.contains(
                query.casefold(), regex=False, na=False)
            frame = frame[mask]
        return _records(frame.head(limit))

    def get_movie(self, movie_id):
        # Keep cached dictionaries private to avoid accidental mutation.
        return dict(self._by_id[movie_id])

    def recommend(self, movie_id, method, top_n=10, alpha=1.0):
        movie = self.get_movie(movie_id)
        if method == "content_based":
            model = self._hybrid.content_model
        elif method == "item_based":
            model = self._hybrid.collaborative_model
        elif method == "hybrid":
            # Request-local wrapper shares fitted read-only components. Never
            # change alpha on the shared singleton: requests may run concurrently.
            model = copy(self._hybrid)
            model.alpha = alpha
        else:
            raise ValueError(f"Unknown method: {method}")
        result = model.recommend(movie_id, top_n=top_n)
        result = result[result["movieId"] != movie_id]
        return {"movie": movie, "method": method,
                "parameters": {"top_n": top_n, "alpha": alpha if method == "hybrid" else None,
                               "min_common_users": 3 if method != "content_based" else None},
                "recommendations": _records(result)}
