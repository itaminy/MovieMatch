"""Content-based recommendations using sparse genre TF-IDF features."""

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.data_loader import DEFAULT_DATA_DIR, load_data


def _genre_tokens(text: str) -> list[str]:
    """Split only on pipes: Sci-Fi and Film-Noir remain whole tokens."""
    return sorted({token.strip() for token in text.split("|")
                   if token.strip() and token.strip() != "(no genres listed)"})


class ContentBasedRecommender:
    """Fit genres once; compute a single query-to-catalog similarity vector.

    Pass a movies DataFrame for in-memory use, or omit it to load MovieLens
    through the existing loader. Missing genres give no recommendations and
    are excluded from candidate results. Valid zero-score candidates remain.
    """

    def __init__(self, movies: pd.DataFrame | None = None, *,
                 data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        if movies is None:
            movies = load_data(data_dir)["movies"]
        required = ["movieId", "title", "genres"]
        missing = set(required) - set(movies.columns)
        if missing:
            raise ValueError(f"Missing movie columns: {', '.join(sorted(missing))}")
        self._movies = movies[required].copy().reset_index(drop=True)
        ids = self._movies["movieId"]
        if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
               or value <= 0 for value in ids):
            raise ValueError("movieId values must be non-null positive integers")
        if ids.duplicated().any():
            raise ValueError("movieId values must be unique")
        for column in ("title", "genres"):
            if any(not isinstance(value, str) for value in self._movies[column].dropna()):
                raise ValueError(f"{column} values must be strings or missing")
        self._positions = {int(movie_id): index for index, movie_id in enumerate(ids)}
        documents = self._movies["genres"].fillna("").tolist()
        self._has_genres = np.array([bool(_genre_tokens(text)) for text in documents], dtype=bool)
        self.vectorizer = TfidfVectorizer(analyzer=_genre_tokens)
        # Empty/all-missing catalogs have no vocabulary, but remain valid inputs.
        self._features = (self.vectorizer.fit_transform(documents) if self._has_genres.any()
                          else csr_matrix((len(documents), 0), dtype=float))

    def _empty_result(self) -> pd.DataFrame:
        result = self._movies.iloc[:0].copy()
        result["similarity_score"] = pd.Series(dtype=float)
        return result

    def recommend(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return descending similarity, breaking ties by ascending movieId.

        Invalid arguments and unknown IDs raise ValueError. A genre-less query
        returns an empty DataFrame with the usual four result columns.
        top_n must be a positive integer; oversized requests return all candidates.
        """
        if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if isinstance(movie_id, (bool, np.bool_)) or not isinstance(movie_id, Integral):
            raise ValueError("movie_id must be an integer")
        if movie_id not in self._positions:
            raise ValueError(f"Unknown movie_id: {movie_id}")
        position = self._positions[movie_id]
        if not self._has_genres[position]:
            return self._empty_result()
        scores = cosine_similarity(self._features[position:position + 1], self._features).ravel()
        candidates = self._has_genres.copy()
        candidates[position] = False
        result = self._movies.loc[candidates].copy()
        result["similarity_score"] = np.clip(scores[candidates], 0.0, 1.0)
        return result.sort_values(
            ["similarity_score", "movieId"], ascending=[False, True], kind="stable"
        ).head(int(top_n)).reset_index(drop=True)
