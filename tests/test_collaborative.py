"""Item-based behavior on tiny synthetic rating tables."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

import ml.collaborative as collaborative
from ml.collaborative import ItemBasedCollaborativeRecommender


@pytest.fixture
def data():
    movies = pd.DataFrame({
        "movieId": [40, 30, 10, 20, 50, 60, 70],
        "title": ["Query", "Same B", "Same A", "Partial", "Disjoint", "Unrated", "One user"],
        "genres": ["Drama"] * 7,
    }, index=range(100, 107))
    ratings = pd.DataFrame([
        (40, 100, 3.0), (40, 200, 4.0),
        (30, 100, 3.0), (30, 200, 4.0),
        (10, 100, 3.0), (10, 200, 4.0),
        (20, 100, 4.0), (20, 200, 3.0),
        (50, 300, 5.0), (70, 100, 3.0),
    ], columns=["movieId", "userId", "rating"])
    return movies, ratings


def test_cosine_ranking_ties_and_schema(data):
    model = ItemBasedCollaborativeRecommender(*data, min_common_users=1)
    result = model.recommend(40)
    assert list(result.columns) == ["movieId", "title", "genres", "similarity_score"]
    assert result["movieId"].tolist() == [10, 30, 20, 70]
    assert result["similarity_score"].tolist() == pytest.approx([1, 1, 24 / 25, 3 / 5])
    assert (result["similarity_score"] > 0).all()
    assert result["similarity_score"].between(0, 1).all()


def test_common_users_filter(data):
    assert ItemBasedCollaborativeRecommender(*data, min_common_users=2).recommend(
        40
    )["movieId"].tolist() == [10, 30, 20]
    assert ItemBasedCollaborativeRecommender(*data, min_common_users=3).recommend(40).empty


def test_unrated_and_zero_similarity(data):
    model = ItemBasedCollaborativeRecommender(*data, min_common_users=1)
    assert model.recommend(60).empty
    assert model.recommend(50).empty
    assert list(model.recommend(60).columns) == ["movieId", "title", "genres", "similarity_score"]


def test_no_ratings_and_empty_catalog(data):
    movies, ratings = data
    assert ItemBasedCollaborativeRecommender(movies, ratings.iloc[:0]).recommend(40).empty
    empty = ItemBasedCollaborativeRecommender(movies.iloc[:0], ratings.iloc[:0])
    with pytest.raises(ValueError, match="Unknown movie_id"):
        empty.recommend(40)


@pytest.mark.parametrize("top_n", [0, -1, 1.5, "2", True, None, np.bool_(True)])
def test_bad_top_n(data, top_n):
    with pytest.raises(ValueError, match="top_n"):
        ItemBasedCollaborativeRecommender(*data).recommend(40, top_n)


@pytest.mark.parametrize("movie_id", [999, "40", 40.0, None, True, 0])
def test_bad_movie_id(data, movie_id):
    with pytest.raises(ValueError, match="movie_id"):
        ItemBasedCollaborativeRecommender(*data).recommend(movie_id)


@pytest.mark.parametrize("minimum", [0, -2, 2.5, True, None, "3"])
def test_bad_min_common_users(data, minimum):
    with pytest.raises(ValueError, match="min_common_users"):
        ItemBasedCollaborativeRecommender(*data, min_common_users=minimum)


def test_limit_numpy_ints_and_stability(data):
    movies, ratings = data
    original_movies, original_ratings = movies.copy(deep=True), ratings.copy(deep=True)
    model = ItemBasedCollaborativeRecommender(movies, ratings, min_common_users=np.int64(1))
    assert len(model.recommend(np.int64(40), np.int64(2))) == 2
    expected = model.recommend(40, 100)
    shuffled = ItemBasedCollaborativeRecommender(
        movies.sample(frac=1, random_state=5), ratings.sample(frac=1, random_state=2),
        min_common_users=1,
    )
    pd.testing.assert_frame_equal(expected, shuffled.recommend(40))
    expected.loc[0, "title"] = "Changed"
    assert model.recommend(40).iloc[0]["title"] == "Same A"
    pd.testing.assert_frame_equal(movies, original_movies)
    pd.testing.assert_frame_equal(ratings, original_ratings)


def test_duplicate_pairs_averaged_not_counted_twice(data):
    movies, ratings = data
    # Mean (1, 5) = 3, identical to the original query vector's first rating.
    ratings = ratings.drop(index=0)
    ratings = pd.concat([ratings, pd.DataFrame([
        (40, 100, 1.0), (40, 100, 5.0), (70, 100, 3.0),
    ], columns=ratings.columns)], ignore_index=True)
    model = ItemBasedCollaborativeRecommender(movies, ratings, min_common_users=2)
    result = model.recommend(40)
    assert result["movieId"].tolist() == [10, 30, 20]
    assert result.iloc[0]["similarity_score"] == pytest.approx(1)


def test_missing_rows_are_skipped(data):
    movies, ratings = data
    incomplete = pd.DataFrame({
        "movieId": pd.Series([40, None, 10], dtype="Int64"),
        "userId": pd.Series([None, 100, 100], dtype="Int64"),
        "rating": pd.Series([3, 4, None], dtype="Float64"),
    })
    combined = pd.concat([ratings.astype({"movieId": "Int64", "userId": "Int64"}), incomplete])
    expected = ItemBasedCollaborativeRecommender(movies, ratings, min_common_users=1).recommend(40)
    actual = ItemBasedCollaborativeRecommender(movies, combined, min_common_users=1).recommend(40)
    pd.testing.assert_frame_equal(expected, actual)


def test_sparse_query_only_similarity(data, monkeypatch):
    original_cosine = collaborative.cosine_similarity
    calls = []

    def spy(query, catalog, *, dense_output):
        assert issparse(query) and issparse(catalog)
        assert query.shape[0] == 1
        assert dense_output is False
        calls.append(query.shape)
        return original_cosine(query, catalog, dense_output=dense_output)

    monkeypatch.setattr(collaborative, "cosine_similarity", spy)
    model = ItemBasedCollaborativeRecommender(*data, min_common_users=1)
    assert issparse(model._ratings)
    assert model._ratings.shape == (7, 3)
    assert not calls
    model.recommend(40)
    assert len(calls) == 1


def test_many_common_users_do_not_overflow(data):
    movies, _ = data
    ratings = pd.DataFrame([(mid, uid, 4.0) for mid in (40, 10) for uid in range(1, 301)],
                           columns=["movieId", "userId", "rating"])
    result = ItemBasedCollaborativeRecommender(movies, ratings, min_common_users=300).recommend(40)
    assert result["movieId"].tolist() == [10]


def test_loading_through_existing_loader(data, monkeypatch):
    paths = []

    def loader(path):
        paths.append(path)
        return dict(zip(["movies", "ratings"], data))

    monkeypatch.setattr(collaborative, "load_data", loader)
    model = ItemBasedCollaborativeRecommender(data_dir="example", min_common_users=1)
    assert not model.recommend(40).empty
    assert paths == ["example"]


def test_invalid_schema_and_movie_references(data):
    movies, ratings = data
    with pytest.raises(ValueError, match="both"):
        ItemBasedCollaborativeRecommender(movies)
    with pytest.raises(ValueError, match="unique"):
        ItemBasedCollaborativeRecommender(pd.concat([movies, movies.iloc[:1]]), ratings)
    with pytest.raises(ValueError, match="missing columns"):
        ItemBasedCollaborativeRecommender(movies, ratings.drop(columns="userId"))
    with pytest.raises(ValueError, match="unknown movieId"):
        ItemBasedCollaborativeRecommender(movies, ratings.assign(movieId=999))


@pytest.mark.parametrize("value", [-1, 0, 4.2, 6, np.inf, "five", True])
def test_invalid_rating(data, value):
    movies, ratings = data
    ratings = ratings.astype({"rating": object})
    ratings.loc[0, "rating"] = value
    with pytest.raises(ValueError, match="ratings.rating"):
        ItemBasedCollaborativeRecommender(movies, ratings)
