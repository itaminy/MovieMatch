import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

import ml.content_based as content_based
from ml.content_based import ContentBasedRecommender


@pytest.fixture
def movies():
    return pd.DataFrame({
        "movieId": [40, 30, 10, 20, 50, 60, 70, 80],
        "title": ["Query", "Same B", "Same A", "Partial", "Other", "Missing", "Empty", "Null"],
        "genres": ["Sci-Fi|Film-Noir", "Film-Noir|Sci-Fi", "Sci-Fi|Film-Noir",
                   "Sci-Fi", "Drama", "(no genres listed)", "  ", None],
    }, index=range(100, 108))


def test_ranking_schema_and_cosine(movies):
    model = ContentBasedRecommender(movies)
    result = model.recommend(40)
    assert list(result.columns) == ["movieId", "title", "genres", "similarity_score"]
    assert result["movieId"].tolist() == [10, 30, 20, 50]
    assert result["similarity_score"].iloc[:2].tolist() == pytest.approx([1, 1])
    assert 0 < result.iloc[2]["similarity_score"] < 1
    assert result.iloc[3]["similarity_score"] == 0
    expected_features = model.vectorizer.transform(movies["genres"].fillna(""))
    expected = cosine_similarity(expected_features[0:1], expected_features).ravel()
    scores_by_id = dict(zip(movies["movieId"], expected))
    assert result["similarity_score"].tolist() == pytest.approx(
        [scores_by_id[mid] for mid in result["movieId"]]
    )


def test_compound_genres_are_whole_tokens(movies):
    model = ContentBasedRecommender(movies)
    assert set(model.vectorizer.get_feature_names_out()) == {"Sci-Fi", "Film-Noir", "Drama"}


def test_stable_ties_after_shuffling_and_no_mutation(movies):
    original = movies.copy(deep=True)
    model = ContentBasedRecommender(movies)
    expected = model.recommend(40)
    pd.testing.assert_frame_equal(expected, ContentBasedRecommender(
        movies.sample(frac=1, random_state=12)
    ).recommend(40))
    expected.loc[0, "title"] = "Changed"
    assert model.recommend(40).iloc[0]["title"] == "Same A"
    pd.testing.assert_frame_equal(movies, original)


@pytest.mark.parametrize("top_n", [0, -1, 1.5, "2", True, None, np.bool_(True)])
def test_invalid_top_n(movies, top_n):
    with pytest.raises(ValueError, match="top_n"):
        ContentBasedRecommender(movies).recommend(40, top_n)


@pytest.mark.parametrize("movie_id", [999, "40", 40.0, None, True])
def test_unknown_or_invalid_movie_id(movies, movie_id):
    with pytest.raises(ValueError, match="movie_id"):
        ContentBasedRecommender(movies).recommend(movie_id)


@pytest.mark.parametrize("movie_id", [60, 70, 80])
def test_genreless_query(movies, movie_id):
    result = ContentBasedRecommender(movies).recommend(movie_id)
    assert result.empty
    assert "similarity_score" in result.columns


def test_limits_and_numpy_integers(movies):
    model = ContentBasedRecommender(movies)
    assert len(model.recommend(np.int64(40), np.int64(2))) == 2
    assert len(model.recommend(40, 100)) == 4


def test_empty_all_genreless_and_single_catalog(movies):
    with pytest.raises(ValueError, match="Unknown movie_id"):
        ContentBasedRecommender(movies.iloc[:0]).recommend(40)
    assert ContentBasedRecommender(movies.iloc[5:]).recommend(60).empty
    assert ContentBasedRecommender(movies.iloc[:1]).recommend(40).empty


def test_no_pairwise_catalog_matrix(movies, monkeypatch):
    calls = []

    def query_cosine(query, catalog):
        calls.append((query.shape, catalog.shape))
        assert query.shape[0] == 1
        assert catalog.shape[0] == len(movies)
        return cosine_similarity(query, catalog)

    monkeypatch.setattr(content_based, "cosine_similarity", query_cosine)
    model = ContentBasedRecommender(movies)
    assert calls == []
    model.recommend(40)
    assert len(calls) == 1


def test_default_loading_uses_existing_loader(movies, monkeypatch):
    paths = []

    def loader(path):
        paths.append(path)
        return {"movies": movies}

    monkeypatch.setattr(content_based, "load_data", loader)
    assert not ContentBasedRecommender(data_dir="example").recommend(40).empty
    assert paths == ["example"]


def test_reject_ambiguous_ids_and_bad_schema(movies):
    with pytest.raises(ValueError, match="unique"):
        ContentBasedRecommender(pd.concat([movies, movies.iloc[:1]]))
    with pytest.raises(ValueError, match="Missing movie columns"):
        ContentBasedRecommender(movies.drop(columns="genres"))
    for bad_id in (None, True, 0, 1.5):
        bad = movies.copy()
        bad["movieId"] = bad["movieId"].astype(object)
        bad.iloc[0, bad.columns.get_loc("movieId")] = bad_id
        with pytest.raises(ValueError, match="movieId"):
            ContentBasedRecommender(bad)


def test_demo_handles_windows_output_encoding(movies, monkeypatch):
    import io
    import sys

    from ml import demo_recommendations

    movies = movies.copy()
    movies.iloc[0, movies.columns.get_loc("title")] = "Toy Story (1995)"
    movies.iloc[1, movies.columns.get_loc("title")] = "Jumanji (1995)"
    movies.iloc[2, movies.columns.get_loc("title")] = "Pokémon"
    monkeypatch.setattr(demo_recommendations, "load_data", lambda path: {"movies": movies})
    monkeypatch.setattr(sys, "argv", ["demo_recommendations"])
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1251")
    monkeypatch.setattr(sys, "stdout", output)
    demo_recommendations.main()
    output.flush()
    text = buffer.getvalue().decode("utf-8")
    assert "Toy Story (1995)" in text
    assert "Jumanji (1995)" in text
    assert "Pokémon" in text
