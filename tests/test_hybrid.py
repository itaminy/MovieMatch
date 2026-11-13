import numpy as np
import pandas as pd
import pytest

import ml.evaluation as evaluation
from ml.hybrid import HybridRecommender


@pytest.fixture
def data():
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4, 5], "title": list("ABCDE"),
                           "genres": ["Drama", "Drama", None, "Comedy", "Drama"]})
    ratings = pd.DataFrame([(1, 1, 4.), (1, 2, 4.), (3, 1, 4.), (3, 2, 4.),
                            (4, 3, 4.)], columns=["movieId", "userId", "rating"])
    return movies, ratings


@pytest.mark.parametrize("alpha", [0, 1])
def test_exact_endpoints(data, alpha):
    model = HybridRecommender(*data, alpha=alpha, min_common_users=1)
    component = model.content_model if alpha == 0 else model.collaborative_model
    for mid in [1, 3, 5]:
        pd.testing.assert_frame_equal(model.recommend(mid), component.recommend(mid))


def test_union_missing_contributions_and_ties(data):
    model = HybridRecommender(*data, alpha=.5, min_common_users=1)
    result = model.recommend(1)
    # Exact CB ties use IDs 2 then 5; CF cosine may be a few ulps below one.
    assert result.movieId.tolist()[:2] == [2, 5]
    assert set(result.movieId) == {2, 3, 5}
    assert result.similarity_score.tolist() == pytest.approx([.5, .5, .5])
    assert list(result.columns) == ["movieId", "title", "genres", "similarity_score"]
    assert model.recommend(3).movieId.tolist() == [1]
    assert model.recommend(4).empty


def test_weighted_scores_and_whole_candidate_pool(data, monkeypatch):
    model = HybridRecommender(*data, alpha=.25, min_common_users=1)
    def cb(mid, top_n):
        assert top_n == 5
        return pd.DataFrame({"movieId": [2, 3], "similarity_score": [1., .4]})
    def cf(mid, top_n):
        assert top_n == 5
        return pd.DataFrame({"movieId": [3, 5], "similarity_score": [.8, .9]})
    monkeypatch.setattr(model.content_model, "recommend", cb)
    monkeypatch.setattr(model.collaborative_model, "recommend", cf)
    result = model.recommend(1, 2)
    assert result.movieId.tolist() == [2, 3]
    assert result.similarity_score.tolist() == pytest.approx([.75, .5])


@pytest.mark.parametrize("alpha", [-.1, 1.1, np.nan, np.inf, True, None, "0.5"])
def test_invalid_alpha(data, alpha):
    with pytest.raises(ValueError, match="alpha"):
        HybridRecommender(*data, alpha=alpha)


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True, None])
def test_invalid_top_n(data, top_n):
    with pytest.raises(ValueError, match="top_n"):
        HybridRecommender(*data).recommend(1, top_n)


@pytest.mark.parametrize("mid", [999, True, None, 1.0])
def test_invalid_movie_id(data, mid):
    with pytest.raises(ValueError, match="movie_id"):
        HybridRecommender(*data).recommend(mid)


def test_shuffled_data_and_unchanged_inputs(data):
    movies, ratings = data
    original = movies.copy(deep=True)
    expected = HybridRecommender(*data, min_common_users=1).recommend(1)
    actual = HybridRecommender(movies.sample(frac=1, random_state=42),
        ratings.sample(frac=1, random_state=42), min_common_users=1).recommend(1)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(movies, original)


@pytest.fixture
def experiment():
    movies = pd.DataFrame({"movieId": range(1, 25), "title": [str(i) for i in range(24)],
                           "genres": ["Drama"] * 24})
    ratings = pd.DataFrame([(uid, mid, 4., pos + uid * 100)
        for uid, ids in [(1, range(1, 21)), (2, range(5, 25))]
        for pos, mid in enumerate(ids)], columns=["userId", "movieId", "rating", "timestamp"])
    return movies, ratings


def test_tuning_fit_has_no_validation_or_test_ratings(experiment, monkeypatch):
    movies, ratings = experiment
    outer_train, outer_test = evaluation.temporal_split(ratings)
    fit, validation = evaluation.temporal_split(outer_train)
    real_cf = evaluation.ItemBasedCollaborativeRecommender
    captured = []
    def cf(catalog, fit_ratings, **kwargs):
        captured.append(fit_ratings.copy())
        pd.testing.assert_frame_equal(fit_ratings, fit)
        pairs = set(zip(fit_ratings.userId, fit_ratings.movieId))
        assert not pairs & set(zip(validation.userId, validation.movieId))
        assert not pairs & set(zip(outer_test.userId, outer_test.movieId))
        assert set(catalog.movieId) == set(fit.movieId)
        return real_cf(catalog, fit_ratings, **kwargs)
    monkeypatch.setattr(evaluation, "ItemBasedCollaborativeRecommender", cf)
    alpha, details = evaluation.tune_alpha(movies, outer_train, min_common_users=1)
    assert len(captured) == 1
    rows = [row for row in details["metrics"] if row["k"] == 10]
    assert alpha == min(rows, key=lambda row: (-row["ndcg_at_k"], row["alpha"]))["alpha"]
    assert len(rows) == 5
    assert len({row["users"] for row in rows}) == 1


def test_outer_test_not_passed_to_tuner_and_final_refit_on_full_train(experiment, monkeypatch):
    movies, ratings = experiment
    outer_train, outer_test = evaluation.temporal_split(ratings)
    original_tune = evaluation.tune_alpha
    real_cf = evaluation.ItemBasedCollaborativeRecommender
    fits = []
    def cf(catalog, frame, **kwargs):
        fits.append(frame.copy())
        return real_cf(catalog, frame, **kwargs)
    def tune(catalog, train, **kwargs):
        pd.testing.assert_frame_equal(train, outer_train)
        return original_tune(catalog, train, **kwargs)
    monkeypatch.setattr(evaluation, "tune_alpha", tune)
    monkeypatch.setattr(evaluation, "ItemBasedCollaborativeRecommender", cf)
    metrics, summary = evaluation.evaluate(movies, ratings, include_hybrid=True, min_common_users=1)
    pd.testing.assert_frame_equal(fits[-1], outer_train)
    assert len(metrics) == 8 and set(metrics.users) == {2}
    changed = ratings.copy()
    changed.loc[changed.timestamp.isin(outer_test.timestamp), "rating"] = 5.
    _, other = evaluation.evaluate(movies, changed, include_hybrid=True, min_common_users=1)
    assert summary["validation"]["metrics"] == other["validation"]["metrics"]
    assert summary["validation"]["selected_alpha"] == other["validation"]["selected_alpha"]


def test_pipeline_hybrid_matches_direct_model_aggregation(experiment):
    movies, ratings = experiment
    train, _ = evaluation.temporal_split(ratings)
    catalog = movies[movies.movieId.isin(train.movieId)]
    seeds = {uid: group.movieId.tolist() for uid, group in train.groupby("userId")}
    seen = {uid: set(ids) for uid, ids in seeds.items()}
    hybrid = HybridRecommender(catalog, train, alpha=.25, min_common_users=1)
    users, ids, cb = evaluation._stream_rankings(hybrid.content_model, seeds, seen, catalog.movieId,
                                                10, return_scores=True)
    _, _, cf = evaluation._stream_rankings(hybrid.collaborative_model, seeds, seen, catalog.movieId,
                                          10, return_scores=True)
    actual = evaluation._rank_totals(users, ids, evaluation.combine_scores(cb, cf, .25), 10)
    assert actual == {uid: evaluation.aggregate_recommendations(hybrid, seeds[uid], seen[uid], len(catalog), 10)
                      for uid in users}


def test_no_validation_is_error_and_tie_break(experiment, monkeypatch):
    movies, ratings = experiment
    with pytest.raises(ValueError, match="validation"):
        evaluation.tune_alpha(movies, ratings.assign(rating=3.))
    alpha, _ = evaluation.tune_alpha(movies, ratings, alphas=[.5, .5], min_common_users=1)
    assert alpha == .5
    with pytest.raises(ValueError, match="alphas"):
        evaluation.tune_alpha(movies, ratings, alphas=[])
    monkeypatch.setattr(evaluation, "ndcg_at_k", lambda *args: 0.)
    alpha, _ = evaluation.tune_alpha(movies, ratings, alphas=[1, .75, .25], min_common_users=1)
    assert alpha == .25
