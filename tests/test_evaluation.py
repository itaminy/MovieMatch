import numpy as np
import pandas as pd
import pytest

import ml.evaluation as evaluation
from ml.evaluation import (aggregate_recommendations, evaluate, ndcg_at_k,
                           popularity_recommendations, precision_at_k, recall_at_k,
                           temporal_split)


def test_binary_metrics_and_ideal_normalization():
    recommendations, relevant = [1, 9, 2], [1, 2, 3, 4]
    assert precision_at_k(recommendations, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recommendations, relevant, 5) == pytest.approx(2 / 4)
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, 5))
    assert ndcg_at_k(recommendations, relevant, 5) == pytest.approx(1.5 / ideal)
    assert ndcg_at_k([1], [1], 10) == 1


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
def test_empty_and_cutoff(metric):
    assert metric([], [1], 5) == 0
    assert metric([1], [], 5) == 0
    assert metric([], [], 1) == 0
    assert metric([9, 1], [1], 1) == 0


def test_duplicates_get_no_extra_credit_and_keep_slots():
    assert precision_at_k([1, 1, 2], [1, 1, 2], 2) == 0.5
    assert recall_at_k([1, 1, 2], [1, 1, 2], 3) == 1
    assert ndcg_at_k([1, 1, 2], [1, 2], 3) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
@pytest.mark.parametrize("k", [0, -1, 2.5, True, "5", None])
def test_invalid_k(metric, k):
    with pytest.raises(ValueError, match="k"):
        metric([], [], k)


@pytest.fixture
def dataset():
    movies = pd.DataFrame({"movieId": range(1, 11), "title": [f"Film {i}" for i in range(1, 11)],
                           "genres": ["Drama|Comedy"] * 10})
    ratings = pd.DataFrame([(uid, mid, 4.0, pos + uid * 100)
                            for uid, ids in [(1, range(1, 9)), (2, range(3, 11))]
                            for pos, mid in enumerate(ids)],
                           columns=["userId", "movieId", "rating", "timestamp"])
    return movies, ratings


def test_split_strict_temporal_and_reproducible(dataset):
    _, ratings = dataset
    train, test = temporal_split(ratings)
    assert len(train) == 12 and len(test) == 4
    for uid in (1, 2):
        assert train.loc[train.userId == uid, "timestamp"].max() < test.loc[test.userId == uid, "timestamp"].min()
    assert not set(zip(train.userId, train.movieId)) & set(zip(test.userId, test.movieId))
    shuffled = temporal_split(ratings.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(train, shuffled[0])
    pd.testing.assert_frame_equal(test, shuffled[1])


def test_timestamp_ties_insufficient_users_and_duplicate_events(dataset):
    _, ratings = dataset
    # A later repeat must not move a training pair into the test set.
    repeated = ratings.iloc[:1].assign(timestamp=9999, rating=5.0)
    too_short = pd.DataFrame([(3, 1, 4.0, 0), (3, 2, 4.0, 1)], columns=ratings.columns)
    all_tied = ratings[ratings.userId == 1].assign(userId=4, timestamp=100)
    train, test = temporal_split(pd.concat([ratings, repeated, too_short, all_tied]))
    assert set(train.userId) == {1, 2}
    assert not ((test.userId == 1) & (test.movieId == 1)).any()
    tied = ratings.copy()
    tied.loc[(tied.userId == 1) & tied.movieId.isin([6, 7]), "timestamp"] = 106
    train, test = temporal_split(tied)
    assert train.loc[train.userId == 1, "timestamp"].max() < test.loc[test.userId == 1, "timestamp"].min()


def test_aggregation_full_pool_seen_filter_ties_and_deduplication():
    class Model:
        def recommend(self, seed, top_n):
            assert top_n == 100
            scores = {1: [1, .9, .4, .1], 2: [1, .1, .6, .9]}[seed]
            return pd.DataFrame({"movieId": [1, 2, 3, 4], "similarity_score": scores})

    assert aggregate_recommendations(Model(), [1, 2, 1], {1, 2}, 100, 2) == [3, 4]
    assert aggregate_recommendations(Model(), [], {1}, 100, 2) == []


def test_popularity_only_train_excludes_seen_and_breaks_ties(dataset):
    _, ratings = dataset
    train, _ = temporal_split(ratings)
    result = popularity_recommendations(train, {3, 4, 5, 6}, 10)
    assert result == [1, 2, 7, 8]
    assert 9 not in result and 10 not in result


def test_models_receive_only_train_and_same_catalog(dataset, monkeypatch):
    movies, ratings = dataset
    train, test = temporal_split(ratings)
    real_cf, real_cb = evaluation.ItemBasedCollaborativeRecommender, evaluation.ContentBasedRecommender
    calls = []

    def cf(catalog, fit_ratings, **kwargs):
        pd.testing.assert_frame_equal(fit_ratings, train)
        assert not set(zip(fit_ratings.userId, fit_ratings.movieId)) & set(zip(test.userId, test.movieId))
        calls.append(set(catalog.movieId))
        return real_cf(catalog, fit_ratings, **kwargs)

    def cb(catalog):
        calls.append(set(catalog.movieId))
        return real_cb(catalog)

    monkeypatch.setattr(evaluation, "ItemBasedCollaborativeRecommender", cf)
    monkeypatch.setattr(evaluation, "ContentBasedRecommender", cb)
    metrics, summary = evaluate(movies, ratings, min_common_users=1)
    assert calls == [set(train.movieId), set(train.movieId)]
    assert summary["evaluated_user_ids"] == [1, 2]
    assert summary["cold_start_relevant_test_items"] == 2
    assert len(metrics) == 6
    assert set(metrics.users) == {2}
    assert (metrics.users_below_k == 2).all()
    assert metrics.precision_at_k.between(0, 1).all()


def test_test_ratings_do_not_change_model_rankings(dataset):
    movies, ratings = dataset
    metrics, _ = evaluate(movies, ratings, min_common_users=1)
    _, test = temporal_split(ratings)
    changed = ratings.copy()
    changed.loc[changed.timestamp.isin(test.timestamp), "rating"] = 5.0
    same, _ = evaluate(movies, changed, min_common_users=1)
    pd.testing.assert_frame_equal(metrics, same)


def test_streaming_equals_per_user_aggregation(dataset):
    movies, ratings = dataset
    train, _ = temporal_split(ratings)
    catalog = movies[movies.movieId.isin(train.movieId)]
    seeds = {uid: group.movieId.tolist() for uid, group in train.groupby("userId")}
    seen = {uid: set(ids) for uid, ids in seeds.items()}
    for model in (evaluation.ContentBasedRecommender(catalog),
                  evaluation.ItemBasedCollaborativeRecommender(catalog, train, min_common_users=1)):
        actual = evaluation._stream_rankings(model, seeds, seen, catalog.movieId, 10)
        assert actual == {uid: aggregate_recommendations(model, ids, seen[uid], len(catalog), 10)
                          for uid, ids in seeds.items()}


def test_user_eligibility_shared_and_no_fallback(dataset):
    movies, ratings = dataset
    ratings.loc[ratings.userId == 2, "rating"] = 3
    metrics, summary = evaluate(movies, ratings, min_common_users=999)
    assert summary["evaluated_user_ids"] == [1]
    cf = metrics[metrics.method == "item_based"]
    assert (cf.users_below_k == 1).all()
    assert (cf.user_coverage == 0).all()
    assert (cf.precision_at_k == 0).all()


def test_no_evaluable_users(dataset):
    movies, ratings = dataset
    with pytest.raises(ValueError, match="No users"):
        evaluate(movies, ratings.assign(rating=3.0))
