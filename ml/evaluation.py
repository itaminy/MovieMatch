"""Deterministic temporal offline evaluation; no test ratings enter model fitting."""

from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd

from ml.collaborative import ItemBasedCollaborativeRecommender
from ml.content_based import ContentBasedRecommender
from ml.hybrid import combine_scores, validate_alpha


def _validate_k(k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")


def _gains(recommendations, relevant, k):
    _validate_k(k)
    relevant = set(relevant)
    seen = set()
    gains = []
    for item in list(recommendations)[:k]:
        gains.append(int(item in relevant and item not in seen))
        seen.add(item)
    return np.asarray(gains, dtype=float), relevant


def precision_at_k(recommendations, relevant, k=10) -> float:
    """Hits / K; missing slots and repeated recommendations receive no credit."""
    gains, _ = _gains(recommendations, relevant, k)
    return float(gains.sum() / k)


def recall_at_k(recommendations, relevant, k=10) -> float:
    gains, relevant = _gains(recommendations, relevant, k)
    return float(gains.sum() / len(relevant)) if relevant else 0.0


def ndcg_at_k(recommendations, relevant, k=10) -> float:
    """Binary DCG / ideal DCG for min(K, number of distinct relevant items)."""
    gains, relevant = _gains(recommendations, relevant, k)
    if not relevant:
        return 0.0
    dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    return float(dcg / ideal)


def temporal_split(ratings, test_fraction=0.2, min_train=5, min_test=1):
    """Hold out latest timestamp groups per user with strictly earlier train.

    Incomplete rows are removed; repeated user/movie events keep the earliest
    event (lowest rating breaks simultaneous conflicts). No pair spans splits.
    Users failing minimum histories after timestamp grouping are excluded.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    _validate_k(min_train)
    _validate_k(min_test)
    columns = ["userId", "movieId", "rating", "timestamp"]
    if not set(columns).issubset(ratings.columns):
        raise ValueError("ratings requires userId, movieId, rating, timestamp")
    clean = ratings.dropna(subset=columns).copy()
    clean = clean.sort_values(columns[:1] + ["timestamp", "movieId", "rating"], kind="stable")
    clean = clean.drop_duplicates(["userId", "movieId"], keep="first")
    train_parts, test_parts = [], []
    for _, group in clean.groupby("userId", sort=True):
        boundary = max(0, min(len(group) - 1, int(np.floor(len(group) * (1 - test_fraction)))))
        cutoff = group.iloc[boundary]["timestamp"]
        train, test = group[group["timestamp"] < cutoff], group[group["timestamp"] >= cutoff]
        if len(train) >= min_train and len(test) >= min_test:
            train_parts.append(train)
            test_parts.append(test)
    empty = clean.iloc[:0].copy()
    train = pd.concat(train_parts, ignore_index=True) if train_parts else empty.copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy()
    return train, test


def aggregate_recommendations(model, seed_movie_ids, seen_movie_ids, catalog_size, k=10):
    """Mean similarity over all positive seeds; full pool before seen filtering.

    No fallback is added. Zero-score neighbors do not enter the final list.
    """
    _validate_k(k)
    seeds = sorted(set(seed_movie_ids))
    seen = set(seen_movie_ids)
    totals = {}
    for seed in seeds:
        neighbors = model.recommend(seed, top_n=max(1, catalog_size))
        for row in neighbors.itertuples():
            if row.movieId not in seen and row.similarity_score > 0:
                totals[row.movieId] = totals.get(row.movieId, 0.0) + row.similarity_score
    return sorted(totals, key=lambda mid: (-totals[mid], mid))[:k]


def popularity_recommendations(train, seen_movie_ids, k=10):
    _validate_k(k)
    counts = train.groupby("movieId")["userId"].nunique().rename("count").reset_index()
    counts = counts[~counts["movieId"].isin(set(seen_movie_ids))]
    return counts.sort_values(["count", "movieId"], ascending=[False, True])["movieId"].head(k).tolist()


def _stream_rankings(model, seeds_by_user, seen_by_user, catalog_ids, k, progress=None,
                     return_scores=False):
    """Stream one seed's neighbors once, accumulating user/item scores.

    This holds U×M score totals, not an M×M item similarity matrix. Neighbor
    rows are discarded after each seed and never cached across the catalog.
    """
    users = sorted(seeds_by_user)
    ids = np.asarray(sorted(catalog_ids), dtype=np.int64)
    positions = {int(mid): pos for pos, mid in enumerate(ids)}
    totals = np.zeros((len(users), len(ids)), dtype=np.float64)
    seed_users = {}
    for user_pos, uid in enumerate(users):
        for seed in sorted(set(seeds_by_user[uid])):
            seed_users.setdefault(seed, []).append(user_pos)
    for step, seed in enumerate(sorted(seed_users), 1):
        neighbors = model.recommend(seed, top_n=max(1, len(ids)))
        neighbors = neighbors[neighbors["similarity_score"] > 0]
        cols = np.array([positions[int(mid)] for mid in neighbors["movieId"]], dtype=int)
        scores = neighbors["similarity_score"].to_numpy(dtype=float)
        for user_pos in seed_users[seed]:
            totals[user_pos, cols] += scores
        if progress and (step % 500 == 0 or step == len(seed_users)):
            progress(f"{type(model).__name__}: {step}/{len(seed_users)} seeds")
    # Mask viewed films before either individual or weighted ranking.
    for user_pos, uid in enumerate(users):
        for mid in seen_by_user[uid]:
            if mid in positions:
                totals[user_pos, positions[mid]] = 0
    if return_scores:
        return users, ids, totals
    return _rank_totals(users, ids, totals, k)


def _rank_totals(users, ids, totals, k):
    rankings = {}
    for user_pos, uid in enumerate(users):
        scores = totals[user_pos]
        candidates = np.flatnonzero(scores > 0)
        order = np.lexsort((ids[candidates], -scores[candidates]))
        rankings[uid] = ids[candidates[order[:k]]].tolist()
    return rankings


def _metric_rows(name, rankings, relevant, users, catalog_size, ks):
    rows = []
    for k in ks:
        recommended = set().union(*(set(rankings[uid][:k]) for uid in users))
        row = {"method": name, "k": int(k), "users": len(users)}
        for metric, function in [("precision", precision_at_k), ("recall", recall_at_k), ("ndcg", ndcg_at_k)]:
            row[f"{metric}_at_k"] = float(np.mean([function(rankings[uid], relevant[uid], k) for uid in users]))
        row.update({"catalog_coverage": len(recommended) / catalog_size,
                    "user_coverage": sum(bool(rankings[uid][:k]) for uid in users) / len(users),
                    "users_below_k": sum(len(rankings[uid]) < k for uid in users),
                    "mean_list_length": float(np.mean([min(k, len(rankings[uid])) for uid in users]))})
        rows.append(row)
    return rows


def tune_alpha(movies, train, *, alphas=(0, .25, .5, .75, 1), min_train=5,
               min_test=1, min_common_users=3, progress=None):
    """Accept ONLY outer train. Pick alpha on a second temporal 80/20 split.

    No evaluable validation users is an error, never a test-based fallback.
    Exact NDCG@10 ties choose the smaller alpha deterministically.
    """
    started = perf_counter()
    alphas = list(alphas)
    if not alphas:
        raise ValueError("alphas must not be empty")
    for alpha in alphas:
        validate_alpha(alpha)
    alphas = sorted(set(float(alpha) for alpha in alphas))
    fit, validation = temporal_split(train, min_train=min_train, min_test=min_test)
    catalog = movies[movies["movieId"].isin(fit["movieId"])].sort_values("movieId")
    seeds = {int(uid): sorted(set(group.movieId)) for uid, group in fit[fit.rating >= 4].groupby("userId")}
    relevant = {int(uid): set(group.movieId) for uid, group in validation[validation.rating >= 4].groupby("userId")}
    users = sorted(set(seeds) & set(relevant))
    if not users:
        raise ValueError("No evaluable validation users; alpha cannot be selected without validation")
    seeds = {uid: seeds[uid] for uid in users}
    seen = {int(uid): set(group.movieId) for uid, group in fit.groupby("userId") if uid in seeds}
    cb = ContentBasedRecommender(catalog)
    cf = ItemBasedCollaborativeRecommender(catalog, fit, min_common_users=min_common_users)
    _, ids, cb_scores = _stream_rankings(cb, seeds, seen, catalog.movieId, 10, progress, return_scores=True)
    _, _, cf_scores = _stream_rankings(cf, seeds, seen, catalog.movieId, 10, progress, return_scores=True)
    rows = []
    for alpha in alphas:
        rankings = _rank_totals(users, ids, combine_scores(cb_scores, cf_scores, alpha), 10)
        for row in _metric_rows("hybrid", rankings, relevant, users, len(catalog), (5, 10)):
            row["alpha"] = alpha
            rows.append(row)
    best = min((row for row in rows if row["k"] == 10),
               key=lambda row: (-row["ndcg_at_k"], row["alpha"]))
    return best["alpha"], {"selected_alpha": best["alpha"], "selection_metric": "ndcg_at_10",
        "tie_break": "smallest alpha", "validation_fraction": .2,
        "fit_rows": len(fit), "validation_rows": len(validation), "users": len(users),
        "evaluated_user_ids": users, "catalog_size": len(catalog), "metrics": rows,
        "elapsed_seconds": perf_counter() - started}


def evaluate(movies, ratings, *, test_fraction=0.2, min_train=5, min_test=1,
             min_common_users=3, ks=(5, 10), progress=None, include_hybrid=False,
             alphas=(0, .25, .5, .75, 1)):
    """Fit both models and a popularity baseline on exactly the same train."""
    started = perf_counter()
    ks = tuple(sorted(set(ks)))
    if not ks:
        raise ValueError("ks must not be empty")
    for k in ks:
        _validate_k(k)
    train, test = temporal_split(ratings, test_fraction, min_train, min_test)
    tuning = None
    if include_hybrid:
        selected_alpha, tuning = tune_alpha(movies, train, alphas=alphas, min_train=min_train,
            min_test=min_test, min_common_users=min_common_users, progress=progress)
    if movies["movieId"].isna().any() or movies["movieId"].duplicated().any():
        raise ValueError("movies must have non-null unique movieId values")
    known = set(movies["movieId"])
    if not set(train["movieId"]).union(test["movieId"]).issubset(known):
        raise ValueError("Ratings reference unknown movies")
    catalog = movies[movies["movieId"].isin(train["movieId"])].sort_values("movieId").copy()
    seeds = {int(uid): sorted(set(group["movieId"])) for uid, group in
             train[train["rating"] >= 4].groupby("userId")}
    relevant = {int(uid): set(group["movieId"]) for uid, group in
                test[test["rating"] >= 4].groupby("userId")}
    users = sorted(set(seeds) & set(relevant))
    if not users:
        raise ValueError("No users with sufficient history, positive train seeds and relevant test films")
    seeds = {uid: seeds[uid] for uid in users}
    seen = {int(uid): set(group["movieId"]) for uid, group in train.groupby("userId") if uid in seeds}
    models = {
        "content_based": ContentBasedRecommender(catalog),
        "item_based": ItemBasedCollaborativeRecommender(catalog, train, min_common_users=min_common_users),
    }
    rows, timings, component_scores = [], {}, {}
    for name in [*models, "popularity", *(["hybrid"] if include_hybrid else [])]:
        model_started = perf_counter()
        if name == "popularity":
            rankings = {uid: popularity_recommendations(train, seen[uid], max(ks)) for uid in users}
        elif name == "hybrid":
            scores = combine_scores(component_scores["content_based"], component_scores["item_based"], selected_alpha)
            rankings = _rank_totals(users, ids, scores, max(ks))
        elif include_hybrid:
            _, ids, scores = _stream_rankings(models[name], seeds, seen, catalog["movieId"],
                                             max(ks), progress, return_scores=True)
            component_scores[name] = scores
            rankings = _rank_totals(users, ids, scores, max(ks))
        else:
            rankings = _stream_rankings(models[name], seeds, seen, catalog["movieId"], max(ks), progress)
        timings[name] = perf_counter() - model_started
        rows.extend(_metric_rows(name, rankings, relevant, users, len(catalog), ks))
    summary = {
        "protocol": {"test_fraction": test_fraction, "min_train": min_train, "min_test": min_test,
                     "min_common_users": min_common_users, "relevance_threshold": 4.0,
                     "ks": [int(k) for k in ks], "random_state": 42, "randomness_used": False,
                     "aggregation": "mean positive similarity over all positive train seeds; ties by movieId",
                     "split": "per-user strict timestamp boundary; earliest event per user/movie pair"},
        "input_rows": len(ratings), "train_rows": len(train), "test_rows": len(test),
        "split_users": int(train["userId"].nunique()), "evaluated_users": len(users),
        "excluded_history_users": int(ratings["userId"].nunique() - train["userId"].nunique()),
        "excluded_no_positive_train_or_test": int(train["userId"].nunique() - len(users)),
        "catalog_size": len(catalog), "evaluated_user_ids": users,
        "cold_start_relevant_test_items": sum(len(relevant[uid] - set(catalog["movieId"])) for uid in users),
        "model_ranking_seconds": timings, "elapsed_seconds": perf_counter() - started,
        "metrics": rows,
    }
    if tuning is not None:
        summary["validation"] = tuning
        summary["protocol"]["selected_alpha"] = selected_alpha
    return pd.DataFrame(rows), summary
