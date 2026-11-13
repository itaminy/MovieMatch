"""Presentation and result-state helpers, independent of Streamlit."""
METHODS = {"content_based": "Content-Based", "item_based": "Item-Based", "hybrid": "Hybrid"}


def movie_label(movie):
    return f"{movie['title'] or 'Без названия'} · ID {movie['movieId']}"


def genres_label(genres):
    return genres.replace("|", ", ") if genres and genres != "(no genres listed)" else "Жанры не указаны"


def request_key(api_url, movie_id, method, top_n, alpha):
    return (api_url, movie_id, method, top_n, alpha if method == "hybrid" else None)


def sync_results(state, key):
    """Keep results on rerun, discard them as soon as inputs change."""
    if state.get("result_key") != key:
        state.pop("result", None)
        state["result_key"] = key


def recommendation_rows(result):
    return [{"Название": m["title"] or "Без названия", "Жанры": genres_label(m["genres"]),
             "similarity_score": m["similarity_score"], "movieId": m["movieId"]}
            for m in result["recommendations"] if m["movieId"] != result["movie"]["movieId"]]
