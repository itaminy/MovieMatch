"""MovieMatch recommendations and analytics; no local models at UI runtime."""
from pathlib import Path
import sys

# Streamlit executes this file with frontend/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from frontend.api_client import APIError, MovieMatchClient
from frontend.display import METHODS, genres_label, movie_label, recommendation_rows, request_key, sync_results


def main():
    st.set_page_config(page_title="MovieMatch", page_icon="🎬", layout="wide")
    st.title("MovieMatch")
    page = st.sidebar.radio("Раздел", ["Рекомендации", "Аналитика"], key="page")
    if page == "Аналитика":
        from frontend.analytics import render_analytics
        render_analytics()
        return
    st.write("Найдите похожие фильмы по жанрам и предпочтениям зрителей.")
    st.caption("1. Найдите фильм → 2. Настройте алгоритм → 3. Получите рекомендации")
    client = MovieMatchClient()
    with st.sidebar:
        st.header("Настройки рекомендаций")
        method = st.selectbox("Алгоритм", list(METHODS), format_func=METHODS.get)
        top_n = st.slider("Количество рекомендаций", 1, 50, 10)
        alpha = 1.0
        if method == "hybrid":
            alpha = st.slider("alpha — вес Item-Based", 0.0, 1.0, 1.0, 0.05)
            st.caption("0 — только сходство жанров (Content-Based), 1 — только оценки зрителей (Item-Based). Промежуточные значения смешивают оба подхода.")
        st.caption(f"API: {client.base_url}")
    st.header("1. Найдите фильм")
    query = st.text_input("Название фильма", placeholder="Например, Toy Story")
    if st.session_state.get("search_query") != query.strip():
        st.session_state["search_query"] = query.strip()
        st.session_state.pop("movies", None)
        st.session_state.pop("selected_movie", None)
    if st.button("Найти", type="primary"):
        if not query.strip():
            st.info("Введите название фильма для поиска.")
        else:
            try:
                with st.spinner("Ищем фильмы…"):
                    st.session_state["movies"] = client.search_movies(query)
                st.session_state.pop("selected_movie", None)
            except APIError as exc:
                st.session_state.pop("movies", None)
                st.error(str(exc))
    movies = st.session_state.get("movies", [])
    selected = None
    if movies:
        st.caption(f"Найдено: {len(movies)} (показано не более 50). Уточните название, если нужного фильма нет.")
        by_id = {movie["movieId"]: movie for movie in movies}
        movie_id = st.selectbox("Выберите фильм", list(by_id), key="selected_movie",
                                format_func=lambda mid: movie_label(by_id[mid]))
        selected = by_id[movie_id]
        st.caption(genres_label(selected["genres"]))
    elif "movies" in st.session_state:
        st.info("Фильмы не найдены. Попробуйте другое название.")
    elif not query.strip():
        st.info("Введите название и нажмите «Найти».")
    key = request_key(client.base_url, selected["movieId"] if selected else None, method, top_n, alpha)
    sync_results(st.session_state, key)
    st.header("2. Получите рекомендации")
    if st.button("Получить рекомендации", disabled=selected is None, type="primary"):
        st.session_state.pop("result", None)
        try:
            with st.spinner("Подбираем похожие фильмы…"):
                st.session_state["result"] = client.recommend(selected["movieId"], method, top_n, alpha)
        except APIError as exc:
            st.error(str(exc))
    result = st.session_state.get("result")
    if result is not None:
        st.subheader(f"Рекомендации для: {movie_label(result['movie'])}")
        p = result["parameters"]
        caption = f"Алгоритм: {METHODS[result['method']]} · Запрошено: {p['top_n']}"
        if p["alpha"] is not None:
            caption += f" · alpha: {p['alpha']:.2f}"
        if p["min_common_users"] is not None:
            caption += f" · Минимум общих зрителей: {p['min_common_users']}"
        st.caption(caption)
        rows = recommendation_rows(result)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"similarity_score": st.column_config.NumberColumn("similarity_score", format="%.4f")})
            st.caption("similarity_score: сходство от 0 до 1. Чем выше значение, тем ближе фильм к исходному.")
        else:
            st.info("Подходящие рекомендации не найдены. Попробуйте другой алгоритм или фильм.")


if __name__ == "__main__":
    main()
