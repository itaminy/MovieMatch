"""Streamlit analytics page. Reads only the compact, cached snapshot."""
import streamlit as st
import plotly.express as px

from frontend.analytics_data import (DEFAULT_SNAPSHOT, METRICS, DashboardDataError, load_snapshot,
                                     overview_indicators, eda_chart_data, model_chart_data, alpha_chart_data)


@st.cache_data(show_spinner=False)
def cached_snapshot(path, modified_ns):
    return load_snapshot(path)


def render_analytics():
    st.header("Аналитика")
    st.write("Обзор MovieLens и результаты offline-сравнения рекомендательных алгоритмов.")
    try:
        data = cached_snapshot(str(DEFAULT_SNAPSHOT), DEFAULT_SNAPSHOT.stat().st_mtime_ns)
    except (OSError, DashboardDataError):
        st.warning("Данные аналитики отсутствуют или повреждены. Подготовьте агрегаты по инструкции в README.")
        st.code("python -m ml.eda\npython -m ml.run_evaluation\npython -m frontend.prepare_dashboard", language="powershell")
        st.caption("Команды выполняются отдельно в терминале. Полная offline evaluation нужна только для создания новых отчётов.")
        return
    st.caption(f"Источник: {data['source']}. Сохранённый снимок эксперимента; не живая статистика API.")
    st.subheader("Обзор датасета")
    columns = st.columns(2)
    for index, (label, value) in enumerate(overview_indicators(data).items()):
        columns[index % 2].metric(label, "Нет данных" if value is None else f"{value:.3f}" if label == "Средний рейтинг" else f"{value:,}".replace(",", " "))
    st.caption("Пользователи — уникальные ID из оценок и тегов. Жанры считаются по уникальным фильмам; один фильм может иметь несколько жанров. Среднее не учитывает пропущенные оценки.")
    ratings, genres, movies = eda_chart_data(data)
    st.plotly_chart(px.bar(ratings, x="Оценка", y="Количество оценок", title="Распределение оценок"), width="stretch")
    st.plotly_chart(px.bar(genres, x="Количество фильмов", y="Жанр", orientation="h", title="Топ-10 жанров").update_layout(yaxis={"autorange": "reversed"}), width="stretch")
    st.plotly_chart(px.bar(movies, x="Количество оценок", y="Фильм", orientation="h", title="Топ-10 фильмов по количеству оценок").update_layout(yaxis={"autorange": "reversed"}, height=550), width="stretch")
    st.subheader("Сравнение моделей на test")
    metric = st.selectbox("Метрика", list(METRICS), format_func=METRICS.get, key="analytics_metric")
    k = st.radio("K — длина списка рекомендаций", [5, 10], horizontal=True, key="analytics_k")
    descriptions = {"precision_at_k": "Precision@K — доля релевантных фильмов среди первых K рекомендаций.",
                    "recall_at_k": "Recall@K — доля найденных релевантных test-фильмов пользователя.",
                    "ndcg_at_k": "NDCG@K — качество ранжирования: попадания выше в списке имеют больший вес.",
                    "catalog_coverage": "Catalog Coverage — доля train-каталога, попавшая хотя бы в один список top-K."}
    st.caption(descriptions[metric])
    comparison = model_chart_data(data, metric, k)
    figure = px.bar(comparison, x="Значение", y="Алгоритм", orientation="h", title=f"{METRICS[metric]} · K={k}")
    if metric == "catalog_coverage":
        figure.update_xaxes(tickformat=".1%")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("Метрики качества усреднены по пользователям. Релевантность: оценка ≥ 4. Разбиение — по времени внутри пользователя; это не общий временной срез. Serving API использует другой режим данных, поэтому его выдача не служит повторной test-оценкой.")
    st.subheader("Выбор alpha на validation")
    alpha = alpha_chart_data(data)
    selected = data["selected_alpha"]
    figure = px.line(alpha, x="alpha", y="NDCG@10", markers=True, title="Влияние alpha на validation NDCG@10")
    figure.add_vline(x=selected, line_dash="dash", annotation_text=f"Выбран alpha={selected:g}")
    st.plotly_chart(figure, width="stretch")
    st.write(f"Протестированные alpha: {', '.join(f'{a:g}' for a in alpha['alpha'])}. Выбрано: **{selected:g}**.")
    st.info("Alpha подбирался только на validation по максимуму NDCG@10 (при равенстве — меньший alpha). Test использовался только для итоговой оценки после обучения на полном train.")
    if selected == 1:
        st.write("Hybrid не превзошёл Item-Based: выбран alpha=1, поэтому вес Content-Based равен нулю. Гибридная модель совпадает с Item-Based; смешивание жанров не улучшило validation NDCG@10. Это результат данного эксперимента, а не универсальное свойство Hybrid.")
    else:
        st.write("Alpha задаёт вес Item-Based: score = alpha × CF + (1 − alpha) × Content-Based. Итоговое сравнение приведено на графике test выше.")
    with st.expander("Воспроизводимость снимка"):
        st.write("Снимок содержит только агрегаты. Для обновления существующих отчётов: python -m frontend.prepare_dashboard. Полная инструкция — в README.")
        st.json(data["source_sha256"])
