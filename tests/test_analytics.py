"""Dashboard tests never load MovieLens or run offline evaluation."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend.analytics_data import (DEFAULT_SNAPSHOT, METRICS, DashboardDataError, load_snapshot,
                                     overview_indicators, eda_chart_data, model_chart_data, alpha_chart_data)
from frontend.prepare_dashboard import prepare


@pytest.fixture
def snapshot():
    return load_snapshot()


def test_committed_snapshot_and_indicators(snapshot):
    assert DEFAULT_SNAPSHOT.stat().st_size < 20000
    assert overview_indicators(snapshot) == {"Фильмы": 9742, "Пользователи": 610,
                                           "Оценки": 100836, "Средний рейтинг": pytest.approx(3.5015569836)}
    assert "userId" not in json.dumps(snapshot)
    assert snapshot["selected_alpha"] == 1


def test_chart_preparation(snapshot):
    ratings, genres, movies = eda_chart_data(snapshot)
    assert ratings["Количество оценок"].sum() == 100836
    assert ratings["Оценка"].tolist() == sorted(ratings["Оценка"])
    assert len(genres) == len(movies) == 10
    assert genres.iloc[0]["Жанр"] == "Drama"
    assert "ID" in movies.iloc[0]["Фильм"]
    alpha = alpha_chart_data(snapshot)
    assert alpha["alpha"].tolist() == [0, .25, .5, .75, 1]
    assert alpha.loc[alpha["NDCG@10"].idxmax(), "alpha"] == snapshot["selected_alpha"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("k", [5, 10])
def test_metric_selection(snapshot, metric, k):
    chart = model_chart_data(snapshot, metric, k)
    assert len(chart) == 4
    assert chart["Значение"].tolist() == [row[metric] for row in snapshot["metrics"] if row["k"] == k]


@pytest.mark.parametrize("metric,k", [("bad", 5), ("ndcg_at_k", 20)])
def test_bad_selection(snapshot, metric, k):
    with pytest.raises(ValueError):
        model_chart_data(snapshot, metric, k)


def test_missing_and_invalid_json(tmp_path):
    path = tmp_path / "snapshot.json"
    with pytest.raises(DashboardDataError):
        load_snapshot(path)
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(DashboardDataError):
        load_snapshot(path)


@pytest.mark.parametrize("change", ["mean", "missing_model", "duplicate", "alpha", "nan", "count", "version"])
def test_invalid_snapshot(snapshot, tmp_path, change):
    data = copy.deepcopy(snapshot)
    if change == "mean":
        data["overview"]["mean_rating"] = 1
    elif change == "missing_model":
        data["metrics"].pop()
    elif change == "duplicate":
        data["validation"].append(data["validation"][0])
    elif change == "alpha":
        data["selected_alpha"] = 0
    elif change == "nan":
        data["metrics"][0]["ndcg_at_k"] = float("nan")
    elif change == "count":
        data["overview"]["counts"]["ratings"] = -1
    else:
        data["schema_version"] = 99
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DashboardDataError):
        load_snapshot(path)


def test_export_from_small_reports(snapshot, tmp_path):
    reports = tmp_path / "reports"
    (reports / "evaluation").mkdir(parents=True)
    (reports / "summary.json").write_text(json.dumps(snapshot["overview"]), encoding="utf-8")
    (reports / "evaluation/summary.json").write_text(json.dumps({"validation": {"selected_alpha": 1}}), encoding="utf-8")
    pd.DataFrame(snapshot["metrics"]).to_csv(reports / "evaluation/metrics.csv", index=False)
    pd.DataFrame([{**r, "method": "hybrid", "k": 10, "ndcg_at_k": r["ndcg_at_10"]}
                  for r in snapshot["validation"]]).to_csv(reports / "evaluation/validation_metrics.csv", index=False)
    output = tmp_path / "output.json"
    prepare(reports, output)
    loaded = load_snapshot(output)
    assert loaded["overview"] == snapshot["overview"]
    assert loaded["metrics"] == snapshot["metrics"]
    assert len(loaded["source_sha256"]) == 4
    with pytest.raises(OSError):
        prepare(tmp_path / "missing", output)


def test_navigation_and_filters():
    app = AppTest.from_file("../frontend/app.py", default_timeout=15).run()
    assert not app.exception
    assert len(app.text_input) == 1
    app.sidebar.radio(key="page").set_value("Аналитика").run()
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.get("plotly_chart")) == 5
    app.selectbox(key="analytics_metric").select("catalog_coverage").run()
    app.radio(key="analytics_k").set_value(10).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 4
    app.sidebar.radio(key="page").set_value("Рекомендации").run()
    assert not app.exception
    assert len(app.text_input) == 1
    assert app.button[1].disabled


def test_missing_data_page(monkeypatch, tmp_path):
    import frontend.analytics as page
    monkeypatch.setattr(page, "DEFAULT_SNAPSHOT", tmp_path / "missing.json")
    app = AppTest.from_file("../frontend/app.py").run()
    app.sidebar.radio(key="page").set_value("Аналитика").run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "prepare_dashboard" in app.code[0].value
