"""Frontend tests use mocked HTTP; a running backend is never required."""
import httpx
import pytest
from frontend.api_client import APIError, MovieMatchClient
from frontend.display import movie_label, genres_label, request_key, sync_results, recommendation_rows

MOVIE = {"movieId": 1, "title": "Same", "genres": "Comedy|Drama"}


def payload(method="hybrid", alpha=1.0):
    return {"movie": MOVIE, "method": method,
            "parameters": {"top_n": 10, "alpha": alpha, "min_common_users": 3},
            "recommendations": [{**MOVIE, "similarity_score": 1.0},
                                {**MOVIE, "movieId": 2, "similarity_score": 0.5}]}


def client(handler, base="http://example.test/api/"):
    return MovieMatchClient(base, transport=httpx.MockTransport(handler))


def test_search_url_json_and_timeout():
    def handler(request):
        assert request.url.path == "/api/movies"
        assert dict(request.url.params) == {"query": "Toy & Story", "limit": "50"}
        assert request.extensions["timeout"]["connect"] == 5
        assert request.extensions["timeout"]["read"] == 30
        return httpx.Response(200, json=[MOVIE])
    assert client(handler).search_movies(" Toy & Story ") == [MOVIE]


def test_environment(monkeypatch):
    monkeypatch.setenv("MOVIEMATCH_API_URL", "http://localhost:9999/")
    assert MovieMatchClient().base_url == "http://localhost:9999"
    monkeypatch.delenv("MOVIEMATCH_API_URL")
    assert MovieMatchClient().base_url == "http://127.0.0.1:8001"


def test_empty_search():
    def handler(request):
        pytest.fail("Empty search must not call API")
    assert client(handler).search_movies("  ") == []
    assert client(lambda r: httpx.Response(200, json=[])).search_movies("missing") == []


@pytest.mark.parametrize("method,alpha", [("hybrid", 0), ("hybrid", 0.5), ("hybrid", 1),
                                          ("content_based", None), ("item_based", None)])
def test_recommendation_parameters(method, alpha):
    def handler(request):
        assert request.url.path == "/api/recommendations/1"
        expected = {"method": method, "top_n": "10"}
        if alpha is not None:
            expected["alpha"] = str(float(alpha))
        assert dict(request.url.params) == expected
        return httpx.Response(200, json=payload(method, alpha))
    result = client(handler).recommend(1, method, alpha=alpha if alpha is not None else 1)
    assert [m["movieId"] for m in result["recommendations"]] == [2]


@pytest.mark.parametrize("status,message", [(404, "не найден"), (422, "параметры"), (500, "HTTP 500")])
def test_http_errors(status, message):
    with pytest.raises(APIError, match=message):
        client(lambda r: httpx.Response(status)).search_movies("Toy")


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_connection_errors(error):
    def handler(request):
        raise error("failed", request=request)
    with pytest.raises(APIError, match="связаться"):
        client(handler).search_movies("Toy")


@pytest.mark.parametrize("body", [b"not json", b'{}', b'[{"movieId": "bad"}]'])
def test_unexpected_search_response(body):
    with pytest.raises(APIError, match="неожиданный"):
        client(lambda r: httpx.Response(200, content=body)).search_movies("Toy")


def test_unexpected_recommendation_response():
    response = payload()
    response["movie"] = {**MOVIE, "movieId": 99}
    with pytest.raises(APIError, match="другого фильма"):
        client(lambda r: httpx.Response(200, json=response)).recommend(1, "hybrid")
    response["recommendations"][0]["similarity_score"] = "bad"
    with pytest.raises(APIError, match="неожиданный"):
        client(lambda r: httpx.Response(200, json=response)).recommend(1, "hybrid")


def test_display():
    assert movie_label(MOVIE) != movie_label({**MOVIE, "movieId": 2})
    assert "Без названия" in movie_label({**MOVIE, "title": None})
    assert genres_label(None) == "Жанры не указаны"
    assert genres_label(MOVIE["genres"]) == "Comedy, Drama"
    assert len(recommendation_rows(payload())) == 1


@pytest.mark.parametrize("index,value", [(0, "other-api"), (1, 2), (2, "item_based"), (3, 20), (4, 0.5)])
def test_result_state(index, value):
    args = ["api", 1, "hybrid", 10, 1.0]
    key = request_key(*args)
    state = {}
    sync_results(state, key)
    state["result"] = payload()
    sync_results(state, key)
    assert state["result"] == payload()
    args[index] = value
    sync_results(state, request_key(*args))
    assert "result" not in state


def test_alpha_is_irrelevant_outside_hybrid():
    assert request_key("api", 1, "item_based", 10, 0) == request_key("api", 1, "item_based", 10, 1)


def test_streamlit_flow(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(MovieMatchClient, "search_movies", lambda self, query: [MOVIE, {**MOVIE, "movieId": 2}])
    response = payload("content_based", None)
    monkeypatch.setattr(MovieMatchClient, "recommend", lambda *args: response)
    app = AppTest.from_file("../frontend/app.py").run()
    assert not app.exception
    app.text_input[0].input("Same").run()
    app.button[0].click().run()
    app.button[1].click().run()
    assert len(app.dataframe) == 1
    app.run()
    assert len(app.dataframe) == 1
    app.selectbox(key="selected_movie").select(2).run()
    assert len(app.dataframe) == 0
    app.selectbox(key="selected_movie").select(1).run()
    app.button[1].click().run()
    assert len(app.dataframe) == 1
    app.slider[0].set_value(20).run()
    assert len(app.dataframe) == 0
    app.sidebar.selectbox[0].select("hybrid").run()
    assert len(app.sidebar.slider) == 2 and app.sidebar.slider[1].value == 1.0
    assert not app.exception

