"""Fast HTTP tests with real models on a tiny in-memory dataset."""

import importlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main
import backend.service as service_module
from backend.main import create_app
from backend.service import RecommendationService


@pytest.fixture
def data():
    movies = pd.DataFrame({
        "movieId": [30, 10, 50, 20, 40, 60],
        "title": ["A TOY adventure", "Toy Story (1995)", "Unrated", "Jumanji (1995)", None, "Film [1995]"],
        "genres": ["Adventure|Comedy", "Adventure|Comedy", "Drama", "Adventure", None, "Drama"],
    })
    ratings = pd.DataFrame([(mid, uid, 4.) for mid in (10, 20, 30, 40) for uid in (1, 2, 3)],
                           columns=["movieId", "userId", "rating"])
    return movies, ratings


@pytest.fixture
def client(data):
    with TestClient(create_app(lambda: RecommendationService(*data))) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_deterministic_limit_case_and_literal_query(client):
    assert [row["movieId"] for row in client.get("/movies").json()] == [10, 20, 30, 40, 50, 60]
    assert len(client.get("/movies?limit=2").json()) == 2
    upper = client.get("/movies", params={"query": "TOY"}).json()
    lower = client.get("/movies", params={"query": "toy"}).json()
    assert upper == lower and [row["movieId"] for row in upper] == [10, 30]
    assert len(client.get("/movies", params={"query": "[1995]"}).json()) == 1
    assert client.get("/movies?query=does-not-exist").json() == []
    assert client.get("/movies?query=").json() == client.get("/movies").json()


def test_get_movie_and_nullable_metadata(client):
    response = client.get("/movies/10")
    assert response.status_code == 200
    assert response.json() == {"movieId": 10, "title": "Toy Story (1995)", "genres": "Adventure|Comedy"}
    assert client.get("/movies/40").json() == {"movieId": 40, "title": None, "genres": None}


@pytest.mark.parametrize("path", ["/movies/999", "/recommendations/999"])
def test_unknown_movie_404(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize("method", ["content_based", "item_based", "hybrid"])
def test_recommendations_json_and_source_excluded(client, method):
    response = client.get("/recommendations/10", params={"method": method, "top_n": 2})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"movie", "method", "parameters", "recommendations"}
    assert payload["movie"]["movieId"] == 10 and payload["method"] == method
    assert payload["parameters"]["top_n"] == 2
    assert payload["parameters"]["alpha"] == (1. if method == "hybrid" else None)
    assert len(payload["recommendations"]) == 2
    for row in payload["recommendations"]:
        assert set(row) == {"movieId", "title", "genres", "similarity_score"}
        assert row["movieId"] != 10
        assert 0 <= row["similarity_score"] <= 1


def test_defaults_and_alpha_endpoints(client):
    default = client.get("/recommendations/10").json()
    assert default["method"] == "content_based" and default["parameters"]["top_n"] == 10
    cf = client.get("/recommendations/10?method=item_based").json()["recommendations"]
    hybrid = client.get("/recommendations/10?method=hybrid").json()
    assert hybrid["parameters"]["alpha"] == 1.0
    assert hybrid["recommendations"] == cf
    cb = client.get("/recommendations/10?method=hybrid&alpha=0").json()
    assert cb["recommendations"] == default["recommendations"]
    assert client.get("/recommendations/10?method=hybrid&alpha=0.5").status_code == 200
    ignored = client.get("/recommendations/10?method=item_based&alpha=0").json()
    assert ignored["recommendations"] == cf and ignored["parameters"]["alpha"] is None


def test_empty_recommendations_are_valid(client):
    response = client.get("/recommendations/50?method=item_based")
    assert response.status_code == 200 and response.json()["recommendations"] == []
    assert client.get("/recommendations/40?method=content_based").json()["recommendations"] == []
    assert client.get("/recommendations/10?top_n=50").status_code == 200


@pytest.mark.parametrize("path", [
    "/movies?limit=0", "/movies?limit=101", "/movies?limit=abc",
    "/movies/abc", "/movies/0", "/movies/-1", "/recommendations/abc",
    "/recommendations/0", "/recommendations/10?method=unknown",
    "/recommendations/10?top_n=0", "/recommendations/10?top_n=51",
    "/recommendations/10?top_n=-1", "/recommendations/10?top_n=1.5",
    "/recommendations/10?alpha=-0.1", "/recommendations/10?alpha=1.1",
    "/recommendations/10?alpha=nan", "/recommendations/10?alpha=inf",
    "/recommendations/10?alpha=text",
])
def test_invalid_parameters_422(client, path):
    response = client.get(path)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_one_load_and_model_initialization_per_lifespan(data, monkeypatch):
    loads, fits = [], []
    original = service_module.HybridRecommender
    def loader(path):
        loads.append(path)
        return dict(zip(["movies", "ratings"], data))
    def hybrid(*args, **kwargs):
        fits.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service_module, "load_data", loader)
    monkeypatch.setattr(service_module, "HybridRecommender", hybrid)
    app = create_app()
    assert not loads and not fits
    with TestClient(app) as client:
        for alpha in (0, .25, .5, .75, 1):
            assert client.get(f"/recommendations/10?method=hybrid&alpha={alpha}").status_code == 200
        assert client.get("/movies").status_code == 200
        assert len(loads) == len(fits) == 1
    assert not hasattr(app.state, "recommendations")


def test_missing_files_fail_startup_with_instruction(monkeypatch):
    def loader(path):
        raise FileNotFoundError("Missing movies.csv")
    monkeypatch.setattr(service_module, "load_data", loader)
    with pytest.raises(RuntimeError, match="MovieMatch startup failed.*movies.csv.*README"):
        with TestClient(create_app()):
            pass


def test_import_does_not_load_data(monkeypatch):
    def forbidden(path):
        raise AssertionError("Data loaded during import")
    monkeypatch.setattr(service_module, "load_data", forbidden)
    importlib.reload(main)


def test_concurrent_hybrid_weights_do_not_mutate_shared_model(data):
    service = RecommendationService(*data)
    alphas = [0., .25, .5, 1.] * 4
    expected = {alpha: service.recommend(10, "hybrid", alpha=alpha) for alpha in set(alphas)}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda alpha: service.recommend(10, "hybrid", alpha=alpha), alphas))
    assert results == [expected[alpha] for alpha in alphas]
    assert service._hybrid.alpha == 1.


def test_openapi_documents_routes(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/movies", "/movies/{movie_id}", "/recommendations/{movie_id}"} <= set(schema["paths"])
    assert client.get("/docs").status_code == 200
