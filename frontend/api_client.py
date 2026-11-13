"""Small HTTP-only client; importing it never initializes models."""
import os

import httpx
from pydantic import TypeAdapter, ValidationError

from backend.schemas import MovieResponse, RecommendationQuery, RecommendationsResponse


class APIError(Exception):
    """An error safe to display to the user."""


class MovieMatchClient:
    def __init__(self, base_url=None, transport=None):
        self.base_url = (base_url or os.getenv("MOVIEMATCH_API_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.transport = transport

    def _get(self, path, params, schema):
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0), transport=self.transport) as client:
                response = client.get(f"{self.base_url}{path}", params=params)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            messages = {404: "Фильм или маршрут не найден. Повторите поиск.",
                        422: "API отклонил параметры. Проверьте настройки рекомендаций."}
            raise APIError(messages.get(exc.response.status_code,
                           f"Ошибка API (HTTP {exc.response.status_code}). Попробуйте позже.")) from exc
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise APIError("Не удалось связаться с API. Проверьте адрес и запуск FastAPI и повторите попытку.") from exc
        try:
            validated = TypeAdapter(schema).validate_python(response.json())
            return ([movie.model_dump() for movie in validated]
                    if isinstance(validated, list) else validated.model_dump())
        except (ValueError, ValidationError) as exc:
            raise APIError("API вернул неожиданный ответ. Попробуйте позже.") from exc

    def search_movies(self, query, limit=50):
        if not query.strip():
            return []
        return self._get("/movies", {"query": query.strip(), "limit": limit}, list[MovieResponse])

    def recommend(self, movie_id, method="content_based", top_n=10, alpha=1.0):
        parameters = RecommendationQuery(method=method, top_n=top_n, alpha=alpha)
        params = parameters.model_dump(exclude={"alpha"} if method != "hybrid" else set())
        result = self._get(f"/recommendations/{movie_id}", params, RecommendationsResponse)
        if (result["movie"]["movieId"] != movie_id or result["method"] != method
                or result["parameters"]["top_n"] != top_n
                or result["parameters"]["alpha"] != (alpha if method == "hybrid" else None)):
            raise APIError("API вернул ответ для другого фильма или параметров. Повторите запрос.")
        result["recommendations"] = [m for m in result["recommendations"] if m["movieId"] != movie_id]
        return result
