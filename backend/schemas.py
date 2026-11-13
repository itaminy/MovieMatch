"""Validated HTTP query parameters and JSON response contracts."""

from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["content_based", "item_based", "hybrid"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class MovieResponse(BaseModel):
    movieId: int
    title: str | None
    genres: str | None


class RecommendationItem(MovieResponse):
    similarity_score: float = Field(ge=0, le=1, allow_inf_nan=False)


class RecommendationQuery(BaseModel):
    method: Method = "content_based"
    top_n: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)


class RecommendationParameters(BaseModel):
    top_n: int
    alpha: float | None
    min_common_users: int | None


class RecommendationsResponse(BaseModel):
    movie: MovieResponse
    method: Method
    parameters: RecommendationParameters
    recommendations: list[RecommendationItem]
