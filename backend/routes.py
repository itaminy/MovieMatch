"""HTTP routing; model and data logic live in the service."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from backend.schemas import (HealthResponse, MovieResponse, RecommendationQuery,
                             RecommendationsResponse)
from backend.service import RecommendationService

router = APIRouter()
MovieId = Annotated[int, Path(ge=1)]


def get_service(request: Request) -> RecommendationService:
    return request.app.state.recommendations


Service = Annotated[RecommendationService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@router.get("/movies", response_model=list[MovieResponse])
def movies(service: Service, query: str | None = None,
           limit: Annotated[int, Query(ge=1, le=100)] = 20):
    return service.search_movies(query, limit)


@router.get("/movies/{movie_id}", response_model=MovieResponse)
def movie(movie_id: MovieId, service: Service):
    try:
        return service.get_movie(movie_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found") from None


@router.get("/recommendations/{movie_id}", response_model=RecommendationsResponse)
def recommendations(movie_id: MovieId, parameters: Annotated[RecommendationQuery, Query()],
                    service: Service):
    try:
        return service.recommend(movie_id, parameters.method, parameters.top_n, parameters.alpha)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found") from None
