"""MovieMatch API entry point: uvicorn backend.main:app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes import router
from backend.service import RecommendationService


def create_app(service_factory=None) -> FastAPI:
    """Creating/importing an app does not load data or initialize models."""
    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.recommendations = (
                service_factory() if service_factory is not None
                else RecommendationService.from_directory())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"MovieMatch startup failed: {exc}. Place the real MovieLens CSVs in "
                "data/ml-latest-small/; see README.md for download instructions."
            ) from exc
        try:
            yield
        finally:
            del application.state.recommendations

    application = FastAPI(title="MovieMatch", version="0.1.0", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
