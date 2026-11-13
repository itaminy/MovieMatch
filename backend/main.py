"""Минимальный HTTP API MovieMatch."""

from fastapi import FastAPI

app = FastAPI(title="MovieMatch")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
