"""Validated, small dashboard snapshot and chart preparation. No ML imports."""
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing import Literal

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "dashboard_data" / "snapshot.json"
MODEL_NAMES = {"content_based": "Content-Based", "item_based": "Item-Based Collaborative Filtering",
               "popularity": "Popularity", "hybrid": "Hybrid"}
METRICS = {"precision_at_k": "Precision", "recall_at_k": "Recall", "ndcg_at_k": "NDCG",
           "catalog_coverage": "Покрытие каталога"}


class DashboardDataError(ValueError):
    pass


class Checked(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Counts(Checked):
    movies: int = Field(ge=0, strict=True)
    users: int = Field(ge=0, strict=True)
    ratings: int = Field(ge=0, strict=True)
    valid_rating_values: int = Field(ge=0, strict=True)


class TopMovie(Checked):
    movieId: int
    title: str | None
    rating_count: int = Field(ge=0)


class Overview(Checked):
    counts: Counts
    mean_rating: float | None = Field(ge=0.5, le=5)
    rating_distribution: dict[str, int]
    genres: dict[str, int]
    top_10_movies: list[TopMovie] = Field(max_length=10)

    @model_validator(mode="after")
    def consistent(self):
        values = self.rating_distribution
        if any(float(k) not in [i / 2 for i in range(1, 11)] or v < 0 for k, v in values.items()):
            raise ValueError("Invalid rating distribution")
        if any(v < 0 for v in self.genres.values()):
            raise ValueError("Invalid genre counts")
        count = sum(values.values())
        if count != self.counts.valid_rating_values or count > self.counts.ratings:
            raise ValueError("Inconsistent rating count")
        mean = sum(float(k) * v for k, v in values.items()) / count if count else None
        if (mean is None) != (self.mean_rating is None) or (mean is not None and abs(mean - self.mean_rating) > 1e-6):
            raise ValueError("Inconsistent mean rating")
        return self


class MetricRow(Checked):
    method: Literal["content_based", "item_based", "popularity", "hybrid"]
    k: Literal[5, 10]
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    catalog_coverage: float = Field(ge=0, le=1)


class AlphaRow(Checked):
    alpha: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)


class Snapshot(Checked):
    schema_version: Literal[1]
    source: str
    overview: Overview
    metrics: list[MetricRow]
    validation: list[AlphaRow] = Field(min_length=1)
    selected_alpha: float = Field(ge=0, le=1)
    source_sha256: dict[str, str]

    @model_validator(mode="after")
    def complete(self):
        keys = [(r.method, r.k) for r in self.metrics]
        if len(keys) != 8 or set(keys) != {(m, k) for m in MODEL_NAMES for k in (5, 10)}:
            raise ValueError("Expected four methods at K=5/10")
        if len({r.alpha for r in self.validation}) != len(self.validation):
            raise ValueError("Duplicate alpha")
        best = min(self.validation, key=lambda r: (-r.ndcg_at_10, r.alpha))
        if self.selected_alpha != best.alpha:
            raise ValueError("Selected alpha disagrees with validation")
        return self


def load_snapshot(path=DEFAULT_SNAPSHOT):
    try:
        return Snapshot.model_validate_json(Path(path).read_text(encoding="utf-8")).model_dump()
    except (OSError, ValueError, ValidationError) as exc:
        raise DashboardDataError("Данные аналитики отсутствуют или повреждены. Подготовьте снимок командой python -m frontend.prepare_dashboard. Инструкции — в README.") from exc


def overview_indicators(data):
    overview = data["overview"]
    return {"Фильмы": overview["counts"]["movies"], "Пользователи": overview["counts"]["users"],
            "Оценки": overview["counts"]["ratings"], "Средний рейтинг": overview["mean_rating"]}


def eda_chart_data(data):
    overview = data["overview"]
    ratings = pd.DataFrame([{"Оценка": float(k), "Количество оценок": v}
                            for k, v in overview["rating_distribution"].items()]).sort_values("Оценка") if overview["rating_distribution"] else pd.DataFrame(columns=["Оценка", "Количество оценок"])
    genres = pd.DataFrame(sorted(overview["genres"].items(), key=lambda r: (-r[1], r[0]))[:10],
                          columns=["Жанр", "Количество фильмов"])
    movies = pd.DataFrame([{"Фильм": f"{r['title'] or 'Без названия'} [ID {r['movieId']}]", "Количество оценок": r["rating_count"]}
                           for r in sorted(overview["top_10_movies"], key=lambda r: (-r["rating_count"], r["movieId"]))],
                          columns=["Фильм", "Количество оценок"])
    return ratings, genres, movies


def model_chart_data(data, metric, k):
    if metric not in METRICS or k not in (5, 10):
        raise ValueError("Unknown metric or K")
    return pd.DataFrame([{"Алгоритм": MODEL_NAMES[r["method"]], "Значение": r[metric]}
                         for r in data["metrics"] if r["k"] == k])


def alpha_chart_data(data):
    return pd.DataFrame(data["validation"]).sort_values("alpha").rename(columns={"ndcg_at_10": "NDCG@10"})
