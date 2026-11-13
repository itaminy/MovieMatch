"""Read MovieLens CSVs without dropping missing values or duplicate rows."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"
SCHEMAS = {
    "movies": {"movieId": "Int64", "title": "string", "genres": "string"},
    "ratings": {"userId": "Int64", "movieId": "Int64", "rating": "Float64", "timestamp": "Int64"},
    "tags": {"userId": "Int64", "movieId": "Int64", "tag": "string", "timestamp": "Int64"},
    "links": {"movieId": "Int64", "imdbId": "Int64", "tmdbId": "Int64"},
}


def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Validate required columns and non-null values; preserve nulls for EDA.

    IDs must be positive integers, timestamps nonnegative Unix seconds and
    ratings half-star values in [0.5, 5]. Additional columns are preserved.
    Empty fields are nulls; text such as the literal tag 'NA' stays text.
    """
    directory = Path(data_dir)
    missing = [f"{name}.csv" for name in SCHEMAS if not (directory / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files in {directory}: {', '.join(missing)}")
    result = {}
    for name, schema in SCHEMAS.items():
        path = directory / f"{name}.csv"
        try:
            frame = pd.read_csv(path, dtype="string", encoding="utf-8-sig", keep_default_na=False, na_values=[""])
        except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
            raise ValueError(f"{path.name}: cannot read CSV: {exc}") from exc
        absent = set(schema) - set(frame.columns)
        if absent:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(absent))}")
        for column, dtype in schema.items():
            try:
                if dtype == "string":
                    frame[column] = frame[column].mask(frame[column].str.strip().eq(""))
                    continue
                values = pd.to_numeric(frame[column], errors="raise")
                nonnull = values.dropna().to_numpy(dtype=float)
                if not np.isfinite(nonnull).all():
                    raise ValueError("non-finite number")
                if dtype == "Int64":
                    if not np.equal(nonnull, np.floor(nonnull)).all():
                        raise ValueError("expected integers")
                    if (nonnull < (0 if column == "timestamp" else 1)).any():
                        raise ValueError("invalid identifier or timestamp")
                elif not (np.isin(nonnull, np.arange(0.5, 5.5, 0.5))).all():
                    raise ValueError("rating must be between 0.5 and 5 in half-star steps")
                frame[column] = values.astype(dtype)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path.name}: invalid {column}: {exc}") from exc
        result[name] = frame
    return result
