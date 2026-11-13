"""Synthetic fixtures are used only for tests, never as downloaded data."""

import json

import pandas as pd
import pytest

from ml.data_loader import load_data
from ml.eda import analyze, run_eda


@pytest.fixture
def dataset(tmp_path):
    files = {
        "movies": "movieId,title,genres\n1,First,Comedy|Drama\n2,Second,Drama\n3,Third,(no genres listed)\n",
        "ratings": "userId,movieId,rating,timestamp\n1,1,4.5,100\n2,1,3.5,101\n1,2,5.0,102\n",
        "tags": "userId,movieId,tag,timestamp\n3,1,NA,103\n",
        "links": "movieId,imdbId,tmdbId\n1,0114709,10\n2,0113497,\n3,0113228,30\n",
    }
    for name, text in files.items():
        (tmp_path / f"{name}.csv").write_text(text, encoding="utf-8")
    return tmp_path


def test_load_valid_data_and_nullable_types(dataset):
    data = load_data(dataset)
    assert set(data) == {"movies", "ratings", "tags", "links"}
    assert str(data["movies"]["movieId"].dtype) == "Int64"
    assert str(data["ratings"]["rating"].dtype) == "Float64"
    assert isinstance(data["movies"]["title"].dtype, pd.StringDtype)
    assert pd.isna(data["links"].loc[1, "tmdbId"])
    assert data["tags"].loc[0, "tag"] == "NA"


def test_missing_file(dataset):
    (dataset / "tags.csv").unlink()
    with pytest.raises(FileNotFoundError, match="tags.csv"):
        load_data(dataset)


def test_missing_column(dataset):
    (dataset / "movies.csv").write_text("movieId,title\n1,First\n", encoding="utf-8")
    with pytest.raises(ValueError, match="movies.csv: missing columns: genres"):
        load_data(dataset)


@pytest.mark.parametrize("column,value", [
    ("userId", "abc"), ("userId", "1.5"), ("movieId", "0"),
    ("timestamp", "-1"), ("timestamp", "tomorrow"),
    ("rating", "inf"), ("rating", "NaN"), ("rating", "bad"),
    ("rating", "0"), ("rating", "5.5"), ("rating", "4.2"),
])
def test_reject_invalid_values(dataset, column, value):
    frame = pd.read_csv(dataset / "ratings.csv", dtype=str)
    frame.loc[0, column] = value
    frame.to_csv(dataset / "ratings.csv", index=False)
    with pytest.raises(ValueError, match=f"ratings.csv: invalid {column}"):
        load_data(dataset)


def test_invalid_link_id(dataset):
    (dataset / "links.csv").write_text("movieId,imdbId,tmdbId\n1,abc,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="links.csv: invalid imdbId"):
        load_data(dataset)


def test_missing_values_and_duplicates_are_preserved(dataset):
    with (dataset / "ratings.csv").open("a", encoding="utf-8") as file:
        file.write("1,1,4.5,100\n2,2,,104\n")
    summary = analyze(load_data(dataset))
    assert summary["counts"]["ratings"] == 5
    assert summary["tables"]["ratings"]["missing"]["rating"] == 1
    assert summary["tables"]["ratings"]["duplicate_rows"] == 1
    assert summary["duplicate_keys"]["ratings_userId_movieId"] == 1


def test_eda_statistics_and_artifacts(dataset):
    output = dataset / "reports"
    summary = run_eda(dataset, output)
    assert summary["counts"] == {
        "movies": 3, "users": 3, "rating_users": 2, "ratings": 3,
        "valid_rating_values": 3, "tags": 1,
    }
    assert summary["mean_rating"] == pytest.approx(13 / 3)
    assert summary["genres"] == {"Drama": 2, "Comedy": 1}
    assert summary["top_10_movies"][0]["movieId"] == 1
    assert summary["top_10_movies"][0]["rating_count"] == 2
    assert summary["top_10_movies"][0]["mean_rating"] == 4.0
    assert summary["movie_statistics"][-1]["mean_rating"] is None
    assert summary["tables"]["links"]["missing"]["tmdbId"] == 1
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    expected = {"rating_distribution.png", "ratings_per_movie.png", "top_10_genres.png", "top_10_movies.png"}
    assert {path.name for path in (output / "figures").glob("*.png")} == expected
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in (output / "figures").glob("*.png"))


def test_empty_tables(dataset):
    for path in dataset.glob("*.csv"):
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    summary = run_eda(dataset, dataset / "reports")
    assert summary["counts"]["ratings"] == 0
    assert summary["mean_rating"] is None
    assert summary["top_10_movies"] == []
