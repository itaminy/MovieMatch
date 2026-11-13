"""Deployment tests use tiny ZIP/CSV fixtures, never the real network."""
import io
import json
from pathlib import Path
from urllib.error import URLError
from zipfile import ZipFile

import pytest
import deployment.prepare_data as preparation
from deployment.start import api_url, launch_command


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "sample.zip"
    files = {
        "movies.csv": "movieId,title,genres\n1,Toy,Comedy\n",
        "ratings.csv": "userId,movieId,rating,timestamp\n1,1,4.0,100\n",
        "tags.csv": "userId,movieId,tag,timestamp\n1,1,fun,100\n",
        "links.csv": "movieId,imdbId,tmdbId\n1,1,1\n",
    }
    with ZipFile(path, "w") as zipped:
        for name, text in files.items():
            zipped.writestr(f"ml-latest-small/{name}", text)
    return path


def prepare(archive, target):
    return preparation.prepare_data(target, archive_path=archive,
                                    expected_sha256=preparation.file_hash(archive))


def test_local_preparation_and_cache(archive, tmp_path, monkeypatch):
    target = tmp_path / "data"
    assert prepare(archive, target) == target
    assert all((target / name).exists() for name in preparation.REQUIRED)
    manifest = json.loads((target / ".moviematch-source.json").read_text())
    assert manifest["archive_sha256"] == preparation.file_hash(archive)
    monkeypatch.setattr(preparation, "urlopen", lambda *a, **k: pytest.fail("cache must avoid network"))
    preparation.prepare_data(target, expected_sha256=preparation.file_hash(archive))
    (target / "movies.csv").write_text("corrupted")
    prepare(archive, target)
    assert "Toy" in (target / "movies.csv").read_text()


def test_download(archive, tmp_path, monkeypatch):
    def download(url, timeout):
        assert url == preparation.URL and timeout == 30
        return io.BytesIO(archive.read_bytes())
    monkeypatch.setattr(preparation, "urlopen", download)
    preparation.prepare_data(tmp_path / "downloaded", expected_sha256=preparation.file_hash(archive))


def test_network_error(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("unavailable")
    monkeypatch.setattr(preparation, "urlopen", unavailable)
    with pytest.raises(ValueError, match="Cannot download"):
        preparation.prepare_data(tmp_path / "data")


def test_checksum_mismatch(archive, tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        preparation.prepare_data(tmp_path / "data", archive_path=archive)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("case", ["missing", "unsafe", "invalid_csv"])
def test_bad_archive(archive, tmp_path, case):
    if case == "unsafe":
        with ZipFile(archive, "a") as zipped:
            zipped.writestr("../outside", "bad")
    elif case == "invalid_csv":
        with ZipFile(archive) as zipped:
            contents = {name: zipped.read(name) for name in zipped.namelist()}
        contents["ml-latest-small/ratings.csv"] = b"wrong\n"
        with ZipFile(archive, "w") as zipped:
            for name, content in contents.items():
                zipped.writestr(name, content)
    if case == "missing":
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("ml-latest-small/movies.csv", "nothing")
    with pytest.raises(ValueError):
        prepare(archive, tmp_path / "data")
    assert not (tmp_path / "outside").exists()


def test_size_limits(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        prepare(archive, tmp_path / "data")
    monkeypatch.setattr(preparation, "urlopen", lambda *a, **kw: io.BytesIO(b"too large"))
    with pytest.raises(ValueError, match="size limit"):
        preparation.prepare_data(tmp_path / "download")


def test_api_command():
    command = launch_command("api", {"PORT": "12345"})
    assert command[1:] == ["-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "12345", "--workers", "1"]
    assert "--reload" not in command


@pytest.mark.parametrize("port", ["0", "65536", "bad", "-1", "12.5", ""])
def test_invalid_port(port):
    with pytest.raises(ValueError, match="PORT"):
        launch_command("api", {"PORT": port})


def test_frontend_internal_address():
    env = {"PORT": "12345", "MOVIEMATCH_API_HOSTPORT": "moviematch-api:10000"}
    command = launch_command("frontend", env)
    assert env["MOVIEMATCH_API_URL"] == "http://moviematch-api:10000"
    assert "--server.port=12345" in command
    assert "--server.address=0.0.0.0" in command
    assert "--server.enableCORS=false" not in command


def test_explicit_api_url():
    env = {"MOVIEMATCH_API_URL": "https://example.test/api/", "MOVIEMATCH_API_HOSTPORT": "other:10000"}
    assert api_url(env) == "https://example.test/api"


@pytest.mark.parametrize("url", ["", "http://localhost:8001", "http://127.0.0.1:8001", "http://[::1]", "http://0.0.0.0", "ftp://api", "http://user:secret@api", "http://api?query=1", "http://api:99999"])
def test_deployed_frontend_rejects_missing_or_local_url(url):
    with pytest.raises(ValueError):
        launch_command("frontend", {"MOVIEMATCH_API_URL": url})


def test_render_configuration():
    root = Path(__file__).resolve().parents[1]
    config = (root / "render.yaml").read_text(encoding="utf-8-sig")
    assert config.count('- type:') == 1
    assert 'type: web' in config and 'plan: free' in config
    assert 'healthCheckPath: /health' in config
    assert 'buildCommand: python -m pip install -r requirements.txt && python -m pip check && python -m deployment.prepare_data' in config
    assert 'startCommand: python -m deployment.start api' in config
    assert 'OPENBLAS_NUM_THREADS' in config and 'OMP_NUM_THREADS' in config
    assert config.count('value: "1"') == 2
    assert 'pserv' not in config and 'fromService' not in config
    assert '0.5c-' not in config and 'moviematch-ui' not in config
    assert 'deployment.start frontend' not in config
    assert 'run_evaluation' not in config and '127.0.0.1' not in config
    assert (root / '.python-version').read_text().strip() == '3.14'


def test_api_default_port():
    command = launch_command("api", {})
    assert command[command.index("--port") + 1] == "10000"
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--workers") + 1] == "1"
