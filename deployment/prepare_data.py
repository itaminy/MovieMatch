"""Download and validate the pinned official MovieLens archive at build time."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile

from ml.data_loader import DEFAULT_DATA_DIR, load_data

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ARCHIVE_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
REQUIRED = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_BYTES = 50 * 1024 * 1024


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_data(destination=DEFAULT_DATA_DIR, *, archive_path=None, expected_sha256=ARCHIVE_SHA256):
    destination = Path(destination)
    manifest = destination / ".moviematch-source.json"
    if manifest.is_file():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if (saved["archive_sha256"] == expected_sha256
                    and set(saved["files"]) == set(REQUIRED)
                    and all(file_hash(destination / name) == saved["files"][name] for name in REQUIRED)):
                load_data(destination)
                return destination
        except (OSError, ValueError, KeyError, TypeError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moviematch-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        archive = staging / "download.zip"
        if archive_path is not None:
            if Path(archive_path).stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("MovieLens archive exceeds size limit")
            shutil.copyfile(archive_path, archive)
        else:
            try:
                with urlopen(URL, timeout=30) as response, archive.open("wb") as output:
                    total = 0
                    while chunk := response.read(64 * 1024):
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise ValueError("MovieLens download exceeds size limit")
                        output.write(chunk)
            except OSError as exc:
                raise ValueError(f"Cannot download official MovieLens archive: {exc}") from exc
        if file_hash(archive) != expected_sha256:
            raise ValueError("MovieLens SHA-256 mismatch. Upstream latest may have changed; review the dataset version before updating the pinned checksum.")
        extracted = staging / "ml-latest-small"
        extracted.mkdir()
        with ZipFile(archive) as zipped:
            names = [f"ml-latest-small/{name}" for name in REQUIRED]
            if not set(names).issubset(zipped.namelist()):
                raise ValueError("MovieLens archive is missing required CSV files")
            if len(zipped.namelist()) != len(set(zipped.namelist())):
                raise ValueError("Duplicate archive entries")
            for info in zipped.infolist():
                if info.filename.startswith("/") or ".." in info.filename.split("/") or "\\" in info.filename:
                    raise ValueError("Unsafe archive path")
            if sum(info.file_size for info in zipped.infolist()) > MAX_EXTRACTED_BYTES:
                raise ValueError("MovieLens extracted data exceeds size limit")
            for name in (*REQUIRED, "README.txt"):
                member = f"ml-latest-small/{name}"
                if member in zipped.namelist():
                    with zipped.open(member) as source, (extracted / name).open("wb") as output:
                        shutil.copyfileobj(source, output)
        load_data(extracted)
        metadata = {"url": URL, "archive_sha256": expected_sha256,
                    "files": {name: file_hash(extracted / name) for name in REQUIRED}}
        destination.mkdir(parents=True, exist_ok=True)
        for item in extracted.iterdir():
            item.replace(destination / item.name)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional local official archive, still checksum-verified")
    args = parser.parse_args()
    try:
        target = prepare_data(archive_path=args.archive)
    except (OSError, ValueError, BadZipFile, KeyError) as exc:
        parser.exit(1, f"MovieLens preparation failed: {exc}\n")
    print(f"MovieLens ready: {target}; pinned SHA-256: {ARCHIVE_SHA256}")


if __name__ == "__main__":
    main()
