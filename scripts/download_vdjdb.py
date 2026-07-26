from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


VERSION = "2026-06-03"
RELEASE_TAG = "2026-06-11-ZENODO"
ARCHIVE_NAME = f"vdjdb-{VERSION}.zip"
DOWNLOAD_URL = (
    "https://github.com/antigenomics/vdjdb-db/releases/download/"
    f"{RELEASE_TAG}/{ARCHIVE_NAME}"
)
ARCHIVE_SHA256 = "9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1"
FULL_TABLE_SHA256 = "55ee89fa1a0109f52f60fe30fd47c2f6842f94a0476d034b2d98b2511748bddc"
RELEASE_PUBLISHED_AT_UTC = "2026-06-11T10:01:20Z"


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )


def download(url: str, destination: Path, attempts: int = 5) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tcrue-data-downloader/1.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 20))


def locate_full_table(archive: zipfile.ZipFile) -> str:
    candidates = [
        name
        for name in archive.namelist()
        if Path(name).name in {"vdjdb_full.txt", "vdjdb.full.txt"}
        and not name.endswith("/")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one VDJdb full table in archive, found: {candidates}")
    return candidates[0]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download and verify a pinned VDJdb release")
    parser.add_argument("--out-dir", type=Path, default=project_dir / "data" / "raw")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded release archive instead of downloading it",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / ARCHIVE_NAME
    table_path = out_dir / f"vdjdb_full_{VERSION}.tsv"
    manifest_path = out_dir / f"vdjdb_{VERSION}_manifest.json"

    if args.archive:
        source_archive = args.archive.resolve()
        verify(source_archive, ARCHIVE_SHA256)
        if source_archive != archive_path:
            if archive_path.exists() and not args.force:
                verify(archive_path, ARCHIVE_SHA256)
            else:
                shutil.copy2(source_archive, archive_path)
    elif archive_path.exists() and not args.force:
        verify(archive_path, ARCHIVE_SHA256)
    else:
        download(DOWNLOAD_URL, archive_path)
        verify(archive_path, ARCHIVE_SHA256)

    with zipfile.ZipFile(archive_path) as archive:
        member = locate_full_table(archive)
        if table_path.exists() and not args.force:
            verify(table_path, FULL_TABLE_SHA256)
        else:
            temporary_table = table_path.with_suffix(table_path.suffix + ".part")
            with archive.open(member) as source, temporary_table.open("wb") as target:
                shutil.copyfileobj(source, target)
            verify(temporary_table, FULL_TABLE_SHA256)
            temporary_table.replace(table_path)

    manifest = {
        "database": "VDJdb",
        "content_version": VERSION,
        "release_tag": RELEASE_TAG,
        "download_url": DOWNLOAD_URL,
        "release_published_at_utc": RELEASE_PUBLISHED_AT_UTC,
        "archive": {
            "file": archive_path.name,
            "sha256": ARCHIVE_SHA256,
        },
        "full_table": {
            "archive_member": member,
            "file": table_path.name,
            "sha256": FULL_TABLE_SHA256,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(table_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
