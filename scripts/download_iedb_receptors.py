from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, specification: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    observed_sha256 = file_sha256(path)
    if observed_size != int(specification["size_bytes"]):
        raise ValueError(
            f"IEDB archive size mismatch for {path}: "
            f"{observed_size} != {specification['size_bytes']}"
        )
    if observed_sha256 != specification["sha256"]:
        raise ValueError(
            f"IEDB archive SHA-256 mismatch for {path}: "
            f"{observed_sha256} != {specification['sha256']}"
        )
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"corrupt ZIP member: {corrupt_member}")
        if specification["member"] not in archive.namelist():
            raise ValueError(
                f"expected ZIP member is absent: {specification['member']}"
            )
        member = archive.getinfo(specification["member"])
    return {
        "path": str(path.relative_to(PROJECT_DIR)),
        "url": specification["url"],
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "member": specification["member"],
        "member_uncompressed_size_bytes": member.file_size,
        "member_crc32": f"{member.CRC:08x}",
        "zip_integrity": "passed",
    }


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TCR-unseen-epitope-IEDB-audit/1.0"},
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                shutil.copyfileobj(response, temporary)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify pinned IEDB receptor_full v3 exports."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_audit.yaml",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing archive after downloading the pinned URL.",
    )
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    source = config["source"]
    verified: dict[str, Any] = {}
    for name, specification in source["archives"].items():
        destination = PROJECT_DIR / specification["path"]
        if arguments.force_download or not destination.exists():
            download_archive(specification["url"], destination)
        verified[name] = verify_archive(destination, specification)

    manifest = {
        "schema_version": 1,
        "database": source["database"],
        "export_name": source["export_name"],
        "export_page": source["export_page"],
        "export_update_date": str(source["export_update_date"]),
        "download_date": str(source["download_date"]),
        "update_frequency": source["update_frequency"],
        "license": source["license"],
        "citation": source["citation"],
        "source_note": (
            "Official IEDB receptor_full v3 and reference_full v3 CSV exports; "
            "the export page states that these database exports are updated weekly."
        ),
        "archives": verified,
    }
    manifest_path = PROJECT_DIR / source["download_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
