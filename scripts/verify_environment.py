from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = "3.10.13"
EXPECTED_PACKAGES = {
    "joblib": "1.5.3",
    "networkx": "3.1",
    "numpy": "1.26.0",
    "pandas": "2.3.3",
    "pyyaml": "6.0",
    "rapidfuzz": "3.14.5",
    "scikit-learn": "1.7.2",
    "scipy": "1.15.3",
    "torch": "2.1.0",
    "tqdm": "4.65.0",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_packages() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_PACKAGES.items():
        version = importlib.metadata.version(name)
        if version != expected:
            raise RuntimeError(f"package mismatch for {name}: {version} != {expected}")
        observed[name] = version
    return observed


def verify_registry() -> dict[str, int]:
    registry_path = (
        PROJECT_DIR / "results/external/iedb/external_checkpoint_registry.csv"
    )
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("checkpoint registry is empty")
    for row in rows:
        artifact = PROJECT_DIR / row["path"]
        if not artifact.is_file():
            raise RuntimeError(f"registered artifact is missing: {row['path']}")
        observed = file_sha256(artifact)
        if observed != row["expected_sha256"] or observed != row["observed_sha256"]:
            raise RuntimeError(f"registered artifact hash mismatch: {row['path']}")
        if artifact.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"registered artifact size mismatch: {row['path']}")
    return {
        "artifact_count": len(rows),
        "model_checkpoint_count": sum(
            row["artifact_type"] == "model_checkpoint" for row in rows
        ),
        "training_reference_count": sum(
            row["artifact_type"] == "train_positive_reference" for row in rows
        ),
    }


def verify_result_files() -> int:
    paths = (
        "results/final/result_registry.json",
        "results/final/freeze_manifest.json",
        "results/external/iedb/external_result_registry.json",
        "results/external/iedb/external_freeze_manifest.json",
        "results/external/iedb/evaluation/primary_comparison.json",
    )
    for relative in paths:
        value = json.loads((PROJECT_DIR / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object: {relative}")
    return len(paths)


def main() -> None:
    python_version = platform.python_version()
    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Python mismatch: {python_version} != {EXPECTED_PYTHON}"
        )
    result = {
        "status": "passed",
        "python_version": python_version,
        "platform": platform.platform(),
        "packages": verify_packages(),
        "checkpoint_registry": verify_registry(),
        "frozen_result_file_count_read": verify_result_files(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
