from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.data_tiers import build_data_tiers  # noqa: E402


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze development and strict positives")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "data_tiers.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        parser.error("config must contain a YAML mapping")

    source_path = _resolve_project_path(config["source"]["input_csv"])
    outputs = config["outputs"]
    development_path = _resolve_project_path(outputs["development_csv"])
    strict_path = _resolve_project_path(outputs["strict_csv"])
    manifest_path = _resolve_project_path(outputs["manifest_json"])
    report_path = _resolve_project_path(outputs["report_md"])

    for path in (development_path, strict_path, manifest_path, report_path):
        if path.exists() and not args.force:
            parser.error(f"refusing to overwrite existing output: {path}")

    manifest = build_data_tiers(
        source_path=source_path,
        config_path=config_path,
        config=config,
        development_path=development_path,
        strict_path=strict_path,
        manifest_path=manifest_path,
        report_path=report_path,
    )
    summary = {
        name: {
            "positive_pairs": values["positive_pairs"],
            "unique_pmhcs": values["unique_pmhcs"],
        }
        for name, values in manifest["tiers"].items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
