from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.query_balanced_n1 import build_query_balanced_n1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cap100 query-balanced N1 data for rebalanced five-fold splits"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_DIR
            / "configs"
            / "distance_aware_5fold_query_balanced_n1_cap100.yaml"
        ),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    manifest = build_query_balanced_n1(
        arguments.config, PROJECT_DIR, force=arguments.force
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
