#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.distance_aware_5fold import (  # noqa: E402
    build_distance_aware_five_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build rebalanced Development distance-aware five-fold positives."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/distance_aware_5fold_rebalanced.yaml",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    manifest = build_distance_aware_five_folds(
        arguments.config.resolve(), PROJECT_DIR, force=arguments.force
    )
    print(
        json.dumps(
            {
                fold_name: {
                    "test_queries": values["test_query_count"],
                    "test_records": values["partitions"]["test"][
                        "positive_pairs"
                    ],
                    "maximum_similarity": values["cross_partition_similarity"][
                        "maximum_similarity"
                    ],
                }
                for fold_name, values in manifest["folds"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
