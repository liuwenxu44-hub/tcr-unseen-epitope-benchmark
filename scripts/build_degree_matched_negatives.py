from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.degree_matched_negatives import (  # noqa: E402
    ExactDegreeMatchError,
    build_degree_matched_negatives,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact degree-matched negatives for all five folds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/degree_matched_negatives.yaml",
    )
    args = parser.parse_args()
    try:
        manifest = build_degree_matched_negatives(args.config, PROJECT_DIR)
    except ExactDegreeMatchError as error:
        print(json.dumps(error.report, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error
    summary = {
        fold_id: {
            partition: manifest["folds"][fold_id]["partitions"][partition][
                "audit"
            ]["matching_rate"]
            for partition in ("train", "validation", "test")
        }
        for fold_id in (f"fold_{index}" for index in range(1, 6))
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
