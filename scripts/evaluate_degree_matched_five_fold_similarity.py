from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.degree_matched_five_fold_similarity import (  # noqa: E402
    evaluate_degree_matched_five_fold_similarity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate similarity on degree-matched five-fold test data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/degree_matched_5fold_similarity.yaml",
    )
    arguments = parser.parse_args()
    summary = evaluate_degree_matched_five_fold_similarity(
        arguments.config.resolve(), PROJECT_DIR
    )
    print(
        json.dumps(
            {
                "audit": summary["audit"]["status"],
                "available_query_count": summary["coverage"][
                    "available_query_count"
                ],
                "query_count": summary["coverage"]["query_count"],
                "query_coverage": summary["coverage"]["query_coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
