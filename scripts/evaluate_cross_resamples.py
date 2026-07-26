from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cross_resample_evaluation import evaluate_cross_resamples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed Logistic models across negative resamples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/cross_resample_evaluation.yaml",
    )
    arguments = parser.parse_args()
    summary = evaluate_cross_resamples(PROJECT_DIR, arguments.config)
    print(
        json.dumps(
            {
                "combination_count": summary["combination_count"],
                "off_diagonal_statistics": summary["off_diagonal_statistics"],
                "positive_interaction_gain_counts": summary[
                    "positive_interaction_gain_counts"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
