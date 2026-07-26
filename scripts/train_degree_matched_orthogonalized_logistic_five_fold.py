from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.degree_matched_orthogonalized_logistic import (  # noqa: E402
    train_degree_matched_orthogonalized_logistic_five_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the fixed five-fold degree-matched orthogonalized-interaction "
            "Logistic model."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_DIR
            / "configs/degree_matched_orthogonalized_logistic_5fold.yaml"
        ),
    )
    args = parser.parse_args()
    summary = train_degree_matched_orthogonalized_logistic_five_fold(
        PROJECT_DIR, args.config
    )
    print(
        json.dumps(
            {
                "five_fold_test_statistics": summary[
                    "five_fold_test_statistics"
                ],
                "oof_metrics": summary["oof_metrics"],
                "oof_differences": summary["oof_differences"],
                "interaction_orthogonality": summary["audit"][
                    "interaction_orthogonality"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
