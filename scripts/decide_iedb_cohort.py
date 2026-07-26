from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.tcrue.iedb_cohort_decision import decide_iedb_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the minimum-positive threshold for the IEDB strict-C cohort."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_cohort.yaml",
    )
    arguments = parser.parse_args()
    result = decide_iedb_cohort(PROJECT_DIR, arguments.config)
    print(
        json.dumps(
            {
                "sole_conditional_reason": result["original_decision_audit"][
                    "sole_conditional_reason"
                ],
                "thresholds": result["thresholds"],
                "selected_threshold": result["selection"][
                    "minimum_positives_per_query"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
