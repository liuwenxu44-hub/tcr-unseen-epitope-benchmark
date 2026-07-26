from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cross_resample_final_statistics import (  # noqa: E402
    analyze_cross_resample_final_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final paired statistics on persisted cross-resample scores."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/cross_resample_final_statistics.yaml",
    )
    arguments = parser.parse_args()
    result = analyze_cross_resample_final_statistics(PROJECT_DIR, arguments.config)
    print(
        json.dumps(
            {
                "query_count": result["query_count"],
                "combination_count": result["combination_count"],
                "prediction_row_count": result["audit"]["prediction_row_count"],
                "metrics": result["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
