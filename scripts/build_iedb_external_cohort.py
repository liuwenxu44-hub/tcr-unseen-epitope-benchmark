from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.tcrue.iedb_external_cohort import build_iedb_external_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen IEDB strict cap100 external test cohort."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_cohort.yaml",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_iedb_external_cohort(PROJECT_DIR, arguments.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
