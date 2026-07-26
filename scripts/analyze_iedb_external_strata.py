from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.iedb_external_stratified_analysis import (  # noqa: E402
    analyze_iedb_external_strata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-specified exploratory IEDB stratified analysis."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--analysis-source-commit",
        help="Source commit attested by the controlling Git worktree.",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze_iedb_external_strata(
                PROJECT_DIR,
                output_dir=arguments.output_dir,
                analysis_source_commit=arguments.analysis_source_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
