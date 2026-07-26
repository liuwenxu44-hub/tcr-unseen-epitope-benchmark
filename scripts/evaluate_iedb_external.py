from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.iedb_external_evaluation import evaluate_iedb_external  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen IEDB external cohort from persisted scores."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_evaluation.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--analysis-source-commit",
        help="Source commit attested by the controlling Git worktree.",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            evaluate_iedb_external(
                PROJECT_DIR,
                arguments.config,
                output_dir=arguments.output_dir,
                analysis_source_commit=arguments.analysis_source_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
