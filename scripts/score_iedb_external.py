from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.iedb_external_scoring import score_iedb_external_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind-score the frozen IEDB external cohort without metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_evaluation.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--isolated-clean-source-commit",
        help="Clean source commit attested by the controlling worktree for an isolated execution copy.",
    )
    arguments = parser.parse_args()
    isolated = arguments.isolated_clean_source_commit is not None
    print(
        json.dumps(
            score_iedb_external_cohort(
                PROJECT_DIR,
                arguments.config,
                output_dir=arguments.output_dir,
                enforce_clean_worktree=not isolated,
                clean_source_commit=arguments.isolated_clean_source_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
