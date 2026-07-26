from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.degree_matched_negatives import ExactDegreeMatchError  # noqa: E402
from tcrue.degree_matched_resamples import (  # noqa: E402
    build_degree_matched_resamples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build five exact degree-matched negative resamples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/degree_matched_negative_resamples.yaml",
    )
    args = parser.parse_args()
    try:
        manifest = build_degree_matched_resamples(args.config, PROJECT_DIR)
    except ExactDegreeMatchError as error:
        print(json.dumps(error.report, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                "exact_solution_success_rate": manifest["statistics"][
                    "exact_solution_success_rate"
                ],
                "resample_pairwise_jaccard": manifest["statistics"][
                    "resample_pairwise_jaccard"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
