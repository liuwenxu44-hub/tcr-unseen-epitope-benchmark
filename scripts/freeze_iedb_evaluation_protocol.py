from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.tcrue.iedb_evaluation_protocol import (  # noqa: E402
    freeze_iedb_evaluation_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-scoring IEDB external evaluation protocol."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_evaluation.yaml",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            freeze_iedb_evaluation_protocol(PROJECT_DIR, arguments.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
