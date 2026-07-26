from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.freeze_iedb_results import freeze_iedb_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the traceable IEDB external validation conclusions."
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            freeze_iedb_results(PROJECT_DIR, output_dir=arguments.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
