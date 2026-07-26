from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.tcrue.iedb_external_audit import run_iedb_feasibility_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit IEDB receptor_full v3 as an independent external data source."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/external_iedb_audit.yaml",
    )
    arguments = parser.parse_args()
    result = run_iedb_feasibility_audit(PROJECT_DIR, arguments.config)
    print(
        json.dumps(
            {
                "raw_record_count": result["raw_record_count"],
                "cleaned_record_count": result["cleaned_record_count"],
                "tier_summary": result["tier_summary"],
                "decision": result["decision"],
                "reproducibility_passed": result["reproducibility_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
