from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tcrue.vdjdb import preprocess_vdjdb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cleaned human TRB-MHCI VDJdb table")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "positive_pairs.csv",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "positive_pairs_stats.json",
    )
    parser.add_argument(
        "--pmhc-counts",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "pmhc_counts.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "results" / "vdjdb_feasibility_2026-06-03.md",
    )
    parser.add_argument("--min-pmhc-positives", type=int, default=1)
    parser.add_argument("--min-epitope-length", type=int, default=8)
    parser.add_argument("--max-epitope-length", type=int, default=15)
    parser.add_argument("--allow-noncanonical-tcr", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in (args.output, args.stats, args.pmhc_counts, args.report):
        if path.exists() and not args.force:
            parser.error(f"refusing to overwrite existing output: {path}")

    stats = preprocess_vdjdb(
        args.input.resolve(),
        args.output.resolve(),
        args.stats.resolve(),
        args.pmhc_counts.resolve(),
        args.report.resolve(),
        min_epitope_length=args.min_epitope_length,
        max_epitope_length=args.max_epitope_length,
        require_canonical_tcr=not args.allow_noncanonical_tcr,
        min_pmhc_positives=args.min_pmhc_positives,
    )
    print(json.dumps(stats["rows"], indent=2, ensure_ascii=True))
    print(json.dumps(stats["output"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
