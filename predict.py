from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate TCR CDR3beta sequences with the frozen 25-model "
            "Orthogonalized-interaction Logistic ensemble. Research use only."
        )
    )
    parser.add_argument("--epitope", required=True, help="Linear peptide sequence.")
    parser.add_argument("--hla", required=True, help="Full-resolution classical HLA.")
    parser.add_argument(
        "--tcr-file",
        required=True,
        type=Path,
        help="CSV containing a tcr_cdr3_beta column.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Destination ranked CSV path."
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        from src.predict_cli import rank_candidate_tcrs

        summary = rank_candidate_tcrs(
            PROJECT_DIR,
            epitope=arguments.epitope,
            hla=arguments.hla,
            tcr_file=arguments.tcr_file,
            out_path=arguments.out,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
