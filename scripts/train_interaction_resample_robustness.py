from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.interaction_resample_robustness import (  # noqa: E402
    train_interaction_resample_robustness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fixed Logistic models across five negative resamples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/interaction_resample_robustness.yaml",
    )
    arguments = parser.parse_args()
    summary = train_interaction_resample_robustness(
        PROJECT_DIR, arguments.config
    )
    print(
        json.dumps(
            {
                "trained_model_count": summary["trained_model_count"],
                "five_resample_statistics": summary["five_resample_statistics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
