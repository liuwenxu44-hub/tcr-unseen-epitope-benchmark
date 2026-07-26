# Setup

## Supported environment

- Python 3.10.13
- Linux x86_64
- CPU execution for preprocessing, Logistic models, statistics, and prediction
- CUDA-capable GPU recommended for CNN training

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_environment.py
```

The versions in `requirements.txt` match the frozen analysis environment. PyTorch installations may require a platform-specific wheel index; consult the official PyTorch installation instructions when the default wheel is unsuitable for the local CUDA runtime.

## Basic checks

```bash
python predict.py --help
python scripts/download_vdjdb.py --help
python scripts/build_distance_aware_5fold_rebalanced.py --help
python scripts/build_degree_matched_negatives.py --help
python scripts/train_degree_matched_orthogonalized_logistic_five_fold.py --help
```

## Data downloads

Raw VDJdb and IEDB exports are intentionally absent. Use:

```bash
python scripts/download_vdjdb.py
python scripts/download_iedb_receptors.py
```

The scripts record source URLs, retrieval metadata, file sizes, and SHA-256 checksums. Continue with the ordered workflow in `docs/reproduce_from_scratch.md`.
