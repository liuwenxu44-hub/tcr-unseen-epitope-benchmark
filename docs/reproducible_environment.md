# Reproducible environment

The frozen analysis used CPython 3.10.13 on Linux x86_64. Direct runtime dependencies are pinned in the root `requirements.txt`.

## Create the environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_environment.py
```

PyTorch 2.1.0 was used. CNN training benefits from a CUDA-capable GPU, but data construction, Logistic models, statistical analyses, and the prediction CLI can run on CPU.

## Key versions

| Package | Version |
|---|---:|
| Python | 3.10.13 |
| PyTorch | 2.1.0 |
| NumPy | 1.26.0 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| PyYAML | 6.0 |
| RapidFuzz | 3.14.5 |

## Release integrity

Verify the curated release files with:

```bash
sha256sum -c RELEASE_MANIFEST.sha256
```

This check reads existing artifacts only. It does not download data, train models, score cohorts, or recompute statistics.
