# TCR unseen-epitope benchmark

This repository contains the curated research release for a TCR CDR3beta-peptide-HLA benchmark designed to evaluate unseen-epitope generalization while controlling marginal sequence and query shortcuts.

The confirmatory workflow combines:

- distance-aware peptide components and five-fold splitting;
- query balancing with a cap of 100 positive interactions per pMHC query;
- exact TCR-degree and pMHC-query-degree matched putative negatives;
- five independent negative resamples and cross-resample evaluation;
- publication-independent external validation using IEDB;
- frozen shallow, convolutional, and sequence-similarity baselines.

The public release contains the scientific pipeline, frozen configurations, benchmark manifests, final result summaries, registered model artifacts, and a command-line ranking utility. Raw VDJdb and IEDB exports are not redistributed.

## Installation

Python 3.10.13 was used for the frozen analysis.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_environment.py
```

See [SETUP.md](SETUP.md) for platform notes and [docs/reproduce_from_scratch.md](docs/reproduce_from_scratch.md) for the ordered analysis workflow.

## Rank candidate TCRs

The command-line utility uses the frozen 25-model Orthogonalized-interaction Logistic ensemble.

```bash
python predict.py \
  --epitope NLVPMVATV \
  --hla HLA-A*02:01 \
  --tcr-file examples/candidate_tcrs.csv \
  --out ranked_tcrs.csv
```

The output is a research ranking score, not a calibrated probability. It must not be used for clinical decisions or as a substitute for experimental validation. Full input and output details are provided in [docs/prediction_cli.md](docs/prediction_cli.md).

## Repository layout

- `src/`: data processing, model, scoring, and statistical analysis modules
- `scripts/`: reproducible command-line stages for the confirmatory workflow
- `configs/`: frozen analysis configurations
- `data/`: compact derived benchmark cohorts and manifests
- `results/`: frozen summaries, provenance records, and registered model artifacts
- `docs/`: reproducibility, model-card, limitations, and CLI documentation

## Data sources

The internal benchmark was derived from VDJdb. The independent external cohort was derived from IEDB. Download scripts, source metadata, checksums, filtering rules, and manifests are included; raw database exports are excluded. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Reproducibility and scope

Internal and IEDB findings are reported separately. Confirmatory results are distinguished from exploratory analyses, and superseded outputs are excluded from this release. The release file hashes are recorded in `RELEASE_MANIFEST.sha256`.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

Original software in this repository is released under the MIT License. Third-party data retain their original terms and citation requirements.
