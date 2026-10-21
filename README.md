# TCR prediction for unseen peptide–HLA-I targets

This benchmark evaluates models that predict whether a T-cell receptor (TCR) recognizes a peptide presented by HLA-I. The evaluation focuses on peptide–HLA-I targets held out from training.

The repository contains the revised study code. Release **v1.1.0** provides the saved inputs, models, predictions and result tables. You can check those results without retraining the models.

Two analysis names are used in the files:

- **H1:** the corrected main analysis, using the original positive examples.
- **G2:** a separate sensitivity analysis that compares seven ways of constructing negative examples, S0–S6, with the same positive examples across that comparison.

H1 and G2 use different input versions. Keep their results separate; G2 does not replace the main H1 analysis.

## Download the study materials

1. Download [tcr-h1-g2-v1.1.0-materials.zip](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/download/v1.1.0/tcr-h1-g2-v1.1.0-materials.zip) from the [v1.1.0 release](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/tag/v1.1.0).
2. Extract its `materials/` directory beside `repaired/` in this repository. Download the materials ZIP as well as the repository source; they are separate files.
3. Use the setup and checks below. Access is public and requires no reviewer account or access request.

The ZIP checksum and the list of included files are in [release/ASSET.json](release/ASSET.json) and [release/MATERIALS_MANIFEST.json](release/MATERIALS_MANIFEST.json).

## Check the results

Use Python 3.10–3.12. The release checks were run with Python 3.12.14 on Windows; the frozen training environment used Python 3.10.13. Create an environment, activate it, and install the small review dependency set:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r repaired/requirements-review.txt
python repaired/review.py verify-integrity
python repaired/review.py verify-results
python repaired/review.py verify-provenance
python repaired/review.py load-models
python repaired/review.py score-smoke --records 4
```

If you extract the materials elsewhere, use `python repaired/review.py --materials /path/to/materials <command>`. See the [command guide](repaired/README.md) for details and numerical tolerances, and [VALIDATION.md](VALIDATION.md) for the recorded test results.

These commands check file integrity, recalculate summaries from **existing predictions**, trace table values to their sources, load the saved models, and score a few records to check agreement with saved predictions. They do not retrain models or perform new statistical tests. Only load the trusted release's joblib files after checking their hashes: loading joblib files can execute Python code.

## What is included

- **H1:** the corrected main inputs; 25 interaction models and 25 analytic Additive controls; predictions and summaries for the internal evaluation; combined predictions from 25 models for the original external cohort; and six fixed exclusion checks.
- **G2:** seven negative-example recipes, five folds and three draws per recipe; 105 interaction models and 105 analytic Additive controls; saved input graphs, predictions for held-out folds, and summaries comparing each recipe with S0.
- The analysis code, settings, environment information and source tables. Links between published table values and their source files are included. Earlier exploratory figure tables are identified separately.

The analytic Additive control gives a constant score. Beating it does not, by itself, show that a model has learned genuine receptor–target pairing. The comparisons remain descriptive rather than formal statistical evidence of such learning; the recorded status is **NOT_INFERENTIAL**.

The external cohort is the same cohort corrected after review, not a new blind validation dataset.

## Reproduction scope and version history

The tested workflow starts from the supplied inputs, saved models and predictions and checks the reported results. It is not a tested download-and-train pipeline.

Original training and graph-construction code is included unchanged in [repaired/frozen_code](repaired/frozen_code), listed in the [source index](repaired/FROZEN_SOURCE_INDEX.json), and also included in the materials ZIP. Old absolute paths in those files describe the original setup; they are not instructions for another machine. This release did not newly validate reconstruction from raw database exports, model retraining, or rescoring every record with every checkpoint.

[**v1.0.0**](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/tag/v1.0.0) remains available for the earlier results. Its original CNN and other exploratory outputs were not rerun as part of the H1/G2 correction. Use the version identified for each result in the revised manuscript and source tables.

## Data and licenses

Original project software and documentation retain the MIT License. Third-party database records retain their own terms: the exact VDJdb 2026-06-03 snapshot bundled AGPL-3.0; IEDB data use CC BY 4.0. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). The package contains computationally necessary derived inputs and minimal provenance, not raw provider exports, donor payloads, manuscript drafts, private audit reports, logs, or temporary files.

Cite the repository version and the underlying databases/publications. No new DOI or Zenodo deposit is claimed.
