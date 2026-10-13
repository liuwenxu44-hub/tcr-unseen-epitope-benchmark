# Strict unseen peptide–HLA-I TCR benchmark

This repository provides the computational materials for the revised TCR–peptide–HLA-I benchmarking study. **v1.1.0** contains the repaired **H1 primary analysis** and the separate **G2 sensitivity analysis**, with a portable entry point for checking frozen results.

## Get the repaired materials

1. Download [tcr-h1-g2-v1.1.0-materials.zip](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/download/v1.1.0/tcr-h1-g2-v1.1.0-materials.zip) from the [v1.1.0 release](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/tag/v1.1.0).
2. Extract its `materials/` directory beside `repaired/` in this repository. The release asset and Git source snapshot are separate downloads.
3. Use the setup and checks below. Access is public and requires no reviewer account or access request.

The asset checksum and member manifest are in [release/ASSET.json](release/ASSET.json) and [release/MATERIALS_MANIFEST.json](release/MATERIALS_MANIFEST.json).

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

For an extraction elsewhere, use `python repaired/review.py --materials /path/to/materials <command>`. See [the review adapter](repaired/README.md) for exact scope and tolerances, and [VALIDATION.md](VALIDATION.md) for checks actually performed.

The commands verify packaged hashes, recalculate summaries from **existing predictions**, verify table source selectors, load frozen models, and check a small scoring roundtrip. They do not retrain models or perform new statistical tests. Joblib artifacts must only be loaded from the trusted, hash-verified release.

## What is included

- **H1:** repaired original-positive primary inputs; 25 interaction-model members and their 25 analytic Additive counterparts; internal fold/cell/query predictions and summaries; the original external cohort and its 25-member probability ensemble; six fixed exclusion diagnostics.
- **G2:** all seven recipes S0–S6, five folds and three corresponding draws; 105 interaction models and 105 analytic Additive artifacts; frozen input graphs, OOF predictions, query/draw/recipe summaries and S0 comparisons.
- Frozen code, configuration, environment records, necessary source tables and compact publication/evidence crosswalks. Historical unchanged exploratory figure source tables are identified separately.

H1 and G2 use different frozen input generations and must not be pooled. G2's common-positive grid does not replace the H1 main analysis. Analytic Additive is a constant-score control; exceeding it does not establish true pairing information. The revised inferential status remains **NOT_INFERENTIAL**. The same external cohort was repaired after review; it is not a new blind validation.

## Reproduction scope and version history

The supported portable path is frozen inputs/models/predictions to result verification. Original training and graph-construction sources are supplied byte-for-byte in [repaired/frozen_code](repaired/frozen_code), with [a source index](repaired/FROZEN_SOURCE_INDEX.json), and also in the materials asset. Their archived absolute paths identify historical dependencies; they are not portable launch instructions. Full reconstruction from raw provider exports, retraining, and full checkpoint rescoring were not newly validated for this release.

[**v1.0.0**](https://github.com/liuwenxu44-hub/tcr-unseen-epitope-benchmark/releases/tag/v1.0.0) remains the historical release. Its original CNN and other exploratory outputs are not presented as newly repaired H1/G2 runs. Use the version specified for each result in the revised manuscript and source tables.

## Data and licenses

Original project software and documentation retain the MIT License. Third-party database records retain their own terms: the exact VDJdb 2026-06-03 snapshot bundled AGPL-3.0; IEDB data use CC BY 4.0. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). The package contains computationally necessary derived inputs and minimal provenance, not raw provider exports, donor payloads, manuscript drafts, private audit reports, logs, or temporary files.

Cite the repository version and the underlying databases/publications. No new DOI or Zenodo deposit is claimed.
