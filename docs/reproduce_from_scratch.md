# Reproduce the confirmatory workflow

This document lists the analysis stages in their frozen order. Commands are run from the repository root after installing the environment described in `docs/reproducible_environment.md`.

Raw VDJdb and IEDB exports are intentionally excluded from the repository. Their download scripts record source metadata and SHA-256 checksums.

## 1. Download and harmonize VDJdb

```bash
python scripts/download_vdjdb.py
python scripts/preprocess_vdjdb.py
python scripts/build_data_tiers.py
```

Outputs include the positive interaction registry, Development tier, filtering counts, and manifests. The processing code restricts the benchmark to human TRB CDR3 amino-acid sequences, linear peptide epitopes, and standardized classical HLA class I annotations.

## 2. Construct the distance-aware benchmark

```bash
python scripts/build_distance_aware_5fold_rebalanced.py
python scripts/build_query_balanced_n1.py
python scripts/build_degree_matched_negatives.py
python scripts/build_degree_matched_negative_resamples.py
```

The split keeps peptide-similarity connected components within one fold at a normalized Levenshtein threshold of 0.5. Query balancing caps positives at 100 per pMHC query. Degree matching requires identical positive and negative degrees for every TCR and every pMHC query.

## 3. Train the confirmatory models

```bash
python scripts/train_degree_matched_additive_logistic_five_fold.py
python scripts/train_degree_matched_orthogonalized_logistic_five_fold.py
python scripts/train_degree_matched_tcr_only_five_fold.py
python scripts/train_interaction_resample_robustness.py
```

The CNN fold configurations and `scripts/train_baseline.py` reproduce the frozen convolutional baseline. Each fold fits feature mappings on training data only.

## 4. Evaluate negative-resample robustness

```bash
python scripts/evaluate_cross_resamples.py
python scripts/analyze_cross_resample_final_statistics.py
python scripts/evaluate_degree_matched_five_fold_similarity.py
```

The final paired analysis preserves the repeated-measure structure across five training-negative and five testing-negative resamples.

## 5. Build the independent IEDB cohort

```bash
python scripts/download_iedb_receptors.py
python scripts/audit_iedb_external.py
python scripts/decide_iedb_cohort.py
python scripts/build_iedb_external_cohort.py
```

The frozen cohort requires mapped publication references, publication independence from VDJdb, unseen peptides, maximum peptide similarity no greater than 0.5, a query cap of 100, and exact degree matching.

## 6. Score and evaluate IEDB

```bash
python scripts/score_iedb_external.py
python scripts/evaluate_iedb_external.py
python scripts/analyze_iedb_external_strata.py
python scripts/freeze_iedb_results.py
```

Scoring uses the checkpoint registry and its SHA-256 values. External performance is not used for model selection, calibration, threshold selection, or hyperparameter tuning.

## 7. Verify the curated release

```bash
sha256sum -c RELEASE_MANIFEST.sha256
python scripts/verify_environment.py
python predict.py --help
```

The final internal and external summaries remain separate under `results/`.
