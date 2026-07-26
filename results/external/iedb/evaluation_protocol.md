# Frozen IEDB external evaluation protocol

## Pre-scoring declaration

This protocol was frozen before any external model scoring. No external model performance has been viewed, and no external prediction was generated during protocol freezing.

Figure 1 remains a draft and must not be formally updated until external validation is complete.

## Formal cohort

- Queries: 141
- Positives: 806
- Negatives: 806
- Exact degree matching: 1.000000

All cohort quantities above are read from the frozen formal-cohort manifest.

## Frozen methods

- Orthogonalized-interaction Logistic: 25 checkpoints; arithmetic mean of all checkpoint probabilities.
- Additive-only Logistic: 25 checkpoints; arithmetic mean of all checkpoint probabilities.
- Degree-matched CNN: 5 checkpoints; arithmetic mean across folds.
- Degree-matched TCR-only Logistic: 5 checkpoints; arithmetic mean across folds.
- Sequence Similarity: 5 fold-specific positive training references; same-HLA only and no cross-HLA fallback. Available-fold scores are averaged.

Any missing artifact or SHA-256 mismatch is a hard stop.

## Endpoints and comparison

The sole primary endpoint is query-level Macro AUPRC. The primary comparison is Orthogonalized-interaction Logistic minus Additive-only Logistic over all formal-cohort queries.

Secondary endpoints are Macro AUROC, pooled AUROC, and pooled AUPRC. Similarity-related comparisons are restricted to common evaluable queries and must report coverage separately.

## Statistical analysis

The query is the independent sampling unit. The paired analysis uses 10,000 query-cluster bootstrap replicates for 95% confidence intervals and 10,000 query-level paired permutations for two-sided p values. The three secondary endpoints use Holm correction; the pre-specified Macro AUPRC primary endpoint is reported as the sole primary test.

No external data may be used for model selection, calibration, threshold selection, or hyperparameter tuning.

## Exploratory stratification

HLA seen/unseen, TCR seen/unseen, and antigen source analyses are exploratory. They cannot replace the primary analysis over the complete formal cohort.

## Operations performed

No training, scoring, prediction, resampling, statistical testing, or plotting was performed while freezing this protocol.
