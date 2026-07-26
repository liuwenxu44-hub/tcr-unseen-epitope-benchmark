# IEDB external evaluation

This report applies the pre-scoring frozen protocol without training, rescoring, calibration, threshold selection, or exploratory stratification.

## Input audit

- Cohort: 141 queries, 1612 records (806 positive / 806 negative).
- Full score coverage: orthogonalized_interaction, additive_only, degree_matched_cnn, degree_matched_tcr_only_logistic.
- Similarity coverage: 112/141 queries (0.794326).
- All declared source SHA-256 values, row keys, labels, and prediction keys passed validation.

## Full-cohort metrics

| Method | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |
|---|---:|---:|---:|---:|
| orthogonalized_interaction | 0.710984 | 0.539774 | 0.531322 | 0.532768 |
| additive_only | 0.706562 | 0.523276 | 0.500973 | 0.498076 |
| degree_matched_cnn | 0.668910 | 0.463839 | 0.497450 | 0.483421 |
| degree_matched_tcr_only_logistic | 0.500000 | 0.500000 | 0.500000 | 0.500000 |

## Frozen primary comparison

Orthogonalized-interaction minus Additive-only; Macro AUPRC is the sole primary endpoint. Holm adjustment is applied only across the three secondary endpoints.

| Endpoint | Difference | 95% CI | Two-sided p | Holm p |
|---|---:|---:|---:|---:|
| query_level_macro_auprc | 0.004421894181 | [-0.044698182378, 0.053700033892] | 0.863413658634 | not applied |
| query_level_macro_auroc | 0.016497693560 | [-0.075810641277, 0.110682647058] | 0.734826517348 | 0.734826517348 |
| pooled_auroc | 0.034691735064 | [0.011173783241, 0.061221566518] | 0.037196280372 | 0.111588841116 |
| pooled_auprc | 0.030348563254 | [0.006533062366, 0.056992759460] | 0.076792320768 | 0.153584641536 |

## Similarity common-query comparison

Similarity is evaluable for 112/141 queries and is reported only on that common subset.

| Method | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |
|---|---:|---:|---:|---:|
| orthogonalized_interaction | 0.708921 | 0.539216 | 0.529757 | 0.532249 |
| additive_only | 0.696548 | 0.510940 | 0.500309 | 0.497040 |
| degree_matched_cnn | 0.674963 | 0.471766 | 0.500498 | 0.483485 |
| degree_matched_tcr_only_logistic | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| sequence_similarity | 0.732502 | 0.584616 | 0.513087 | 0.506084 |

No seen/unseen stratification or figure was generated.
