# Final cross-negative-resample paired statistics

The primary estimand is the equal-weight mean of the 25 train-resample by test-resample metric differences. Predictions were not concatenated into a single pseudo-dataset.

## Primary results

| Metric | Orthogonalized - Additive-only | 95% CI | Raw two-sided p | Holm p | Positive cells |
|---|---:|---:|---:|---:|---:|
| pooled_auroc | +0.00978399 | [-0.00111976, +0.02105300] | 0.19848015 | 0.3969603 | 25/25 |
| pooled_auprc | +0.00622190 | [-0.00253231, +0.01585180] | 0.3369663 | 0.3969603 | 25/25 |
| macro_auroc | +0.01064569 | [-0.00181284, +0.02337236] | 0.11088891 | 0.33266673 | 25/25 |
| macro_auprc | +0.01678515 | [+0.00646261, +0.02712312] | 0.00249975 | 0.0099990001 | 25/25 |

## Leave-one-train-resample-out

| Excluded | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| resample_1 | +0.00875231 | +0.00517586 | +0.00969700 | +0.01658787 |
| resample_2 | +0.00941690 | +0.00596901 | +0.01014307 | +0.01663362 |
| resample_3 | +0.01053949 | +0.00658538 | +0.01092640 | +0.01665151 |
| resample_4 | +0.00968258 | +0.00648199 | +0.01113513 | +0.01702968 |
| resample_5 | +0.01052868 | +0.00689726 | +0.01132686 | +0.01702307 |

## Leave-one-test-resample-out

| Excluded | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| resample_1 | +0.00955177 | +0.00599509 | +0.01030718 | +0.01626304 |
| resample_2 | +0.00944454 | +0.00551114 | +0.01064592 | +0.01677213 |
| resample_3 | +0.00989522 | +0.00631076 | +0.00962022 | +0.01614613 |
| resample_4 | +0.00963503 | +0.00640966 | +0.01069313 | +0.01712480 |
| resample_5 | +0.01039340 | +0.00688285 | +0.01196201 | +0.01761964 |

## Audit

- Prediction files: 250
- Prediction rows: 1,028,900
- Common queries: 168
- Maximum reproduction difference: 0.0
- Bootstrap: 10,000 fold-stratified query-cluster resamples
- Randomization: 10,000 paired query score swaps across all 25 combinations
- No training, rescoring, tuning, or data modification was performed.
