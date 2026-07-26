# Frozen Main Results Table

Core analysis: distance-aware five-fold split, query-balanced cap=100, and degree-matched negatives.

## Core OOF Performance

| Method | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC | Queries | Samples |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.525433 | 0.520219 | 0.520153 | 0.627218 | 168 | 20578 |
| Additive-only Logistic | 0.502006 | 0.500526 | 0.496527 | 0.527917 | 168 | 20578 |
| Orthogonalized-interaction Logistic | 0.544736 | 0.538023 | 0.542541 | 0.581916 | 168 | 20578 |
| Sequence Similarity | 0.505073 | 0.508901 | 0.519568 | 0.562582 | 153 | 18644 |
| TCR-only Logistic | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 168 | 20578 |

Sequence Similarity is evaluated on the 153/168 queries with same-HLA training references; other methods use all 168 queries.

## TCR-only Versus Random Level

| Metric | Observed | Null mean | Difference | Holm p |
|---|---:|---:|---:|---:|
| Pooled Auroc | 0.500000 | 0.500000 | +0 | 1 |
| Pooled Auprc | 0.500000 | 0.500000 | +0 | 1 |
| Macro Auroc | 0.500000 | 0.500000 | +1.11022e-15 | 1 |
| Macro Auprc | 0.500000 | 0.500000 | +1.11022e-15 | 1 |

## Five Independent Negative Resamples

| Method/comparison | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| Additive-only Logistic | 0.500147 +/- 0.000253 [0.499850, 0.500450] | 0.499766 +/- 0.000316 [0.499312, 0.500196] | 0.499753 +/- 0.002714 [0.495541, 0.502391] | 0.514224 +/- 0.003991 [0.509837, 0.520423] |
| Orthogonalized-interaction Logistic | 0.510292 +/- 0.004085 [0.504491, 0.515158] | 0.506236 +/- 0.004043 [0.501827, 0.511643] | 0.511608 +/- 0.005849 [0.502524, 0.518251] | 0.531982 +/- 0.004791 [0.525745, 0.538707] |
| Orthogonalized - Additive-only | 0.010145 +/- 0.004044 [0.004641, 0.015100] | 0.006470 +/- 0.004004 [0.002029, 0.011446] | 0.011855 +/- 0.005329 [0.003851, 0.016971] | 0.017758 +/- 0.003090 [0.013844, 0.021181] |

## Complete 5x5 Cross-resample Evaluation

| Metric | Mean difference | SD | Range | Positive combinations |
|---|---:|---:|---:|---:|
| Pooled Auroc | +0.009784 | 0.003229 | [+0.003965, +0.016421] | 25/25 |
| Pooled Auprc | +0.006222 | 0.003155 | [+0.001877, +0.013907] | 25/25 |
| Macro Auroc | +0.010646 | 0.004195 | [+0.003417, +0.018297] | 25/25 |
| Macro Auprc | +0.016785 | 0.002850 | [+0.011700, +0.021181] | 25/25 |

## Final Paired Inference Across 25 Combinations

| Metric | Mean difference | 95% CI | Holm p | Positive combinations |
|---|---:|---:|---:|---:|
| Pooled Auroc | +0.009784 | [-0.001120, +0.021053] | 0.396960 | 25/25 |
| Pooled Auprc | +0.006222 | [-0.002532, +0.015852] | 0.396960 | 25/25 |
| Macro Auroc | +0.010646 | [-0.001813, +0.023372] | 0.332667 | 25/25 |
| Macro Auprc | +0.016785 | [+0.006463, +0.027123] | 0.009999 | 25/25 |

All values are generated from committed source files. Exact source paths, commits, SHA-256 values, and JSON pointers are retained in `main_results_table.csv` and `result_registry.json`.
