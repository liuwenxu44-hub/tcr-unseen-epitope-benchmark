# IEDB external exploratory stratified analysis

This analysis is restricted to the three exploratory stratifications frozen before external scoring. It does not replace the primary analysis over all 141 queries.

No model training, rescoring, tuning, calibration, threshold selection, plotting, or significance testing was performed.

## Frozen definitions

- HLA seen/unseen: record hla_seen_in_vdjdb annotation; HLA membership in the frozen complete VDJdb positive registry.
- TCR seen/unseen: record tcr_seen_in_vdjdb annotation; TCR membership in the frozen complete VDJdb positive registry.
- Antigen source: pre-scoring external_iedb_audit.yaml keyword mapping applied by the existing _source_category implementation.

## Sample sizes and Macro AUPRC comparison

| Stratification | Stratum | Queries | Positive | Negative | Publications | Macro eligible | Excluded | Orthogonalized - Additive-only Macro AUPRC (95% CI) | Similarity common eligible / coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hla_seen_vs_unseen | seen | 131 | 766 | 766 | 33 | 131 | 0 | 0.008481 [-0.044340, 0.060050] | 112 / 0.854962 |
| hla_seen_vs_unseen | unseen | 10 | 40 | 40 | 6 | 10 | 0 | -0.048753 [-0.155440, 0.049554] | 0 / 0.000000 |
| tcr_seen_vs_unseen | seen | 41 | 43 | 43 | 16 | 8 | 33 | -0.256522 [-0.405556, -0.098601] | 8 / 0.804878 |
| tcr_seen_vs_unseen | unseen | 141 | 763 | 763 | 34 | 128 | 13 | 0.004374 [-0.047028, 0.054102] | 101 / 0.794326 |
| antigen_source | tumor_associated | 5 | 215 | 215 | 3 | 5 | 0 | -0.010415 [-0.092550, 0.041146] | 3 / 0.600000 |
| antigen_source | self_antigen | 18 | 44 | 44 | 13 | 18 | 0 | -0.032499 [-0.188624, 0.117666] | 17 / 0.944444 |
| antigen_source | pathogen_or_nonself | 7 | 16 | 16 | 4 | 7 | 0 | 0.093702 [-0.098810, 0.286905] | 7 / 1.000000 |
| antigen_source | unknown_source | 111 | 531 | 531 | 22 | 111 | 0 | 0.005447 [-0.052444, 0.061028] | 85 / 0.765766 |

All four endpoint differences and confidence intervals, all absolute method metrics, and Similarity common-query descriptive metrics are stored in the CSV and JSON outputs.

Antigen-source strata with fewer than five queries are designated descriptive-only and are not interpreted. Categories were not merged after scoring.

No p values were calculated or reported.
