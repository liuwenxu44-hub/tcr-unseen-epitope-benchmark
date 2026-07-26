# Final IEDB strict external test cohort

## Frozen construction protocol

- Tier: strict C
- Minimum positives per query: 1
- Deterministic query cap: 100
- Publication/reference: canonical PMID or DOI required
- Negatives: exact bipartite degree matching against the complete IEDB + VDJdb positive registry
- No model prediction, performance result, or manual record selection was used

## Cohort summary

- Queries: 141
- Positives: 806
- Putative negatives: 806
- Unique TCRs: 771
- Unique peptides: 141
- Unique HLAs: 33
- Unique publications: 34

## VDJdb overlap annotations

- TCR seen, record-level: 0.053350
- HLA seen, record-level: 0.950372
- TCR seen, unique-level: 0.050584
- HLA seen, unique-level: 0.727273

## Internal fold-train overlap annotations

| Fold | TCR seen records | HLA seen records | Unique TCR seen | Unique HLA seen |
|---|---:|---:|---:|---:|
| fold_1 | 0.009926 | 0.859801 | 0.007782 | 0.333333 |
| fold_2 | 0.009926 | 0.863524 | 0.007782 | 0.363636 |
| fold_3 | 0.011166 | 0.904467 | 0.009079 | 0.393939 |
| fold_4 | 0.012407 | 0.900744 | 0.009079 | 0.333333 |
| fold_5 | 0.006203 | 0.897022 | 0.003891 | 0.303030 |

## Positive antigen-source distribution

| Category | Records |
|---|---:|
| pathogen_or_nonself | 16 |
| self_antigen | 44 |
| tumor_associated | 215 |
| unknown_source | 531 |

## Independence and matching checks

| Check | Result |
|---|---:|
| mapped_publication_reference_missing_count | 0 |
| shared_vdjdb_publication_reference_count | 0 |
| exact_positive_interaction_overlap_with_vdjdb | 0 |
| development_peptide_overlap_count | 0 |
| maximum_similarity_to_development_peptide | 0.5 |
| maximum_similarity_threshold | 0.5 |
| negative_complete_positive_registry_overlap_count | 0 |
| duplicate_positive_interaction_count | 0 |
| duplicate_negative_interaction_count | 0 |
| positive_negative_interaction_overlap_count | 0 |
| tcr_degree_exact | True |
| query_degree_exact | True |
| degree_matching_rate | 1.0 |
| all_independence_checks_passed | True |

All checks passed. No training, model scoring, statistical testing, or plotting was performed.
