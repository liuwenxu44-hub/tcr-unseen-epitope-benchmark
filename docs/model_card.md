# Model Card: TCR-pMHC Interaction Models

## Project and intended purpose

This project benchmarks whether sequence-based models can prioritize candidate TCR-pMHC interactions while controlling marginal TCR and query shortcuts. It is a research benchmark and evidence-audit pipeline, not a clinical product or an experimental binding assay.

Appropriate uses include retrospective method comparison, non-clinical hypothesis generation, and candidate prioritization for subsequent experimental testing. Scores must be interpreted inside the documented data scope and evaluation protocol.

## Inputs

| Field | Meaning |
|---|---|
| TCR CDR3β | Amino-acid sequence of the beta-chain complementarity-determining region. |
| peptide | Linear peptide epitope amino-acid sequence. |
| HLA | Standardized peptide-presenting HLA allele. |

The benchmark is limited to records for which these fields can be represented under the frozen preprocessing rules.

## Outputs and interpretation

The learned classifiers emit a logit or probability-like score, and the external protocol averages probabilities across the registered checkpoints. Sequence Similarity emits a similarity score and is a descriptive control rather than a calibrated probability model. A higher score means more positive-like under this constructed benchmark; it does not establish binding, functional response, causality, or clinical utility.

## Models

- **Orthogonalized-interaction Logistic:** primary interaction model. It retains additive sequence and HLA effects and adds a double-centered amino-acid composition interaction.
- **Additive-only Logistic:** primary comparator without TCR-peptide interaction features.
- **CNN:** two-tower sequence model with separate TCR and peptide encoders plus an HLA embedding.
- **TCR-only Logistic:** shortcut control using no peptide or HLA information.
- **Sequence Similarity:** non-parametric descriptive control using same-HLA training references only.

## Training and evaluation data

The core internal benchmark uses the VDJdb Development positives and a 5-fold distance-aware split. Peptides connected above normalized Levenshtein similarity 0.5 remain in one component, and the frozen split verifies that the maximum cross-partition similarity is no greater than 0.5. Query-balanced sampling caps each query at 100 positives before negative construction.

Across the internal out-of-fold test partitions, the frozen table contains 168 queries, 10289 positives, and 10289 putative negatives.

### Degree-matched putative negatives

Negatives are constructed independently within each fold partition by exact bipartite degree matching. Every TCR has the same positive and negative degree, every peptide-HLA query has the same positive and negative degree, candidate negative interactions are absent from the complete positive registry, and duplicate or cross-partition interactions are prohibited. This controls marginal frequency shortcuts but does not turn a putative negative into an experimentally confirmed non-interaction.

## Internal validation results

The following out-of-fold metrics are internal results only:

| Model | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |
|---|---:|---:|---:|---:|
| Orthogonalized-interaction Logistic | 0.581916 | 0.542541 | 0.538023 | 0.544736 |
| Additive-only Logistic | 0.527917 | 0.496527 | 0.500526 | 0.502006 |
| CNN | 0.627218 | 0.520153 | 0.520219 | 0.525433 |
| TCR-only Logistic | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| Sequence Similarity | 0.562582 | 0.519568 | 0.508901 | 0.505073 |

The frozen cross-negative-resample comparison of Orthogonalized-interaction minus Additive-only is:

| Endpoint | Difference | 95% CI | Raw p | Adjusted p |
|---|---:|---:|---:|---:|
| Macro AUPRC | +0.016785 | [0.006463, 0.027123] | 0.002500 | 0.009999 |
| Macro AUROC | +0.010646 | [-0.001813, 0.023372] | 0.110889 | 0.332667 |
| Pooled AUPRC | +0.006222 | [-0.002532, 0.015852] | 0.336966 | 0.396960 |
| Pooled AUROC | +0.009784 | [-0.001120, 0.021053] | 0.198480 | 0.396960 |

Only the Macro AUPRC endpoint has a frozen confidence interval excluding zero after the registered adjustment. The effect is weak and is not evidence of high predictive performance.

## IEDB external validation results

The strict publication-independent IEDB cohort contains 141 queries, 806 positives, 806 degree-matched putative negatives, and 34 publications. Its peptides are unseen in VDJdb Development and satisfy the frozen distance threshold.

Full-cohort external metrics are:

| Model | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |
|---|---:|---:|---:|---:|
| Orthogonalized-interaction Logistic | 0.710984 | 0.539774 | 0.531322 | 0.532768 |
| Additive-only Logistic | 0.706562 | 0.523276 | 0.500973 | 0.498076 |
| CNN | 0.668910 | 0.463839 | 0.497450 | 0.483421 |
| TCR-only Logistic | 0.500000 | 0.500000 | 0.500000 | 0.500000 |

The pre-specified Orthogonalized-interaction minus Additive-only comparison is:

| Endpoint | Difference | 95% CI | Raw p | Adjusted p |
|---|---:|---:|---:|---:|
| Macro AUPRC | +0.004422 | [-0.044698, 0.053700] | 0.863414 | not adjusted |
| Macro AUROC | +0.016498 | [-0.075811, 0.110683] | 0.734827 | 0.734827 |
| Pooled AUPRC | +0.030349 | [0.006533, 0.056993] | 0.076792 | 0.153585 |
| Pooled AUROC | +0.034692 | [0.011174, 0.061222] | 0.037196 | 0.111589 |

The external Macro AUPRC primary endpoint did not reproduce the internally significant gain. Internal and external results remain separate and must not be merged into one significance claim.

Sequence Similarity was available for 112/141 external queries (coverage 0.794326) and remains a secondary descriptive control.

## Appropriate use

- Audited benchmarking of TCR-peptide-HLA ranking methods within the frozen scope.
- Research prioritization where every candidate is subsequently checked experimentally.
- Methodological studies of shortcut control, negative construction, and split robustness.

## Prohibited or not recommended use

- Clinical diagnosis, patient stratification, treatment selection, or safety-critical decisions.
- Replacement of binding, activation, cytotoxicity, or other wet-lab assays.
- Interpreting the score as a calibrated probability of physical binding or immune response.
- Applying the models to receptor chains, antigen formats, or MHC settings outside the documented inputs without a new independent validation.
- Further tuning, calibration, threshold selection, or model selection on the frozen IEDB cohort.

## Reproducibility and checkpoint provenance

The frozen environment uses Python 3.10.13, PyTorch 2.1.0 built for CUDA 12.1, and a hash-locked dependency file containing 42 packages. Installation and verification commands are documented in `docs/reproducible_environment.md`.

Model artifacts are traced through `results/external/iedb/external_checkpoint_registry.csv`, the internal result registry, and their freeze manifests. Each registered checkpoint includes its path and SHA-256. This card does not create, select, or alter checkpoints.

## Data and result traceability

Frozen data manifests record source files, split membership, negative-construction invariants, seeds, and SHA-256 values. Frozen result registries identify analysis status and source commits. Every number displayed in this card is registered in `results/final/model_card_provenance.json` with its source path, commit, SHA-256, and field selector.

## Limitations

See `docs/limitations.md`. Its restrictions are part of this model card and govern all interpretation.
