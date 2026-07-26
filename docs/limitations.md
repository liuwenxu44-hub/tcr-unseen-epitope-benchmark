# Limitations and Claim Boundaries

## Negative-label uncertainty

The negative class is putative. Registry exclusion and exact degree matching reduce known conflicts and marginal shortcuts, but absence from a positive registry is not experimental confirmation of non-binding or non-response.

## Restricted receptor representation

Only TRB CDR3 sequence is modeled. TRA sequence, V/J genes, paired-chain context, receptor expression, structural geometry, affinity, avidity, and cellular state are not represented.

## Restricted antigen-presentation scope

The project studies linear peptide-HLA records. It does not establish validity for non-peptide antigens, untyped presentation context, other MHC settings, or antigen processing and abundance in vivo.

## Dataset bias

VDJdb and IEDB are literature-derived resources with unevenly studied epitopes, HLA alleles, diseases, antigen sources, laboratories, and publications. Distance-aware splitting and degree matching do not remove ascertainment, annotation, publication, or population bias.

## Weak internal interaction evidence

The final internal Orthogonalized-interaction minus Additive-only Macro AUPRC difference is +0.016785, with 95% CI [0.006463, 0.027123] and Holm-adjusted p=0.009999. It is the only significant endpoint in the final hierarchical internal analysis; the other frozen confidence intervals cross zero. The magnitude is weak.

## External non-replication

The pre-specified IEDB external Macro AUPRC primary difference is +0.004422, with 95% CI [-0.044698, 0.053700] and two-sided p=0.863414. The significant internal interaction gain was not reproduced externally.

Internal and external evidence are therefore reported separately. They must not be combined into a single significant conclusion or used to claim stable superiority over Additive-only.

## Model-comparison limits

The CNN did not establish stable superiority over the shallow models across the frozen internal and external evidence. Architecture complexity should not be interpreted as stronger biological validity.

## Incomplete control coverage

Sequence Similarity was evaluable for 112/141 external queries because same-HLA training references were required and cross-HLA fallback was prohibited. Its descriptive results do not cover the full external cohort.

## No clinical or experimental substitution

These models must not be used for clinical decisions, patient treatment selection, diagnosis, safety assessment, or replacement of experimental validation. Scores do not establish binding or function.

## Claim boundary

The frozen evidence does not support high-performance, state-of-the-art, clinical-utility, or broad biological-generalization claims. It supports only a weak internal interaction signal that did not replicate as a significant primary effect in the independent IEDB cohort.
