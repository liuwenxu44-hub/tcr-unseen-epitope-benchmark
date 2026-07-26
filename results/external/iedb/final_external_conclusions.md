# Frozen IEDB external validation conclusions

**IEDB external analysis status: FROZEN.**

## External primary result

The pre-specified external Macro AUPRC comparison (Orthogonalized - Additive-only) was +0.004422, 95% CI [-0.044698, 0.053700], p=0.863414. The internally significant interaction gain was not reproduced externally.

Pooled AUPRC and AUROC differences were positive (+0.030349 and +0.034692), but were not significant after Holm correction (p=0.153585 and p=0.111589).

All four TCR-only Logistic metrics were 0.500000.

Sequence Similarity covered 112/141 queries and is only a secondary descriptive control.

## Exploratory strata

HLA, TCR, and antigen-source analyses remain exploratory and do not establish a stable, generalizable subgroup conclusion.

The TCR-seen subset had 8 macro-evaluable queries (33 excluded) and must not be emphasized.

## Internal and external evidence

The internal cross-negative-resample result remains a separate frozen finding: Macro AUPRC interaction gain +0.016785, 95% CI [0.006463, 0.027123], Holm p=0.009999, with positive gain in 25/25 combinations.

The internal significant result and the external non-replication must not be merged into a single significance claim.

## Claim boundary

The external results do not support claims of high performance, SOTA, or stable superiority over Additive-only.

No subsequent tuning, calibration, threshold selection, or model modification may be based on this IEDB external dataset. Formal figures remain paused.

Every conclusion and displayed value is registered with its source path, source commit, SHA-256, and selector in `external_result_registry.json`.
