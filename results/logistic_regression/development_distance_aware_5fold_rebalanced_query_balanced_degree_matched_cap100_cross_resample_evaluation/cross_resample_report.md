# Cross-negative-resample Logistic evaluation

All 50 fixed checkpoints were evaluated across all five corresponding test resamples without retraining or tuning.

## Diagonal results

Each cell is Orthogonalized / Additive-only / difference.

| Resample | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| resample_1 | 0.515158 / 0.500058 / +0.015100 | 0.511643 / 0.500196 / +0.011446 | 0.518251 / 0.501280 / +0.016971 | 0.538707 / 0.520423 / +0.018284 |
| resample_2 | 0.512422 / 0.500008 / +0.012414 | 0.509193 / 0.499312 / +0.009882 | 0.514726 / 0.500879 / +0.013847 | 0.534045 / 0.513972 / +0.020073 |
| resample_3 | 0.508250 / 0.500370 / +0.007880 | 0.503988 / 0.499815 / +0.004173 | 0.510926 / 0.495541 / +0.015385 | 0.531017 / 0.509837 / +0.021181 |
| resample_4 | 0.511137 / 0.500450 / +0.010687 | 0.504529 / 0.499708 / +0.004821 | 0.511613 / 0.502391 / +0.009222 | 0.530395 / 0.514985 / +0.015410 |
| resample_5 | 0.504491 / 0.499850 / +0.004641 | 0.501827 / 0.499798 / +0.002029 | 0.502524 / 0.498673 / +0.003851 | 0.525745 / 0.511901 / +0.013844 |

## Orthogonalized OOF metric matrices

### pooled_auroc

| Train \ Test | resample_1 | resample_2 | resample_3 | resample_4 | resample_5 |
|---|---:|---:|---:|---:|---:|
| resample_1 | 0.515158 | 0.516466 | 0.513591 | 0.514346 | 0.510733 |
| resample_2 | 0.510439 | 0.512422 | 0.511817 | 0.512881 | 0.508125 |
| resample_3 | 0.507733 | 0.507663 | 0.508250 | 0.506971 | 0.504577 |
| resample_4 | 0.511215 | 0.510869 | 0.509101 | 0.511137 | 0.508717 |
| resample_5 | 0.509119 | 0.508620 | 0.505312 | 0.507037 | 0.504491 |

### pooled_auprc

| Train \ Test | resample_1 | resample_2 | resample_3 | resample_4 | resample_5 |
|---|---:|---:|---:|---:|---:|
| resample_1 | 0.511643 | 0.513693 | 0.509978 | 0.509499 | 0.507153 |
| resample_2 | 0.506568 | 0.509193 | 0.507756 | 0.506343 | 0.502897 |
| resample_3 | 0.505518 | 0.507189 | 0.503988 | 0.503209 | 0.502567 |
| resample_4 | 0.505983 | 0.507366 | 0.504076 | 0.504529 | 0.502497 |
| resample_5 | 0.505136 | 0.505491 | 0.502422 | 0.501498 | 0.501827 |

### macro_auroc

| Train \ Test | resample_1 | resample_2 | resample_3 | resample_4 | resample_5 |
|---|---:|---:|---:|---:|---:|
| resample_1 | 0.518251 | 0.517002 | 0.516842 | 0.516511 | 0.509977 |
| resample_2 | 0.513031 | 0.514726 | 0.514962 | 0.512658 | 0.508512 |
| resample_3 | 0.510535 | 0.508590 | 0.510926 | 0.508717 | 0.503313 |
| resample_4 | 0.512676 | 0.510391 | 0.510167 | 0.511613 | 0.506467 |
| resample_5 | 0.510697 | 0.505405 | 0.507287 | 0.505529 | 0.502524 |

### macro_auprc

| Train \ Test | resample_1 | resample_2 | resample_3 | resample_4 | resample_5 |
|---|---:|---:|---:|---:|---:|
| resample_1 | 0.538707 | 0.536729 | 0.536403 | 0.536625 | 0.531081 |
| resample_2 | 0.532611 | 0.534045 | 0.531145 | 0.529240 | 0.528164 |
| resample_3 | 0.532994 | 0.530225 | 0.531017 | 0.527036 | 0.525935 |
| resample_4 | 0.533628 | 0.531222 | 0.529551 | 0.530395 | 0.526942 |
| resample_5 | 0.532975 | 0.525906 | 0.529533 | 0.525941 | 0.525745 |

## Twenty off-diagonal combinations

### additive_only

- pooled_auroc: 0.500073 +/- 0.000258 (range 0.499598 to 0.500612)
- pooled_auprc: 0.499682 +/- 0.000266 (range 0.499120 to 0.500151)
- macro_auroc: 0.500120 +/- 0.001872 (range 0.496370 to 0.502929)
- macro_auprc: 0.514152 +/- 0.002312 (range 0.510581 to 0.519380)

### orthogonalized_interaction

- pooled_auroc: 0.509767 +/- 0.003040 (range 0.504577 to 0.516466)
- pooled_auprc: 0.505842 +/- 0.003029 (range 0.501498 to 0.513693)
- macro_auroc: 0.510464 +/- 0.003937 (range 0.503313 to 0.517002)
- macro_auprc: 0.530694 +/- 0.003514 (range 0.525906 to 0.536729)

## Worst case by training resample

Each Orthogonalized, Additive-only, and difference value is independently minimized over the five test resamples.

| Train resample | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| resample_1 | 0.510733 / 0.500021 / +0.010552 | 0.507153 / 0.499728 / +0.007002 | 0.509977 / 0.498545 / +0.007058 | 0.531081 / 0.515475 / +0.011700 |
| resample_2 | 0.508125 / 0.499598 / +0.008528 | 0.502897 / 0.499120 / +0.003506 | 0.508512 / 0.498267 / +0.008320 | 0.528164 / 0.512607 / +0.015136 |
| resample_3 | 0.504577 / 0.500046 / +0.003965 | 0.502567 / 0.499513 / +0.002614 | 0.503313 / 0.495541 / +0.003417 | 0.525935 / 0.509837 / +0.013762 |
| resample_4 | 0.508717 / 0.499658 / +0.008940 | 0.502497 / 0.499454 / +0.002740 | 0.506467 / 0.497723 / +0.004255 | 0.526942 / 0.512443 / +0.012743 |
| resample_5 | 0.504491 / 0.499850 / +0.004641 | 0.501498 / 0.499542 / +0.001877 | 0.502524 / 0.496370 / +0.003851 | 0.525745 / 0.510581 / +0.012521 |

## Positive Orthogonalized interaction gains

- macro_auprc: 25/25
- macro_auroc: 25/25
- pooled_auprc: 25/25
- pooled_auroc: 25/25

## Audit

All checkpoint and test CSV SHA-256 values matched. Every test partition passed exact TCR/query degree, label, known-positive conflict, duplicate, and query-leakage audits. The 5 diagonal cells reproduce the prior OOF results. No checkpoint or data file was modified; no model was trained and no bootstrap was run.
