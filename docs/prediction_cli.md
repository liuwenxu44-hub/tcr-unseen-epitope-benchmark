# Frozen TCR Ranking CLI

`predict.py` ranks candidate TCR CDR3beta sequences for one peptide-HLA query. The first release supports only the frozen Orthogonalized-interaction Logistic ensemble: five negative resamples times five folds, for 25 checkpoints total.

## Input

The candidate CSV must contain `tcr_cdr3_beta`. Additional columns are ignored and no label is loaded into the model feature table.

```csv
tcr_cdr3_beta
CAAAQGYEQYF
CAAGLPIQDTEAFF
```

TCRs must be canonical 5-30 amino-acid sequences using the 20 standard amino acids, beginning with `C` and ending with `F` or `W`. The peptide must contain 8-15 standard amino acids. HLA must be a single classical, full-resolution value such as `HLA-A*02:01`. Normalization and validation reuse the frozen VDJdb/IEDB pipeline.

## Command

```bash
python predict.py \
  --epitope NLVPMVATV \
  --hla HLA-A*02:01 \
  --tcr-file candidate_tcrs.csv \
  --out ranked_tcrs.csv
```

The committed example can be run as:

```bash
python predict.py --epitope NLVPMVATV --hla HLA-A*02:01 --tcr-file examples/candidate_tcrs.csv --out ranked_tcrs.csv
```

## Output

`ranked_tcrs.csv` contains:

- `tcr_cdr3_beta`
- `epitope`
- `hla`
- `score`
- `rank`
- `contributing_model_count`

All records must receive exactly 25 checkpoint scores. The final score is their arithmetic mean. Rows are ordered by descending score, then lexicographically by TCR for ties. TCRs duplicated after whitespace/case normalization are retained once; the removed count is recorded in metadata.

The companion `ranked_tcrs.csv.metadata.json` records the raw input SHA-256, all checkpoint paths and SHA-256 values, command parameters, row counts, output SHA-256, model count, and code commit.

## Interpretation

The score is a **research ranking score**, not a calibrated probability. It must not be used for clinical decisions, patient treatment selection, or as a substitute for experimental binding or functional validation.

The command stops before writing predictions when a checkpoint is missing, a registered SHA-256 or file size differs, the 5x5 checkpoint grid is incomplete, or an input sequence/HLA is invalid.
