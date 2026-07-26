# IEDB external cohort blind-scoring audit

## Scope

The frozen external cohort was scored without loading labels into any model feature table. No performance metric, statistical test, training, tuning, calibration, threshold selection, or figure was produced.

## Inputs

- Cohort interactions: 1612
- Cohort queries: 141
- Registered model checkpoints: 60
- Similarity fold references: 5
- All registered paths, sizes, and SHA-256 values matched the frozen registry.

## Persisted predictions

- Raw prediction rows: 104780
- Ensemble prediction rows: 8060
- Similarity evaluable queries: 112 / 141
- Maximum ensemble arithmetic-mean error: 0.000e+00

## Validation

- All four model methods scored every interaction.
- All probabilities were finite and within [0,1].
- Similarity used same-HLA fold references only; queries without a valid reference were explicitly unavailable.
- Two complete scoring runs produced identical SHA-256 values for every output.
- Prediction files contain no label column.
