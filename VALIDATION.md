# Release validation

The v1.1.0 review adapter was run against the final materials inventory on 14 October 2026 (Asia/Shanghai), using Windows Python 3.12.14, NumPy 1.26.0, SciPy 1.15.3, scikit-learn 1.7.2 and joblib 1.5.3.

| Check | Observed result |
|---|---|
| File integrity | All 1,986 manifest-listed files matched their recorded SHA-256 values. |
| H1 results from frozen predictions | Fold/query/cell summaries, primary aggregation, external 25-member probability ensemble and six fixed exclusion summaries matched. |
| G2 results from frozen predictions | All 105 prediction files, 3,528 query results, 21 draws and seven recipe/S0 summaries matched. |
| Table provenance | 13,927 display cells checked; required source artifacts available; no missing selectors. Translations are checked only against source literals. |
| Model loading | 25 H1 interaction + 25 H1 analytic + 105 G2 interaction + 105 G2 analytic artifacts loaded and passed structural checks. |
| Limited scoring | Four fixed records from each of three checkpoints (12 records total): maximum probability discrepancy 1.11e-16; analytic probability exactly 0.5. |

The frozen metric comparisons use an absolute tolerance of 2e-12; the limited score check uses 1e-12. Windows long-path handling is supported. Commands print JSON to stdout and do not write logs, caches, fits, predictions or analysis outputs.

These checks validate the packaged reviewer entry point and its stated scope. They do not establish full raw-data reconstruction, retraining, exhaustive checkpoint rescoring, statistical inference or new blind external validation. Historical scientific artifacts retain their original bytes and provenance paths.
