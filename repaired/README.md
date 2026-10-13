# Repaired H1/G2 portable review adapter

Place this directory at `repaired/` beside the extracted `materials/` folder.
Use Python 3.10–3.12. The historical analysis used Python 3.10.13; that
environment remains the reference for full scientific reconstruction.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r repaired/requirements-review.txt
python repaired/review.py verify-integrity
python repaired/review.py verify-results
python repaired/review.py verify-provenance
python repaired/review.py load-models
python repaired/review.py score-smoke --records 4
```

For another extraction location, put `--materials /path/to/materials` before the
command. `--group H1` or `--group G2` after `verify-results`, `load-models`, or
`score-smoke` restricts that command to the named experiment. The material
inventory is discovered as `MATERIALS_MANIFEST.json`, `DERIVED_FILES.json`, or
`manifest.json`; `--manifest /path/to/inventory.json` overrides discovery.

`verify-results` uses the frozen H1 aggregation functions and the frozen G2
metric/aggregation functions on **existing predictions**. It compares H1 primary,
all 25 cells, per-query values, three fixed masks in both cohorts, and the external
25-member probability ensemble. It separately compares G2 per-query, draw,
recipe, and paired S0 summaries. Absolute numeric comparison tolerance is 2e-12.
H1 is the original-positive primary repair; G2 is a separate common-positive
sensitivity grid. Neither is fresh blind validation or a new inferential result.

`load-models` loads all 25 H1 interaction and 25 H1 analytic artifacts and all 105
G2 interaction and 105 G2 analytic artifacts. It checks estimator dimensions,
finite parameters, binary classes, and exact-zero analytic coefficients.
Joblib loading executes Python serialization code: use only this trusted,
hash-verified release. The adapter checks each member against the material
inventory before reading it; SHA-256 is an identity check, not a signature.

`score-smoke` explicitly scores a few frozen records for one reused H1 member,
one newly fitted H1 member, and one G2 member. It checks probabilities against
existing predictions (1e-12 tolerance), and analytic Additive probability exactly
0.5. The exact frozen feature constants/functions are selected from hashed source
via an AST whitelist, avoiding unrelated legacy CNN/training imports. No source
function is rewritten; no HLA mapping is fitted. This is a limited runtime
roundtrip, not full checkpoint rescoring.

`verify-provenance` verifies available CSV/JSON source selectors and full-precision
values for Tables 1–3/S1–S4, checks literal editorial labels, and checks source
literals for translated design notes. Missing source artifacts produce an explicit
`INCOMPLETE` report and exit 2. Human translation/editorial equivalence is outside
this check. Historical paths are only identifiers; the adapter resolves controlled
package paths or exact frozen hashes and never opens an old absolute path.

All commands are read-only and print compact JSON to standard output. They do
not create outputs, caches, logs, fits, samples, CIs, P values, or new scientific
analyses. Integrity verifies manifest members, not unlisted extra files. Training,
raw-provider reconstruction, full model rescoring, and the legacy archived
executors remain advanced, unverified workflows; no portable training command is
claimed here. Installation/model/result validation actually performed for this
release must be documented separately in its validation receipt.
