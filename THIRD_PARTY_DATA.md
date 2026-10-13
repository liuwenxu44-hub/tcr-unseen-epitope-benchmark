# Source database and derived-data terms

Original project software and its documentation use the repository MIT License. That license does not relicense third-party database records.

## VDJdb frozen snapshot

The benchmark used content version 2026-06-03. The frozen source archive has SHA-256 9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1. Its included LICENSE is GNU Affero General Public License version 3, reproduced in licenses/VDJDB_2026-06-03_AGPL-3.0.txt. The tagged release script copies that license into the database archive. The official asset with matching bytes is [VDJdb 2026-06-03](https://github.com/antigenomics/vdjdb-db/releases/tag/2026-06-03); the [tagged license](https://github.com/antigenomics/vdjdb-db/blob/659469dc3c54af4a8b8dc8b905217c91e7045bc8/LICENSE) and [release script](https://github.com/antigenomics/vdjdb-db/blob/659469dc3c54af4a8b8dc8b905217c91e7045bc8/release.sh) establish the historical terms.

VDJdb-derived scientific inputs and projections retain those source terms and attribution; they are supplied with readable data, source identities and original transformation source. The repository MIT license does not replace the VDJdb license. Cite VDJdb and the original publications associated with retained reference identifiers. The current upstream master license differs; the license bundled with the exact frozen bytes is retained here.

## IEDB frozen exports

IEDB database data are available for sharing and adaptation under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain IEDB/source attribution, link the license and identify modifications, without implying endorsement or imposing additional restrictions. The provider's [official licensing statement](https://discuss.iedb.org/t/iedb-coretrustseal-approved-application/205) distinguishes database data from separately licensed IEDB analysis tools. This package redistributes no IEDB analysis-tool binary or raw provider export.

The benchmark used the July 2026 frozen exports recorded in the acquisition manifests. A current IEDB download does not establish those historical bytes.

## Public projection

Provenance tables under materials/provenance/ preserve every source row, interaction key, retained evidence flag, source record ID, source hash and selector. Files omit internal path columns, raw provider payloads, study donor/subject pseudonyms and unnecessary cohort context. Public source publication/assay references are retained in the compact crosswalk. See DATA_FIELD_POLICY.json, export_public_provenance.py and PUBLIC_EXPORT_RECEIPT.json for exact transformations and verification. These projections do not alter benchmark input graphs, labels, degrees, model bytes, frozen predictions or reported results.
