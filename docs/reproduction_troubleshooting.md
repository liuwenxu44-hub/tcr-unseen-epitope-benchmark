# Reproduction troubleshooting

## Python version

Use CPython 3.10.13. Binary compatibility errors from NumPy, SciPy, scikit-learn, or PyTorch usually indicate that a different Python or package version was installed.

## PyTorch installation

The pinned PyTorch version may require a platform-specific wheel source. If the default installation is unsuitable for the local CUDA runtime, install the official PyTorch 2.1.0 build for the target platform and keep all other pinned versions unchanged.

## Raw database downloads

VDJdb and IEDB can update their export endpoints. Do not silently substitute a newer export. Record the retrieval date, source metadata, file size, and SHA-256, then compare them with the committed download manifests before continuing.

## Checkpoint verification

The prediction and external-scoring commands stop when a registered checkpoint is missing or its file size or SHA-256 differs. Restore the exact registered artifact rather than bypassing this check.

## Deterministic reconstruction

All stochastic stages use fixed seeds. When a generated file differs, first compare:

- the input file hashes;
- the configuration file hash;
- Python and package versions;
- locale and line-ending settings;
- the selected CPU/GPU execution path.

## Release file integrity

Run:

```bash
sha256sum -c RELEASE_MANIFEST.sha256
```

Any mismatch means the local release differs from the published snapshot.
