# Confirmatory Sudoku dataset report

Date: 2026-08-03, America/Los_Angeles

Builder commit: `94a7199260a7f8467907a56b0d0331d6d0a11f60`

Dataset root: `data/iclr-confirmatory-sudoku4x4-v1`

This is an immutable input artifact. It is not a learned result and does not
authorize a confirmatory training run.

## Registered population

- 4 by 4 Sudoku;
- 6 to 8 empty cells;
- exactly one valid completion;
- 1,024 training records from seed `26080301`;
- 256 validation records from seed `26080302`;
- 512 held-out test records from seed `26080303`;
- unique inputs and input/solution records within every split;
- zero input and record overlap across splits.

These disjointness checks apply to puzzle inputs and complete input/solution
records, not to solution grids alone. A 4 by 4 Sudoku has only 288 valid
completed grids. This corpus contains 283 of them: 274 in train, 161 in
validation, and 226 in test. The splits share 154, 219, and 135 distinct
completions for train/validation, train/test, and validation/test respectively.
In record terms, 494 of 512 test puzzles (96.5%) have the same completion as at
least one training puzzle. The test split therefore measures constraint
satisfaction on unseen givens, not generalization to unseen solution grids.

The validation split is reserved for debug-seed pipeline checks and
implementation diagnostics. It is not used for confirmatory checkpoint,
hyperparameter, stopping, or test-result selection.

## Build command

```bash
buck2 run fbcode//buiksat_trm:build_iclr_confirmatory_4x4 --local-only -- \
  --output-dir data/iclr-confirmatory-sudoku4x4-v1 \
  --train-count 1024 \
  --validation-count 256 \
  --test-count 512 \
  --train-seed 26080301 \
  --validation-seed 26080302 \
  --test-seed 26080303 \
  --min-empty-cells 6 \
  --max-empty-cells 8 \
  --producer-repo-root .
```

The command ran from the Buck cell corresponding to this repository. The
release-facing command uses repository-relative paths; no workstation path is
part of the generated corpus.

## Immutable hashes

| Artifact | SHA-256 |
|---|---|
| Build configuration | `d5331bdd6c6602773ccc057287b268db29baa8d3e94fe8cfcc0985524c558db0` |
| Corpus manifest | `54dc6d07958736cab347a24f008915ab807a9e939267144e8ffe46d9c00f62d2` |
| Training manifest | `8def4f59387c1ab9466d043c40a7fdd3c7c670e2778b8d949295811ae7b6088a` |
| Validation manifest | `4644a3b1bb8b6e384896888154c9e56252368c1fc2f32962b089ade95a5bc1f2` |
| Held-out test manifest | `163a083a9f5744b7cc485663b269b89acc3103d9e1ec64c7e93f78e36fa79d40` |
| `CHECKSUMS.sha256` | `a69c93bf9b26dd31fbfe82a758d58bdca95f1d25669d5573d0c5e00a0551e239` |

Ordered record hashes:

- train: `94aa99e182b338a6238bd3a3db5c347bd28c0df07fb30b12e762ffc2ed2c1737`;
- validation: `2b3a5525d506c3078d9c1a1db184c1a48893cb97ee66c3128f8de516709bf21a`;
- test: `f7a13446e4ffcc5db8d5aa3d17641e7e7851b63c540c80dbe4b12d60b31fad24`.

Ordered input hashes:

- train: `2c914289f4272dbe437e37715dbc1d40b02de265fec7327a87a2102dbb853579`;
- validation: `81aad1fab292d12f5706180b5392793e0c6363c887e178891b9513f350fb6626`;
- test: `14262b1edbc445295f52fec49cf74ba9c99aaad7b9ef0d15b02139c240f512b0`.

## Generation and verification

The accepted records required:

| Split | Attempts | Rejected ambiguous | Rejected duplicate input | Rejected duplicate record |
|---|---:|---:|---:|---:|
| train | 1,140 | 116 | 0 | 0 |
| validation | 289 | 33 | 0 | 0 |
| test | 577 | 65 | 0 | 0 |

The verifier checks counts, shapes, dtypes, token ranges, clue/solution
agreement, Sudoku validity, single-solution status, ordered shared provenance
hashes, file hashes, checksums, and cross-split disjointness. A second full
build from a detached clean checkout of the builder commit produced the same
five manifest/configuration hashes, and `diff -qr` found no byte difference.

Builder runtime recorded in `build_config.json`:

- CPython `3.12.13+meta`;
- NumPy `2.2.1`.

Focused validation after the builder repair:

- dataset and provenance targets: 17 passed, 0 failed;
- isolated logging/checkpoint target: 24 passed, 0 failed;
- persistent checkpoint diagnostic target: 17 passed, 0 failed;
- builder, dataset, utility, and dataset-test type checks: passed;
- the `upi_trm_train` type-check target retains pre-existing errors outside the
  new training-pool helper;
- Python bytecode compilation and `git diff --check`: passed.

## Remaining gate

`corpus_manifest.json` describes raw generated data. It is deliberately not
the strict diagnostic input `{manifest_schema_version, dataset_provenance}`.
That second artifact requires the locked environment and action-mask
configuration and must be exported from the exact training provenance before
persistent diagnostics run.
