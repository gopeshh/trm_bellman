# Experiment 1B run5 additive audit bundle

This directory is an additive audit bundle for Experiment 1B run5. It does not replace or modify the original published tree at `artifacts/exp1b-run5-20260913/`.

Run the independent standard-library checker from this directory:

```sh
python3 checker/verify_exp1b_published.py .
```

The checker writes `checker/exp1b_independent_checks.json`. It recomputes the canonical result digest, all 35 chain links, Stage A manifest and Stage B payload digests, eight signed gaps, the equal-weight mean, the stored-replicate type-7 bootstrap interval, and the parity threshold checks. It stops at the first failure and does not use repository imports, Torch, or owner-tree files.

## Layout

| Path | Contents |
| --- | --- |
| `result/` | Published result, canonical digest sidecar, and Stage A provenance. |
| `study_documents/` | Registered Experiment 1B protocol, registry, and amendment used by the digest chain. |
| `signed_chain/` | Preserved authorization, admission, base-policy amendment, chain declaration, and PAR digest list. |
| `recovered_stage_a_manifests/` | Eight real Stage A `run_manifest.json` files recovered from the checkpoint directories. |
| `stage_a_checkpoints/` | Eight checkpoint byte streams required to recompute the manifest-to-checkpoint chain links. |
| `stage_b_payloads/` | Eight Stage B payloads. These are not run manifests. |
| `replay/` | Per-state replay exports, sidecars, independent recomputation, build identities, comparison rules, and replay source. |
| `checker/` | Independent checker and its machine-readable passing summary. |
| `logs/` | Checker proofs, replay/build logs, Buck test output, and source-verification evidence. |
| `CLAIM_MAP.md` | Claim-by-claim evidence map, including unsupported claims. |
| `MANIFEST.json` | Digest, size, source, role, and digest convention for every bundled file. |

## Digest conventions

Scientific JSON identities use `canonical_json_sha256_v1`: strict finite JSON, keys sorted, ASCII escaping, and separators `,` and `:` with no insignificant whitespace. Binary artifacts and ordinary bundle integrity use `file_bytes_sha256_v1`. `MANIFEST.json` uses the documented zeroed-self canonical convention because a file cannot contain its own ordinary byte digest.

## Scope

The bundle supports audit of the published result and an independent Stage B measurement replay from the eight existing checkpoints. It does not independently reproduce training, prove historical execution, establish coverage beyond the registered nominal bootstrap, or establish dataset and base-policy preparation beyond the identities recorded for this run.

The per-state exports are intentionally kept on disk and must not be added to git. See `replay/README.md` for their identity and regeneration procedure.
