# Repository archive

Run `scripts/build_artifact_zip.sh` from any directory to create:

- `artifact/upi_trm_repository.zip`
- `artifact/upi_trm_repository.zip.sha256`

The archive contains every tracked file plus non-ignored new file in the
working tree. It excludes `.git`, ignored checkpoints, temporary logs, and the
archive itself. The builder uses a temporary staging tree to remove local user,
machine, and prior-venue identifiers. Raw provenance files in the working tree
are not rewritten. Venue-specific historical path components are renamed only
inside the archive.

The archive includes `artifact/SHA256SUMS` for its staged files. The build
normalizes timestamps and ZIP metadata, so identical inputs produce identical
archive bytes.

Historical result files are included for provenance, but their presence does
not validate the corresponding claims. In particular, the retained 57.4%
UPI-TRM evaluation is in-sample, and the external baseline artifacts do not
contain the pool hashes required by the repaired aggregator. See the root
`README.md` and `AUDIT_REPORT.md` before using any result.
