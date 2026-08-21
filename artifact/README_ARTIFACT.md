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

Historical learned-result files are excluded from the current branch because
they were produced by superseded implementations and protocols. Repository
archives built from this branch contain source and registered configuration,
not empirical evidence. See the root `README.md` before interpreting an
archive.
