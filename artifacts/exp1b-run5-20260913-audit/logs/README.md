# Audit logs

- `checker-real-pass.log` records the independent checker passing the real bundle.
- `checker-real-pass-optimized.log` records the same 187 checks passing under `python3 -O`.
- `checker-mutated-result.log` records the checker stopping with exit code 1 after one result byte was changed in a temporary copy.
- `mutation-byte-diff.log` is the `cmp -l` proof that exactly one byte differed.
- `buck-test-exp1b.log` records 336 passes, 0 failures, and 4 explicit artifact-gated skips.
- `buck-test-diagnostics.log` records 46 passes, 0 failures, and 0 skips.
- `buck-tests.log` records the combined 382-pass run.
- `replay/` contains the replay producer, consumer, and build logs.
- `source-verification/` contains the preserved-PAR embedded-source inventory, report, and verifier.
