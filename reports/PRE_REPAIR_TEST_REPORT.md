# Pre-repair test report

Date: 2026-08-03, America/Los_Angeles

## Repository baseline

- Repository: `/home/buiksat/trm_bellman`
- Branch: `iclr-confirmatory-repair`
- Commit: `5cfcd6c31da8035b60d5dc178cd18a9e16ce0153`
- Remote: `https://github.com/gopeshh/trm_bellman.git`
- Initial `git status --short`: clean
- Buck cell path: `/data/repos/fbsource/fbcode/buiksat_trm`
- The Buck cell path is a symlink to the repository above. No tracked source was
  copied into fbsource.
- Host Python: `3.12.13+meta`
- Host Python cannot import `torch`; tests use Buck's hermetic dependency.
- GPU: NVIDIA PG509-210, driver 580.126.09, 81,920 MiB

## Complete declared unittest baseline

All 30 `python_unittest` targets declared in `BUCK` were passed explicitly to
one Buck invocation. No name or test-case filter was used.

Command shape:

```text
buck2 test @fbcode//mode/opt \
  fbcode//buiksat_trm:test_upi_trm_trainer_smoke \
  ... all 30 declared python_unittest targets ... \
  fbcode//buiksat_trm:test_constraint_aware_masking \
  -- --timeout=300
```

Result:

```text
Tests finished: Pass 263. Fail 0. Timeout 0. Fatal 0. Skip 0. Omit 0.
Infra Failure 0. Build failure 0
```

- Start: 2026-08-03 08:27:26 PDT
- Finish: 2026-08-03 08:39:30 PDT
- Log: `reports/pre_repair_unittests.log`
- Log SHA-256:
  `7d23aa4021018abe0e33ebd6aac42ae5d1064b158d9c6ccc8244460865a75fc2`
- Buck test session: `18014398699599549`

## Package-wide attempt

The initial command selected every target in the package, including legacy
analysis binaries and their generated type-check tests:

```text
buck2 test --local-only @fbcode//mode/opt 'fbcode//buiksat_trm:' \
  -- --timeout=300
```

This attempt was stopped after two hours. It had spent the final 80 minutes
compiling unrelated transitive C++ dependencies without producing another test
result. Exit code 141 records the deliberate interrupt, not a completed suite.
The partial log is retained at `reports/pre_repair_package_attempt.log` with
SHA-256
`5a1ef19813dbaf623cee6da5677a1c090eff7c440f6905e446fac446e2325219`.

Before interruption it found six deterministic type-check failures:

1. `plot_exp1_unroll_sensitivity-library-type-checking`
2. `make_paper_figures_exp1_split-library-type-checking`
3. `audit_phase4_paper_ready-library-type-checking`
4. `audit_exp1_paper_ready-library-type-checking`
5. `make_paper_figures_exp1-library-type-checking`
6. `make_paper_figures_exp1_final-library-type-checking`

The errors are retained verbatim in the log. They concern incorrect container
annotations, nullable values, and direct access to `torch.__version__` under
the internal stub. They do not invalidate the 263 passing runtime test cases,
but they remain pre-repair failures and will be rerun after correction.

A focused `--local-only` attempt was also interrupted after confirming that
local-only dependency linking was the source of the delay. Its three-line log
is retained at `reports/pre_repair_local_unittest_attempt.log`, SHA-256
`cbf43913e21d77418bae759943b3f629a38c6ebfd49b6fa082c9ecc6e77eb7cb`.

## Baseline conclusion

The runtime baseline is green: 263 of 263 declared unittest cases pass. The
package-wide static baseline is not green: six legacy script type-check targets
fail. No implementation source was modified before these results were
recorded.
