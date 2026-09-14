# Buck test capture

All commands ran from `/data/repos/fbsource` with `@fbcode//mode/opt`, without `--local-only`.

| Log | Command scope | Buck exit | Tpx summary |
| --- | --- | ---: | --- |
| `buck-test-exp1b.log` | `fbcode//buiksat_trm:test_policy_improvement_exp1b` | 64 | Pass 336, Fail 0, Skip 4 |
| `buck-test-diagnostics.log` | `fbcode//buiksat_trm:test_policy_improvement_exp1_diagnostics` | 0 | Pass 46, Fail 0, Skip 0 |
| `buck-tests.log` | Both targets together | 64 | Pass 382, Fail 0, Skip 4 |

The four skips are printed in the logs and are explicitly gated on unavailable restored-checkpoint or regenerated-corpus fixtures. No test failed, timed out, crashed, had an infrastructure failure, or had a build failure.
