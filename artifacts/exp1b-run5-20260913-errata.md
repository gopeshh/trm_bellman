# Experiment 1B run5 correction note

Issued 2026-09-14. This note applies to the published run5 artifact at `artifacts/exp1b-run5-20260913/`. It is additive. No signed input, result, payload, manifest, authorization, admission, schedule, seal, or checkpoint has been changed.

## Document bound by this note

The run5 admission binds the reduced-study amendment by canonical SHA-256:

```text
6e3c22a3d412c487498a713e96b2c7a0904188b7baabb8f64ec1e9586a8b3092
```

That digest identifies `configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json` as packaged by the run5 production binaries from source commit `2a14eb656bd983c37b60e92852083524348511d8`.

## Chronology correction

The amendment document retains `created_at_utc: 2026-09-09T00:00:00Z`. That field must not be read as the time when its final numerical centering contract was fixed. Scratch-run diagnostic output had been inspected before the numerical revision was committed on 2026-09-13 at 16:37:19 PT. The revision was informed by the observed float32 trainer-versus-reconstructed-advantage discrepancy.

After that revision, scratch chain c exercised the real fail-closed parity path. The run5 runtime authorization, base-policy amendment, and execution admission were then issued on 2026-09-13 at 23:50Z, 23:51Z, and 23:52Z. Run5 was executed against the resulting digest-bound chain. The amendment was fixed before run5, but it was not a pre-inspection registration relative to the earlier scratch diagnostics.

An earlier scratch chain is not affirmative validation: its instrumented build printed a parity failure instead of raising, so its passing audit was vacuous for this condition. The preserved run5 production PARs were later checked against commit `2a14eb6`; the executed path raises on parity failure.

## Scope of the no-inspection fields

The amendment's `outcome_evidence_inspected: false` is limited to run5 outcome evidence. No run5 outcome existed or was inspected when the run5 chain was bound. It does not claim that no output from any earlier scratch rehearsal had been inspected.

Likewise, `validation_data_inspected: false` and `test_data_opened: false` record that the underlying validation and test examples were not opened for selection or tuning in this run. They do not state that aggregate diagnostic values from a separate scratch run were unseen. These fields cannot support a broader claim that the numerical tolerance was chosen blind to all related empirical diagnostics.

## Numerical correction and observed errors

The inherited absolute tolerance is `1e-6`. The amendment adds a relative float32 allowance scaled by the registered advantage clip of 10:

```text
1e-6 + 4 * 2^-23 * 10 = 5.76837158203125e-6
```

The factor of four is an empirical engineering margin. It is not an operation-level rounding-error bound. The earlier source comments claiming a lower bound of `|value| * eps32`, mutually unsatisfiable float32 constants, and universal separation between numerical and algorithmic errors were wrong and have been withdrawn.

The run5 payloads report these finite-census maxima:

| Position | Seed | Constructed centering roundoff | Trainer centering defect | Trainer versus reconstructed parity |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2081976412 | `3.9255100626688163e-16` | `3.1568559638197454e-7` | `1.2773528705878334e-6` |
| 1 | 781025396 | `3.482014469441779e-16` | `1.4281121717015244e-7` | `1.3852601465913494e-6` |
| 2 | 1148619853 | `1.765534461072218e-16` | `7.03031724402505e-8` | `1.3282533277703124e-6` |
| 3 | 913535362 | `1.819294093140153e-16` | `3.000816842081789e-7` | `1.4138221722248545e-6` |
| 4 | 2143253519 | `3.477917588575428e-16` | `4.780727581949678e-8` | `1.507580318360624e-6` |
| 5 | 2279379379 | `4.903119502940478e-16` | `1.7142480861668748e-7` | `1.4686912663819385e-6` |
| 6 | 402312153 | `1.353016982778248e-16` | `2.2111107269781104e-7` | `1.3280378734492615e-6` |
| 7 | 2736405725 | `1.3344159524586459e-16` | `1.0981393968223808e-7` | `1.3759039774186022e-6` |

All eight parity observations exceed the old absolute-only threshold of `1e-6` and remain below the registered effective bound of `5.76837158203125e-6`. A one-epsilon allowance would have produced `2.1920928955078122e-6`, which also accepts the worst observation with 1.45x headroom. The factor of four is unnecessary for these observations. It remains unchanged because the published chain binds it.

These measurements are maxima over the fixed 128-state validation-bridge census. They are not uniform theorem-domain error bounds.
