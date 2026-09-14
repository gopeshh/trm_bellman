# Experiment 1B, run5

The published run. Stage A sealed eight seeds on two A100s between 2026-09-13
20:25 and 2026-09-14 03:41; Stage B ran positions 0 through 7 sequentially to
05:02; the audit passed at 05:02:18.

## Result

    mean_signed_gap  14.248703798855473
    interval         [12.185185177643433, 16.739775084598428]   95%, two-sided percentile
    document_sha256  71f37c6f31d0bb9951b884f5f275e3b9c9bf07d46a836a1da37d9e80c6454e74

`document_sha256` is `sha256(canonical_json_bytes(result))` and matches
`result/exp1b_result.sha256`. Every digest in this tree follows that
convention. Hashing the file bytes gives a different value and makes a valid
document look absent.

The auditor re-derived all eight checks from the published document plus the
registered protocol, registry, amendment, parent populations and the execution
admission:

    document_canonical_digest
    registered_identity_and_access_flags
    census_member_ordering_rederived_from_registered_population
    role_scoped_admission_binding
    study_document_bindings
    per_seed_bound_algebra
    secondary_diagnostics
    frozen_bootstrap_interval

Full auditor output is in `result/audit.txt`.

## Chain

Producer commit `2a14eb656bd983c37b60e92852083524348511d8`, which carries the
trainer-parity fix. Digests, all canonical:

    runtime authorization        203621ee0c5eb3d3b042b6d447f2832c4f37c8c99d8ed0f628480f1b30326ab4
    base-policy amendment        4f12873ac1413860378c72532d6dc5eb6e8793d316bbe5c3025df3b7161a8161
    reduced-study amendment      6e3c22a3d412c487498a713e96b2c7a0904188b7baabb8f64ec1e9586a8b3092
    execution admission          2bbeacf2675e9cb9f3b75c8e475f6ef1620e6d6f59d4c3e04f9974e2472f9a1d

PAR digests are in `signed/par_hashes.json`. The five runtimes are built
`@fbcode//mode/dev-nosan`; `phase4_runtime_launcher` is built `@fbcode//mode/opt`
and that is not interchangeable.

## Why the parity fix was needed

The registered absolute parity tolerance of 1e-06 is unsatisfiable at clip 10 in
float32: `clip_value * 2**-23 = 1.192e-06 > 1e-06`. The amendment now carries a
`centering_contract` block scaling the bound by the registered clip value, with
a relative tolerance of 4.76837158203125e-07.

That fix was proven before this run, not by it. A scratch rehearsal (chain c,
280-interaction clone) ran the identical pipeline end to end and passed
`secondary_diagnostics` non-vacuously, which the earlier chain b could not do
because it ran an instrumented build with the parity refusal downgraded to a
print. Chain c's gap was 5.15 on the reduced budget; the difference from 14.25
here is the interaction budget, not a discrepancy.

## Reproducing on a new machine

Stage A evidence does not survive a machine move. Seals bind the launcher PAR
digest, and the native support pin differs per host, so old seals are
unverifiable no matter what the source says. Budget the full re-run.

    runbook/real-port-run5.sh      sync the four changed files into the fbsource
                                   cell, rebuild the runtimes and the launcher
    runbook/real-mint-run5.sh      mint the three chain documents, create the
                                   owner directory skeleton, generate the stage
                                   scripts
    ln -sfn chain-real-run5.conf ~/.upi-trm-chain.conf
    runbook/upi-trm-supervisor.sh  drives everything from cron

Install the supervisor on `*/3 * * * *` plus `@reboot`. It reads state off the
evidence tree, relaunches only the missing Stage A positions, walks Stage B in
order stopping on the first non-zero rc, and audits. It does not depend on any
agent staying awake.

Four directories must exist before Stage A starts, mode 0700, owned by the
running user and not group-writable. The runtime deliberately refuses to create
them:

    evidence4/
    evidence4/<generation>/
    evidence4/<generation>/payloads/
    evidence4/<generation>/checkpoints/

`payloads/` is required before Stage A even though only Stage B writes into it.
`base_policy/` and the per-seed checkpoint subdirectories are runtime-created;
leave those alone.

Do not run any `buck2 build` between the mint and the end of Stage B. The
authorization pins PAR digests measured at mint time, so a rebuild invalidates
every subsequent launcher invocation.
