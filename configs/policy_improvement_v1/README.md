# Policy-improvement v1 integration boundary

This directory registers the learned outcome study. It does not authorize a
full training grid. `protocol.json` remains `registered_not_authorized`, its
dataset and producer identities are unavailable, and every selection-dependent
row remains a template until a strict validation-only selection amendment is
frozen.

The deterministic registry contains 157 rows:

- 4 Stage 0 smoke rows;
- 48 Stage 1 validation-screen rows;
- 9 Stage 1 validation-only alpha templates;
- 32 Stage 2 confirmatory templates;
- 64 Stage 3 ablation templates.

The 153 non-smoke rows require `RUN_UPITRM_FULL_EXPERIMENTS=1`. The smoke rows
do not. They still require an authenticated runtime, committed dataset and
producer identities, and the dedicated two-segment smoke contract.

`registry.json` must equal byte-for-byte canonical base regeneration from
`protocol.json`. Selection uses three immutable documents:

1. `policy_improvement_compute_freeze_v1` follows the four smoke rows. It binds
   an independently recomputed smoke audit, the external runtime/source
   authorization, and one common compute target per tier. The Stage 1 screen
   is already concrete in the base registry.
2. `policy_improvement_screen_selection_v1` follows the 48-row screen. It binds
   an independently recomputed screen audit and freezes only exact method,
   `n`, and `K`. Only the nine Stage 1 alpha rows become concrete.
3. `policy_improvement_final_selection_v1` follows the nine alpha rows. It
   binds an independently recomputed alpha audit, freezes alpha, and
   materializes Stage 2 and Stage 3 with exact ablation overrides.

Each document binds the prior registry and amendment-history prefix. The audit
requires the prior result and per-instance documents and recomputes them. A
syntax-valid result digest alone cannot authorize the next stage. All three
documents require `test_data_opened: false`.

Selection is mechanical. The screen maximizes the mean of the three seed-level
interaction-matched primary-policy solve rates over the two exact methods and
all registered `n,K` tuples. Ties prefer persistent before episodic, then
smaller `n`, then smaller `K`. The alpha stage uses the same endpoint for the
three registered alphas and breaks ties toward the smaller alpha. The audit
derives both choices from recursively audited per-instance evidence.

Every result contains separate interaction-matched and compute-matched
snapshots. The compute target is common across methods and must be frozen
before non-smoke execution. The per-instance audit recomputes all aggregates,
requires every registered cell including failures, and rejects arbitrary
protocol, registry-row, result, or artifact identities. Confirmatory and
ablation audits also require the single registered test-open record.

`evaluation_populations` registers validation indices 0-7 for smoke, all 256
validation records for pilot work, and all 512 test records for confirmatory
work. Their ordered identities remain unavailable until the dataset is frozen.
Each per-instance record carries its registered index and dataset record
SHA-256. The audit requires an explicit dataset root, authenticates its top and
split manifests, and binds every index to the registered record hash, input
hash, and canonical `<split>-<six-digit-index>` puzzle identifier. It derives
the exact denominator and order from those authenticated bytes.

Runtime authorization is external to both the protocol and PAR, avoiding a
self-hash cycle. `policy_improvement_runtime_authorization_v1` fixes the clean
producer commit and manifest, launcher, and exact training, evaluation, audit,
and analysis source-commit/runtime/profile/source-manifest tuples. The
training-role source commit must equal the producer commit. Results bind the
authorization SHA-256 and their training/evaluation source commits.
Audit and analysis compare their execution tuple with the exact authorized
role. Direct Python invocation is not an authenticated publication path.

The analysis consumer accepts only an audit-validated Stage 2 test,
interaction-matched set. It requires all four methods and all eight seeds,
refuses to drop an explicit failure, and uses exact-mixture persistent UPI and
realized-policy PPO for the primary contrast. It binds its audit, test-open
record, result/per-instance sets, authorization, and analysis runtime.

The registered analysis uses a paired seed-cluster then paired-puzzle
bootstrap, an exact seed-level sign-flip test, and Holm correction for
pre-specified secondary contrasts. Puzzle observations are never treated as
independent training replicates.

Analysis performs semantic revalidation under the authenticated analysis role.
It separately verifies the external audit-role report and records distinct
hashes and runtime identities for both operations. Each contrast reports
per-seed treatment/control counts and rates, aggregate absolute rates, the
percentage-point difference, and a relative ratio. A zero control solve rate
produces an explicit unavailable ratio.

## Current execution boundary

Stage 0 has a dedicated pre-import launcher role and a two-process 16/32
interaction contract. The runner authenticates the committed protocol,
registry, method config, source manifest, dataset manifests, external runtime
authorization, and clean source commit before model construction. Resume
reopens the exact immutable 16-interaction checkpoint and verifies RNG,
optimizer, replay, collector, environment, latent, configuration, and source
identity before continuing. Complete and failed evidence use disjoint
immutable directory inventories.

The audit and analysis PARs use exact pre-import source profiles. Complete
evidence is accepted only after a sealed semantic validator reconstructs the
registered session and strictly reloads every checkpoint role without a
training call, evaluation call, or optimizer step. Summary strings and
checkpoint-validation JSON are not trusted as semantic evidence.

`scripts/policy_improvement_full_runtime.py` implements the registration,
test-isolation, schedule, and immutable publication transaction for Stage 1,
Stage 2, and Stage 3. It deliberately has no source-tree backend. Full learned
execution remains fail-closed until the authenticated launcher, packaged
trainer/evaluator backend, non-smoke checkpoint validator, and Buck runtime
target are connected. This is an explicit infrastructure blocker, not an
authorization to invent full-run commands or evidence.

`scripts/policy_improvement_smoke_plan.py` renders the intended authenticated
launcher commands. It never spawns a process. It accepts the external runtime
authorization and rejects a training runtime, producer commit, or protocol
digest that differs from that document.

Stage 3 materialization reads each registered flat YAML config, checks its byte
and canonical-content SHA-256 values, applies the exact variant override, and
records the recomputed effective-config SHA-256 in every row. The result audit
requires the same override and effective identity. Stage 0, Stage 1, and Stage
2 use the same contract: every concrete row carries its base-config hash,
complete registered override, and expected effective-config hash.

## Buck ownership

The schema, registry, dataset builder, smoke planner, smoke runtime, strict
checkpoint validator, evidence authenticator, audit, analysis, statistics,
test-open transaction, and fail-closed full-runtime core have owned libraries
and tests. Generated type targets cover every changed library and test. The
Stage 1-3 backend remains intentionally unowned because it does not yet exist.
