# Checkpoint schema-v5 verification report

Date: 2026-08-03, America/Los_Angeles

Repository: `/home/buiksat/trm_bellman`

Branch: `iclr-confirmatory-repair`

This report covers the confirmatory checkpoint and run-identity repair. It is
implementation evidence only. No learned checkpoint or confirmatory outcome
was produced.

## Confirmatory checkpoint contract

`fixed_base_exact` runs now write checkpoint schema 5. The schema binds:

- an explicit run ID and seed;
- the exact effective training configuration and its canonical SHA-256;
- clean producer Git commit, source inventory, and source blobs equal to HEAD;
- train, validation, and held-out record identities and dataset provenance;
- initialization mode and weights-checkpoint SHA-256 when applicable;
- Python, NumPy, Torch, CUDA, cuDNN, deterministic-runtime, device, thread, and
  selected environment settings;
- exact environment-interaction and actual optimizer-step counters;
- model modes, gradients, replay, active persistent latent, environment,
  collector, optimizers, schedulers, and process RNG state;
- an explicit parent-checkpoint lineage record for every resumed save.

The loader validates all immutable and mutable payloads on shadow objects
before mutating live training state. A fixed-base run rejects schema 3 or 4,
dummy or overlapping datasets, duplicate evaluation records, incomplete
dataset provenance, incompatible runtime identity, mutable source files,
unregistered budgets, malformed episodic or persistent replay, and invalid
parent lineage.

Checkpoint publication is atomic and refuses overwrite. The saved byte stream
is flushed and synchronized before a no-replace hard link publishes the final
path. Save failures propagate to the caller.

## Regression coverage

New and expanded tests cover:

- canonical run identity and exact effective-configuration matching;
- unsafe Git index flags and source blobs that differ from HEAD;
- schema downgrade, configuration, runtime, and lineage rejection before live
  mutation;
- corrupt module, optimizer, gradient, trainer, replay, and active-latent
  payloads before restore;
- episodic and clock-complete persistent replay round trips;
- actual optimizer and scheduler counters;
- atomic no-overwrite checkpoint publication;
- uninterrupted versus save/restart equivalence for the next value and policy
  optimizer update, including all four modules, both optimizers, counters, and
  subsequent RNG draws;
- mocked all-device CUDA RNG capture and restore;
- diagnostic checkpoint source, runtime, identity, and lineage verification;
- mutation of a diagnostic checkpoint during the same-open load.

## Validation

Commands were run from the Buck cell at `/data/users/buiksat/fbsource` with
`--local-only` because the cell uses an external repository symlink.

- Focused schema and diagnostic gate: 48 passed, 0 failed. Test session
  `27584547753694264`.
- Full non-logging invocation: 320 passes and two TPX fatals in test session
  `25051272963310650`. Both fatal cases printed `Ran 1 test ... OK`; TPX did
  not create their result JSON files.
- Isolated rerun of the affected target: 31 passed, 0 failed, 0 fatal. Test
  session `28428972670956344`.
- Isolated logging/checkpoint target: 22 passed, 0 failed. Test session
  `30680772484645257`.
- Authoritative unique runtime coverage: 344 cases across all 34 declared
  runtime targets, with no assertion failure after the isolated rerun.
- `upi_trm_train` and `persistent_checkpoint_diagnostics` Buck builds passed.
- `git diff --check` passed.
- `python3 -m py_compile` passed for every changed or new Python file.

The package-wide `rl-type-checking` target still fails on the same nine
baseline errors:

- three in `rl/algos/dqn.py`;
- four in `rl/cleanrl/trm_adapter.py`;
- two in `rl/envs/plan_edit_env.py`.

Test session `3096225071876989` reports no schema-v5 error.

## Evidence boundary

A cross-model adversarial review found no remaining blocking checkpoint-contract
issue. Residual coverage limits remain:

- real CUDA resume equivalence: not verifiable from supplied evidence
- nonconstant-scheduler, multi-record resume equivalence: not verifiable from supplied evidence
- historical live optimizer and RNG restoration: not verifiable from supplied evidence
- learned persistent schema-v5 checkpoint: not verifiable from supplied evidence
- theorem-facing learned-checkpoint diagnostics: not verifiable from supplied evidence

The mocked CUDA test establishes code-path coverage, not hardware equivalence.
The one-record constant-learning-rate equivalence test does not establish every
training schedule. These limits block any claim that a historical checkpoint
has been upgraded or reproduced.
