# UPI-TRM experiment code

This repository contains the plan-edit MDP, recurrent evaluator, training
loops, baselines, diagnostics, and authenticated experiment tooling for the
UPI-TRM paper.

## Experiment status

Historical learned outputs were produced by superseded implementations and
protocols. They were invalidated and removed from the current branch. They are
not evidence for the current paper and must not be reconstructed or reused.
Git history retains the deleted files if a forensic comparison is needed.

The current experiment namespace is `policy-improvement-v2-20260818`. No fresh
learned result is committed. Stage 1 through Stage 3 remain fail-closed until
their registered prerequisites are supplied. The test split is not authorized
for the current preparation and Stage 0 work.

## Main paths

- `models/recursive_reasoning/trm.py`: recurrent evaluator and policy/value heads
- `rl/envs/plan_edit_env.py`: finite-horizon plan-edit MDP
- `rl/upi_trm_trainer.py`: UPI-TRM collection and optimization
- `rl/value_targets.py`: one-step and fixed-K targets
- `rl/training_setup.py`: dataset materialization and checker selection
- `upi_trm_train.py`: main training entry point
- `run_baseline.py`: external SB3 baseline entry point
- `scripts/`: evaluation, diagnostics, aggregation, and artifact tools
- `tests/`: Buck and pytest regression tests
- `results/README.md`: invalidation notice; generated results remain untracked

## Data requirements

Real runs require a dataset root with separate `train/` and `test/` directories.
The loader refuses a missing requested dataset, overlapping train/test records,
or an evaluation pool smaller than `eval_num_episodes`.

The historical revision config at
`configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml`
is retained as a non-runnable reference fixture. It is not a registered
confirmatory assignment, so the main runner rejects it under
`fixed_base_exact`. The old multi-GPU launcher for that config also exits
without starting a run. Do not infer a replacement experiment protocol from
that fixture.

Registered runs log both split sizes and the ordered evaluation-pool hash.
Preserve those lines with every result artifact. Legacy smoke-only paths may
omit `--dataset-paths` to use `DummyPuzzleDataset`; `fixed_base_exact` requires
materialized, disjoint train and evaluation splits.

## Confirmatory runtime boundary

Source-tree execution is development-only. A run using `--confirmatory` must
execute the standalone `upi_trm_train` PAR through
`confirmatory_runtime_launcher`. The launcher takes an absolute PAR path and
an externally frozen SHA-256, validates the whole artifact plus its embedded
source manifest before importing training code, copies the verified bytes into
a sealed anonymous file, and starts that immutable descriptor as a supervised
child process. Each launch also uses a fresh private PAR unpack directory
instead of a shared extraction cache. The launcher passes that directory as an
inherited directory descriptor, so path replacement after descriptor binding
cannot redirect extraction or entrypoint validation. Direct unattested PAR
execution and source-tree confirmatory execution fail before model or RL
modules are imported. Under the trusted host namespace, the launcher removes
the private unpack directory when the child exits, including startup failures.

Build both artifacts, freeze the training PAR digest outside the PAR, then use
the launcher as follows:

```bash
buck2 build --local-only @fbcode//mode/opt --show-output \
  fbcode//buiksat_trm:upi_trm_train \
  fbcode//buiksat_trm:confirmatory_runtime_launcher
sha256sum /absolute/path/to/upi_trm_train.par
buck2 run --local-only @fbcode//mode/opt \
  fbcode//buiksat_trm:confirmatory_runtime_launcher -- \
  --runtime-archive /absolute/path/to/upi_trm_train.par \
  --expected-runtime-sha256 <64-lowercase-hex-digest> -- \
  --confirmatory <registered-run-arguments>
```

The current effective configuration, checkpoint evidence identity, and
confirmatory lock all record the verified PAR digest. Do not put that digest
inside `run_matrix.json` or another PAR input because doing so creates a
self-hash cycle.

The launcher executable, operating system, host namespace, other same-UID
processes, and launcher process environment are the external root of trust.
The launcher is not a sandbox against a compromised host or hostile same-UID
process. Start it from a controlled process without loader, Python-path, or PAR
override hooks. The launcher removes those hooks before executing the training
PAR. Its inherited descriptors and attestation variables are capabilities for
the verified runtime and unpack directory, not a claim that environment
variables are cryptographically unforgeable.

Phase 4 publication tools use the separate `phase4_runtime_launcher`. The
launcher accepts one explicit role, runtime PAR, runtime SHA-256, clean source
checkout, and expected source commit. It authenticates the complete artifact
before importing NumPy, Torch, dataset/model/RL code, or a Phase 4 consumer.
Direct entry into a Phase 4 role without the launcher fails before those
behavior imports.

## Tests

From an fbcode checkout where this repository is available as `buiksat_trm`:

```bash
buck2 test --local-only @fbcode//mode/opt fbcode//buiksat_trm:test_upi_trm_trainer_smoke
buck2 test --local-only @fbcode//mode/opt fbcode//buiksat_trm:test_rl_k_step_targets
buck2 test --local-only @fbcode//mode/opt fbcode//buiksat_trm:test_optimized_exact_baseline
```

Run all declared tests with:

```bash
buck2 test --local-only @fbcode//mode/opt 'fbcode//buiksat_trm:'
```

That package pattern also runs Buck's generated Python type-check targets.
Do not treat historical test counts or deleted logs as proof for the current
commit. Record fresh commands and results against the exact source revision
being evaluated.

In a standard environment with the dependencies installed:

```bash
pytest -q
```

## Interpretation constraints

`theory_exact_mixture=true` controls pointwise probability-space deployment. It
does not repair the historical mutable training path by itself. New theorem-facing
runs must also set `training_protocol: fixed_base_exact`, which collects replay
from one frozen base, updates only the value head, and treats all candidate steps
as one proposal rather than a recursive sequence of deployed CPI mixtures.
The exact protocol also uses terminal STOP semantics, leaves value targets
unclipped, disables distillation and exploration mixing, and deploys the exact
pointwise policy mixture. Its fixed-K regression target uses the frozen target
evaluator followed by the configured retention update. Do not identify that
operator with the self-bootstrap Bellman operator used in the residual
certificate.

Projection is configured independently of contraction. Use
`latent_projection_mode: enabled` with a finite positive
`latent_ball_radius`, or `latent_projection_mode: disabled` with no radius.
Radius zero is accepted only when migrating historical payloads; it is not the
current disabling convention.

Operator-norm clamping is a contraction-oriented intervention. Local
finite-difference diagnostics do not certify a global contraction modulus.

Evaluation checkpoints persist the exact model construction config. Legacy
checkpoints with ambiguous per-puzzle embeddings fail closed instead of silently
dropping embedding weights.

New schema-v5 training checkpoints record the training protocol and support
exact continuation when dataset hashes, run identity, source identity, and CUDA
topology match. Schema-v3 and schema-v4 checkpoints are historical formats.
Older checkpoints are weights-only warm starts and are never treated as exact
continuation of a schema-v5 theorem-facing run.

Executable parity does not certify the paper's uniform assumptions. In
particular, regression loss, finite-batch maxima, and local Lipschitz diagnostics
do not establish the required sup-norm residuals, invariant-domain bounds,
policy-overlap constants, or signed occupancy-averaged defect bound. No result
in this repository turns those assumptions into a claim about SGD, TD, BPTT,
distillation, learned-model performance, or training dynamics.

UPI outer updates and SB3 environment steps are different budget units.
Aggregate outputs keep those units separate and do not report a between-method
gap interval unless a common interaction budget is supplied.
