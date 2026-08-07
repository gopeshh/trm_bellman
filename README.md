# UPI-TRM experiment code

This repository contains the plan-edit MDP, recurrent evaluator, training loops,
baselines, diagnostics, and retained experiment outputs for the UPI-TRM paper.
The implementation parity anchor is the manuscript at paper commit `2107125`
and this repository's `full-implementation` branch. See
`reports/PAPER_PARITY_REPORT.md` for the executable Algorithm 1/2 mapping and
the theorem premises that remain conditional.

## Audit status

The July 2026 correctness audit repaired the following implementation paths:

- UPI, SB3, and CleanRL Sudoku evaluation use disjoint held-out splits,
  reject overlapping or undersized pools, and retain ordered-pool SHA-256 values;
- requested datasets fail closed instead of silently using random dummy data;
- persistent replay retains recurrent state and collection-time behavior probability;
- the finite-horizon edit clock is part of each state;
- terminal targets, fixed-K targets, and exact baseline enumeration share one reward contract;
- exact fixed-K targets bootstrap from the frozen target evaluator; this is a
  target-network population backup, not a self-bootstrap with the value head
  currently being fitted;
- exact-mixture evaluation evaluates the mixture, not only its old-policy component;
- evaluation uses a private seeded RNG stream and cannot perturb later training samples;
- exact per-state advantages remain centered after clipping;
- fixed-base training collects from one frozen base actor, updates only the
  policy-independent value head, and evaluates the explicit proposal mixture;
- projection and projection diagnostics use the same joint latent norm, with
  an explicit `enabled` mode requiring `R>0` and a `disabled` identity mode;
- multi-root datasets receive disjoint puzzle-identifier ranges;
- schema-v5 checkpoints retain training-protocol identity, model construction
  config, target state, replay,
  schedulers, counters, optimizer state, dataset identity, and Python/NumPy/Torch
  RNG state without loading replay onto CUDA;
- Sudoku and maze dataset builders use a recorded local RNG seed and accept a
  source revision instead of depending on ambient NumPy state;
- baseline wrappers treat budget exhaustion as an MDP terminal;
- aggregate scripts reject missing or mismatched evaluation provenance.

The retained 57.4% UPI-TRM result was evaluated on the first 32 training
instances. It is in-sample and must not be compared with the 50-instance SB3
test pool. The checkout also lacks the hard-suite data and checkpoints needed
to rerun that result. New claims must come from fresh held-out runs.

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
- `results/`: retained experiment evidence. Do not delete this tree wholesale.

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
The final paper-parity runtime selection passed 243/243, and the six generated
type-check targets touched by the synchronization passed 6/6. Exact commands,
the repository-wide diagnostic result, and the scope of the executable parity
claim are recorded in `reports/PAPER_PARITY_REPORT.md`. The full package pattern
is not green because pre-existing aggregate libraries, tests, runners,
diagnostics, and experiment scripts retain unrelated type-check debt.

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
