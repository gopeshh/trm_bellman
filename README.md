# UPI-TRM experiment code

This repository contains the plan-edit MDP, recurrent evaluator, training loops,
baselines, diagnostics, and retained experiment outputs for the UPI-TRM paper.
The current manuscript is `../UPI_TRM/UPI_TRM_ICLR/main.tex`.

## Audit status

The July 2026 correctness audit repaired the following implementation paths:

- UPI, SB3, and CleanRL Sudoku evaluation use disjoint held-out splits,
  reject overlapping or undersized pools, and retain ordered-pool SHA-256 values;
- requested datasets fail closed instead of silently using random dummy data;
- persistent replay retains recurrent state and collection-time behavior probability;
- the finite-horizon edit clock is part of each state;
- terminal targets, fixed-K targets, and exact baseline enumeration share one reward contract;
- exact-mixture evaluation evaluates the mixture, not only its old-policy component;
- evaluation uses a private seeded RNG stream and cannot perturb later training samples;
- exact per-state advantages remain centered after clipping;
- exact-mixture actors share the post-value-update, post-clamp recurrent snapshot;
- projection and projection diagnostics use the same joint latent norm;
- multi-root datasets receive disjoint puzzle-identifier ranges;
- schema-v2 checkpoints retain model construction config, target state, replay,
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

```bash
python upi_trm_train.py \
  --dataset-paths /path/to/sudoku-dataset \
  --train-split train \
  --eval-split test \
  --config configs/revision/upi_trm_feasibility_episodic_z_hard_suite_theory_exact.yaml
```

Runs log both split sizes and the ordered evaluation-pool hash. Preserve those
lines with every result artifact.

For smoke tests only, omit `--dataset-paths` to use `DummyPuzzleDataset`.

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

That package pattern also runs Buck's generated Python type-check targets. The
July 2026 audit validated 249 runtime tests and 35 generated type-check targets
on a fresh daemon.

In a standard environment with the dependencies installed:

```bash
pytest -q
```

## Interpretation constraints

`theory_exact_mixture=true` evaluates a pointwise exact mixture between a fixed
base policy and the current candidate. Candidate optimization may take many
gradient steps, but this is one fixed-base CPI proposal, not an exact recursive
sequence of deployed CPI mixtures.

Operator-norm clamping is a contraction-oriented intervention. Local
finite-difference diagnostics do not certify a global contraction modulus.

Evaluation checkpoints persist the exact model construction config. Legacy
checkpoints with ambiguous per-puzzle embeddings fail closed instead of silently
dropping embedding weights.

New schema-v2 training checkpoints support exact continuation when the dataset
hashes and CUDA topology match. Older checkpoints require
`--allow-legacy-resume` and are treated as warm starts.

UPI outer updates and SB3 environment steps are different budget units.
Aggregate outputs keep those units separate and do not report a between-method
gap interval unless a common interaction budget is supplied.
