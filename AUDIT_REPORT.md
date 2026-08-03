# UPI-TRM repository audit

Date: 2026-07-30  
Starting commit: `6d5a241027fe72921d5fc039dd6a12434999088b`

## Decision

The retained headline experiments are not acceptance-grade evidence.

- The 57.4% UPI-TRM evaluation used the first 32 `train` records for both
  training and evaluation.
- The SB3 rows used 50 `test` records. Those rows are not a matched comparison.
- The historical UPI path used parameter interpolation, not the distillation
  described by the manuscript.
- The reported theory-oriented 0% run evaluated `policy_model_old`, not the
  explicit old/candidate mixture used for collection.
- The checkout lacks the hard-suite dataset and checkpoints required to rerun
  either result. Existing numerical claims cannot be repaired by relabeling.

Fresh held-out experiments are required.

## Correctness repairs

### Data and evaluation

- Real dataset requests now fail closed. Dummy fallback is explicit and
  opt-in at the loader level.
- Training and evaluation split names must differ.
- Physical input overlap between materialized train and evaluation pools is
  rejected.
- Multiple dataset roots receive disjoint puzzle-identifier ranges, and fresh
  evaluation identifiers are disjoint from training identifiers.
- Evaluation refuses to repeat a smaller pool to reach the requested episode
  count.
- Training and external baselines emit the same ordered-pool SHA-256 format.
- The hard-suite aggregator requires `train`, `test`, exactly 50 evaluation
  records, one pool hash across seeds, and the same hash across methods.
- UPI outer updates and SB3 environment steps are labeled as different units.
  The aggregator no longer emits a between-method gap interval.

### MDP and targets

- `remaining_edits` is included in state snapshots and replay batches.
- Budget exhaustion is an MDP terminal in internal, external, and CleanRL
  wrappers. PPO and DQN no longer bootstrap it as a time-limit truncation.
- Terminal rewards fold the absorbing-state continuation into the final reward.
- Fixed-K targets reject incomplete nonterminal segments, and the sampler
  selects only K-complete or early-terminal segments in exact mode.
- Exact-Q successor evaluation receives the decremented edit clock.
- Exact baseline enumeration uses the same solved predicate and reward helper
  as `PlanEditEnv.step()`.
- Replay batching preserves `solution` and other common tensor state fields, so
  the solution checker cannot silently fall back to the dummy potential.
- History-dependent UNDO is rejected by exact enumeration instead of being
  decoded as an edit.

### Policies and recurrent state

- Persistent replay retains input and successor latent carries.
- Value, policy, debug, and residual paths consume those stored carries.
- Importance sampling uses the behavior log probability stored at collection
  time.
- Disabled STOP remains outside the training support after task-specific masks.
- Exact-mixture evaluation calls the mixture distribution rather than only the
  old model.
- Stochastic mixture evaluation uses a private fixed RNG stream and restores
  CPU and CUDA RNG state before training resumes.
- Clipped exact advantages are re-centered under the current statewise policy.
- The historical legacy path gives old and candidate actors the same
  post-value-update recurrent snapshot. The opt-in `fixed_base_exact` path
  instead freezes the recurrent actor map and trains only the value head.
- Candidate recurrence synchronization copies parameters and persistent
  buffers. Target soft updates also copy persistent buffers.

### Checkpoints and reevaluation

- Full checkpoints persist the model construction config, old/candidate/target
  states, optimizer and scheduler states, counters, replay contents, content-based
  dataset identity, and Python/NumPy/Torch RNG states.
- Resume validates dataset identity before mutating model or replay state and
  restores RNG state last. Legacy checkpoints require an explicit warm-start flag.
- Replay deserializes on CPU before model state is copied to the selected
  device, avoiding accidental CUDA replay allocation.
- Reevaluation reconstructs nondefault architectures from the persisted config.
  Legacy per-puzzle-embedding checkpoints without enough mapping metadata fail
  closed instead of silently evaluating a different model.
- Trusted full-checkpoint loaders opt out of PyTorch's weights-only mode
  explicitly because replay contains `Transition` objects.

### Baselines and dataset generation

- CleanRL Sudoku paths materialize separate `train` and `test` pools, reject
  duplicate inputs and undersized evaluation pools, offset evaluation puzzle
  identifiers, and write the common ordered-pool hash into run artifacts.
- The shared evaluator rejects implicit pool cycling. The historical reevaluation
  script opts into cycling explicitly because reproducing its 50-on-32 protocol is
  the purpose of that script.
- PPO truncations bootstrap the final observation once and stop GAE at the reset
  boundary. DQN epsilon exploration samples only valid masked actions.
- Sudoku and maze builders use local seeded NumPy generators, record their build
  configuration, and accept a source revision for pinned downloads.

`theory_exact_mixture=true` selects exact probability-space deployment but does
not identify the training protocol. `training_protocol=fixed_base_exact` is the
one fixed-base CPI proposal path. The two-network implementation does not
represent a recursively growing exact mixture across outer updates.

### Projection and diagnostics

- Production projection and diagnostics use the Euclidean product norm over
  both recurrent carries.
- Experiment instrumentation hooks the live joint projection method.
- Reference-depth Q diagnostics reuse the depth-n policy probabilities.
- Uniform materialized-batch diagnostics are no longer labeled discounted
  occupancy surrogates.
- Extended-real summaries preserve infinities and serialize them as strict JSON
  strings rather than non-standard `Infinity` or `NaN` literals.
- Radius aggregation fixes both depths at `2 -> 8` by default.

This joint-ball projection changes semantics relative to historical checkpoints
trained with separate per-carry projection. Existing finite-radius tables must
not be relabeled as outputs of the repaired code; they require a versioned rerun.

## Repository cleanup

Removed PID/lock/partial/smoke files, duplicate exports, duplicate plots, stale
internal handoffs, the broken W&B aggregation pipeline, and the
obsolete NeurIPS artifact builder. Raw experiment evidence and distinct backup
runs were retained.

`scripts/build_artifact_zip.sh` now creates one archive from all tracked and
non-ignored working-tree files and writes a SHA-256 sidecar.

## Validation

Final validation on a fresh Buck daemon:

- All 28 declared `python_unittest` targets passed: 249 tests, zero failures.
- Thirty-five generated type-check targets passed across the RL, model,
  dataset, evaluator, baseline, diagnostic, and plotting layers.
- The five production/audit entrypoints built successfully: `upi_trm_train`,
  `run_baseline`, `reevaluate_upi_baseline_interface`,
  `episodic_z_hard_suite_diagnostics`, and `run_exp1_finite_r_primary`.
- Every changed Python file passed `py_compile`; `git diff --check` passed.
- The finite-MDP script regenerated 101 rows; both certificate flags are true
  in every row.
- The revised ICLR manuscript compiled to 38 pages with no undefined
  references or overfull boxes.

## Remaining work before using results in a paper

1. Restore or regenerate the hard-suite train/test dataset with checksums.
2. Run UPI-TRM and each baseline on the same ordered test pool.
3. Match environment interactions, seeds, evaluation checkpoints, and endpoint.
4. Preserve per-seed logs, pool hashes, checkpoints, environment counters, and
   the exact repository commit.
5. Run the persistent-to-episodic bridge one factor at a time.
6. Report null and negative runs. Do not reuse the historical 57.4% result as a
   held-out result.
