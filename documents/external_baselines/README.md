# External Baseline Harness

This directory contains the external baseline harness for hard 4x4 Sudoku.
The dataset directory name `sudoku-4x4-easy_6to8empties` is historical. That
dataset is not retained in this checkout, so the old runs cannot be reproduced
until a checksum-identified train/test dataset is restored.

## What exists now

- `sudoku4x4_env.py`: a minimal hard-4x4 no-mask environment that matches the internal flat action semantics.
- `run_baseline.py`: the entrypoint required by `NIPS_PLAN.md`.
- `npy_reader.py`: a tiny `.npy` loader so the smoke harness works even before NumPy is installed.

## Observation and action contract

- Observation keys: `inputs`, `plan`, `remaining_edits`, `action_mask`
- Action space: `97` discrete actions
- Layout: `16 positions * 6 tokens + 1 STOP`
- Protocol: hard 4x4, no-mask, feasibility reward, `max_edits=16`

This mirrors the internal environment at the level needed for baseline integration:
- same flat action indexing
- same original-puzzle / current-plan split
- same no-mask action-mask semantics

## Backends

`run_baseline.py` supports two execution modes:

- `backend=sb3`: used automatically if `stable_baselines3`, `gym` or `gymnasium`, and `numpy` are installed.
- `backend=smoke`: fallback path that runs a valid-action smoke controller when third-party packages are unavailable.

The plain `python3` shell environment on this machine still does not have SB3/Gym/NumPy importable.
The supported path is Buck: `fbcode//buiksat_trm:run_baseline` pulls the third-party packages from `fbsource//third-party/pypi/...`.
`T0.2` is complete via that Buck target for PPO/A2C, and the same harness now accepts SB3 DQN as well as n-step DQN through `--dqn-n-steps`.

## Masking semantics

The environment exposes `action_mask` in the observation, but vanilla SB3 PPO/A2C/DQN does not use that mask for logit masking.
That is intentional for the historical no-mask protocol: the external policy
faces the unmasked action space and the environment penalizes invalid edits.

For internal fairness, compare against the repo's no-mask A2C baseline config:

```text
configs/baselines/a2c_trm_feasibility_no_mask.yaml
```

That is also the regime used by the paper-side hard-4x4 anchor figure path in:

```text
scripts/plot_table3_hard.py
```

## Fresh-run command

On this host, `python3` is available and `python` is not. The smoke-only command is:

```bash
python3 run_baseline.py --algo ppo --env sudoku4x4
```

After restoring a disjoint train/test dataset, the Buck-backed commands are:

```bash
cd /data/repos/fbsource/fbcode
buck2 run fbcode//buiksat_trm:run_baseline -- \
  --algo ppo --env sudoku4x4 --backend sb3 \
  --dataset-dir /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
  --output-root /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4

buck2 run fbcode//buiksat_trm:run_baseline -- \
  --algo a2c --env sudoku4x4 --backend sb3 \
  --dataset-dir /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
  --output-root /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4

buck2 run fbcode//buiksat_trm:run_baseline -- \
  --algo dqn --env sudoku4x4 --backend sb3 \
  --dataset-dir /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
  --output-root /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4

buck2 run fbcode//buiksat_trm:run_baseline -- \
  --algo dqn --dqn-n-steps 5 --env sudoku4x4 --backend sb3 \
  --dataset-dir /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
  --output-root /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4_20k
```

The command writes a summary JSON to:

```bash
results/neurips2026/external_hard4x4/<algo>/seed<seed>/episode_summary.json
```

For `--algo dqn --dqn-n-steps 5`, artifacts land under:

```bash
results/neurips2026/external_hard4x4_20k/dqn_nstep5/seed<seed>/
```

## Training mode

`run_baseline.py` also supports `--mode train` for launch prep. That path:

- trains PPO, A2C, or DQN with SB3 on the hard-4x4 train split
- evaluates periodically on the test split
- writes non-colliding artifacts per seed
- saves intermediate checkpoints and a final model

Example:

```bash
cd /data/repos/fbsource/fbcode
buck2 run fbcode//buiksat_trm:run_baseline -- \
  --algo a2c --env sudoku4x4 --backend sb3 --mode train \
  --dataset-dir /home/buiksat/trm_bellman/data/sudoku-4x4-easy_6to8empties \
  --output-root /home/buiksat/trm_bellman/results/neurips2026/external_hard4x4 \
  --train-steps 5000 --eval-freq 100 --eval-episodes 50 --checkpoint-freq 1000
```

Artifacts land under:

```bash
results/neurips2026/external_hard4x4/<algo>/seed<seed>/
```

For the full seed sweep, use:

```bash
scripts/run_external_baseline_sweep.sh
```
