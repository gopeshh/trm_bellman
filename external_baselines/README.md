# External Baseline Harness

This directory is the `T0.2` setup path for trusted external baselines on hard 4x4 Sudoku.
The dataset directory name `sudoku-4x4-easy_6to8empties` is historical; this harness is the paper's hard 4x4 no-mask suite.

## What exists now

- `sudoku4x4_env.py`: a minimal hard-4x4 no-mask environment that matches the internal flat action semantics.
- `run_baseline.py`: the entrypoint required by `NIPS_PLAN.md`.
- `npy_reader.py`: a tiny `.npy` loader so the smoke harness works even before NumPy is installed.

## Observation and action contract

- Observation keys: `inputs`, `plan`, `action_mask`
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
`T0.2` is complete via that Buck target: both PPO and A2C now finish one SB3-backed episode successfully on the hard 4x4 wrapper.

## Masking semantics

The environment exposes `action_mask` in the observation, but vanilla SB3 PPO/A2C does not use that mask for logit masking.
That is intentional here: the paper's locked hard-4x4 capability anchor is the no-mask protocol, so the external policy should face the same unmasked action space and let the environment penalize invalid edits.

For internal fairness, compare against the repo's no-mask A2C baseline config:

```text
configs/baselines/a2c_trm_feasibility_no_mask.yaml
```

That is also the regime used by the paper-side hard-4x4 anchor figure path in:

```text
scripts/plot_table3_hard.py
```

## Acceptance command

On this host, `python3` is available and `python` is not. The smoke-only command is:

```bash
python3 run_baseline.py --algo ppo --env sudoku4x4
```

The Buck-backed `T0.2` acceptance checks are:

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
```

The command writes a summary JSON to:

```bash
results/neurips2026/external_hard4x4/<algo>/seed<seed>/episode_summary.json
```

## Training mode

`run_baseline.py` also supports `--mode train` for T0.3 launch prep. That path:

- trains PPO or A2C with SB3 on the hard-4x4 train split
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
