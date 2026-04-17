# Unrolled Policy Iteration in Plan Space for Tiny Recursive Models (UPI–TRM)

This repository extends the Tiny Recursive Model (TRM) codebase with a plan-space reinforcement learning framework we call **UPI–TRM**. It adds latent value estimation (`U_n(s)`), a configurable K-step value operator, and conservative policy improvement (CPI) mixture updates so that TRMs can be trained and evaluated with lightweight RL loops.

## Canonical paper materials

NeurIPS 2026 paper source tree:
- `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`

Markdown planning/provenance mirror in this repo:
- `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS`

For the current NeurIPS 2026 resubmission plan copy, see:
- `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/NIPS_PLAN.md`

## Repository Structure
- `configs/` – experiment configs grouped by purpose: `ablations/`, `baselines/`, `pilots/`, `sudoku9x9/`, and paper-specific sweeps. Pretrain Hydra configs now live under `configs/pretrain/` (the repo-root `config` symlink remains for compatibility).
- `entrypoints/` – canonical Python entrypoints for imitation and simple RL training. The legacy top-level scripts remain thin wrappers.
- `models/recursive_reasoning/trm.py` – `TinyRecursiveReasoningModel_ACTV1` with RL-specific value/policy heads.
- `models/value_head.py` – latent value head \(V_\psi\). Spectral normalization is optional and controlled by `disable_value_head_norm`.
- `rl/config.py` – `RLConfig` for hyperparameters, logging cadence, CPI knobs, evaluation intervals, and theory-metric toggles.
- `rl/training_setup.py` – dataset bootstrap helpers (`DummyPuzzleDataset`, supervised bootstrap fallback) and checker resolution shared by the RL entrypoint and tests.
- `rl/envs/plan_edit_env.py` – plan-space meta-MDP describing edit actions over latent plans.
- `rl/upi_trm_trainer.py` – trainer implementing 1-step + K-step TD, CPI mixtures, and evaluation hooks.
- `upi_trm_train.py` – main RL training CLI that wires configs, datasets, the environment, and trainer selection together.
- `evaluators/rl_plan_evaluator.py` and `rl/evaluator.py` – evaluation helpers for strict success rate and checker-score metrics.
- `scripts/eval/unroll_sensitivity.py` – canonical unroll-sensitivity evaluator (the legacy `scripts/eval_unroll_sensitivity.py` path remains as a wrapper).
- `scripts/provenance/create_provenance_bundle.sh` – canonical provenance-bundle script (the legacy `scripts/create_provenance_bundle.sh` path remains as a wrapper).
- `tests/` – smoke tests and unit tests for heads, TD targets, CPI mixture, logging, evaluator glue, and config integrity.

## Installation

### Standard Installation (pip)
```bash
git clone <repo-url>
cd trm_bellman
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt  # torch, tqdm, pytest, and the original TRM deps
```
If you prefer not to use `requirements.txt`, minimally install `torch`, `tqdm`, and `pytest`.

### Meta/fbcode Installation (Buck2)

For Meta devservers, the project includes a `BUCK` file that uses internal PyPI packages:

```bash
cd ~/fbsource/fbcode

# Symlink or copy the project to fbcode
ln -s /path/to/trm_bellman buiksat_trm

# Build and run with A100 GPU support
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- \
    --train-steps 1000 \
    --batch-size 64 \
    --seed 42
```

**Important flags for A100 GPUs:**
- `-c fbcode.nvcc_arch=a100` - Enables sm_80 CUDA kernels for A100
- `-c fbcode.enable_gpu_sections=true` - Enables GPU code sections

**Quick debug run:**
```bash
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- \
    --train-steps 20 \
    --batch-size 16 \
    --log-interval 5 \
    --debug-checks
```

## Running Tests

### Using Buck2 (Meta devservers)

```bash
cd ~/fbsource/fbcode

# Run all 22 test targets (117 tests total)
buck2 test //buiksat_trm:test_... \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --local-only

# Run a single test
buck2 test //buiksat_trm:test_rl_k_step_targets \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --local-only

# Clear Buck2 cache (if tests are stale after BUCK file changes)
buck2 clean
```

### Using pytest (standard installation)

```bash
pytest -v
```
Useful targeted suites:
- `tests/test_trm_latent_unroll.py`
- `tests/test_rl_k_step_targets.py`
- `tests/test_upi_trm_trainer_smoke.py`
- `tests/test_cpi_mixture_policy_smoke.py`
- `tests/test_upi_trm_logging_smoke.py`

Or simply run `./scripts/run_tests.sh`.

## Running a Simple RL Training Run
You can exercise the full RL stack (dummy dataset, plan edit env, CPI mixture, K-step TD) with:
```bash
python upi_trm_train.py \
    --train-steps 200 \
    --batch-size 32 \
    --rollouts-per-step 1 \
    --max-edits 8 \
    --log-interval 10 \
    --eval-interval 50 \
    --eval-episodes 50
```
This uses the in-memory `DummyPuzzleDataset`, collects short plan-edit episodes, and prints policy/value losses plus both `eval_success_rate` and `eval_mean_score`. The `scripts/run_rl_dummy.sh` helper runs an equivalent configuration with a smaller batch size (16) to finish even faster.

### Sudoku UPI-TRM Training

Experiment configs are now grouped by workflow instead of living at the repo root. The most important directories are:
- `configs/pilots/` – quick 4×4 development runs.
- `configs/ablations/` – core UPI-TRM ablations.
- `configs/baselines/` – PPO/A2C/DQN baselines.
- `configs/sudoku9x9/` – 9×9 training runs and longer baselines.
- `configs/exp2_contraction_sweep/`, `configs/exp3_projection_ablation/`, `configs/phase4_2x2_norm_ablation/`, `configs/table3_hard_controlled/` – paper-specific experiment sweeps.
- `configs/pretrain/` – Hydra configs for supervised pretraining.

After changing YAMLs, run `python scripts/verify_configs.py` to catch drift in the tracked config invariants.

#### 4×4 Sudoku (Recommended Starting Point)

For development, debugging, and curriculum learning, we recommend starting with 4×4 Sudoku:

```bash
# Trivial: 1–4 empties (fastest convergence)
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-trivial \
    --config configs/pilots/feasibility_trivial.yaml \
    --seed 42
```

The 4×4 puzzles are ideal because:
- **Smaller action space**: 16 cells × 5 tokens = 80 edit actions (vs. ~800 for 9×9)
- **Faster episodes**: Max 16 edits needed (vs. 81 for 9×9)
- **Easier debugging**: Can visually verify solutions

#### Harder 4×4 suite (6–8 empties)

Use the (harder) 6–8 empties suite with a feasibility-enabled config:

```bash
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-easy_6to8empties \
    --config configs/rl_sudoku_4x4_feasibility.yaml \
    --seed 42
```

If the canonical 4×4 splits are missing in a fresh checkout, rebuild them with the Buck-backed helper:

```bash
bash scripts/restore_4x4_datasets.sh
```

That restores `data/sudoku-4x4-trivial`, `data/sudoku-4x4-ultra-easy`, and `data/sudoku-4x4-easy_6to8empties`, then validates their empties ranges with `//buiksat_trm:inspect_4x4_dataset`.

#### 9×9 Sudoku (Full-Scale Experiments)

For full-scale 9×9 Sudoku experiments with train/val/test splits:

```bash
# 1. Generate the dataset (10k train, 1k val, 1k test)
python scripts/gen_sudoku9x9.py \
    --output-dir data/sudoku-9x9 \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000 \
    --seed 42

# 2. Train UPI-TRM on 9×9
python upi_trm_train.py \
    --config configs/sudoku9x9/upi_trm_9x9.yaml \
    --seed 0

# 3. Train baselines
python upi_trm_train.py --config configs/sudoku9x9/ppo_9x9.yaml --baseline ppo --seed 0
python upi_trm_train.py --config configs/sudoku9x9/dqn_9x9.yaml --baseline dqn --seed 0
```

**Multi-GPU Parallel Experiments (4 GPUs, 5 seeds):**

```bash
# Dry run to see what would be executed
python scripts/run_experiments_parallel.py \
    --config-dir configs/sudoku9x9 \
    --seeds 0,1,2,3,4 \
    --gpus 0,1,2,3 \
    --dry-run

# Run all experiments
python scripts/run_experiments_parallel.py \
    --config-dir configs/sudoku9x9 \
    --seeds 0,1,2,3,4 \
    --gpus 0,1,2,3 \
    --output-dir results/sudoku9x9
```

**9×9 Dataset Generation Options:**

```bash
# With unique solution guarantee (slower but higher quality)
python scripts/gen_sudoku9x9.py --ensure-unique --num-train 5000

# Custom difficulty distribution
python scripts/gen_sudoku9x9.py --easy-ratio 0.2 --medium-ratio 0.3 --hard-ratio 0.5
```

Key differences from 4×4:
- **Larger action space**: 81 cells × 10 tokens = 810 edit actions
- **Longer episodes**: Max 81 edits needed
- **Harder puzzles**: 17-35 clues (vs 4-10 for 4×4)

#### Config Selection Notes

- For stability experiments, follow `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/NIPS_PLAN.md`: keep `use_feasibility_checker: true`, set `disable_value_head_norm: true`, and change one variable at a time.
- The 4×4 pilot configs are the fastest way to validate code changes locally.
- The 9×9 configs and paper-specific sweep directories are intended for longer-running experiments and artifact generation.

#### Theory-Exact Features Explained

- **`exact_k_step_targets`**: Uses fixed-horizon γ^K bootstrap for T_K^π operator (vs. γ^steps_taken)
- **`exact_baseline_summation`**: Computes E_{a~π}[Q̂(s,a)] via exact summation for O(α·ε_A) bound (Theorem 5.9 - KEY CONTRIBUTION)
- **`batch_centered_advantage`**: Batch-level mean subtraction (heuristic for variance reduction - NOT the exact centering from Theorem 5.9)
- **`distill_mixture_policy`**: Distills CPI mixture back into network (WARNING: NOT covered by theory - Section 6.5)
- **`use_gae`**: Enables GAE (λ-returns) for variance reduction (practical heuristic)

#### CLI Options

```bash
python upi_trm_train.py --help
```

**Core flags:**
- `--dataset-paths`: Path(s) to Sudoku dataset directories
- `--config`: YAML config file for RLConfig overrides
- `--train-steps`: Number of training steps (overrides config)
- `--batch-size`: Mini-batch size (overrides config)
- `--max-edits`: Max edits per episode
- `--seed`: Random seed for reproducibility
- `--tqdm`: Enable progress bar (disabled by default for cleaner log output)
- `--debug-checks`: Enable debug assertions and verbose logging

**Model architecture (match pretrained model if loading):**
- `--hidden-size`: Hidden dimension of TRM (default: 64, pretrained often use 128)
- `--h-cycles`: Number of H (outer) cycles
- `--l-cycles`: Number of L (inner) cycles  
- `--l-layers`: Number of transformer layers

**Puzzle embeddings (per-instance learnable vectors):**
- `--puzzle-emb-ndim`: Embedding dimension (0 to disable, typically hidden_size)
- `--puzzle-emb-len`: Embedding sequence length (default: 16)
- `--puzzle-emb-lr`: Learning rate for embeddings (uses SignSGD)
- `--puzzle-emb-weight-decay`: Weight decay for embeddings

**Checkpointing:**
- `--load-checkpoint`: Path to pretrained checkpoint to initialize from
- `--resume-checkpoint`: Path to RL checkpoint to resume training
- `--checkpoint-dir`: Directory to save checkpoints
- `--save-interval`: Save checkpoint every N steps (0 to disable)

**WandB logging:**
- `--wandb-project`: Project name (default: UPI-TRM-RL)
- `--wandb-run-name`: Run name (auto-generated if not set)
- `--wandb-offline`: Run in offline mode (no cloud sync)
- `--no-wandb`: Disable WandB entirely

#### Smoke Testing Without a Dataset

If you don't have the Sudoku dataset, the script automatically falls back to a `DummyPuzzleDataset` for testing:

```bash
python upi_trm_train.py \
    --train-steps 100 \
    --batch-size 16 \
    --max-edits 8 \
    --seed 0
```

This runs on synthetic data to verify the training loop works correctly.

#### Monitoring Theory Metrics

To track the paper's theoretical quantities (C_z, L_z, L_v, Bellman residual, unrolling term), enable theory metrics:

```bash
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml \
    --seed 42
```

Set `track_theory_metrics: true` in the YAML you are running to log:
- `hat_Cz`: Estimated ||z^(1) - z^(0)|| bound
- `hat_Lz`: Local Lipschitz estimate of inner map
- `hat_Lv`: Lipschitz estimate of value head w.r.t. z
- `unrolling_term`: L_V · L_z^n · C_z / (1 - L_z) (finite unrolling bias bound)
- `bellman_residual_*`: Empirical Bellman residual statistics

## Ablation Experiments
Use the directories under `configs/` as the source of truth for available experiment families. The current paper plan lives in `/home/buiksat/trm_bellman/documents/UPI_TRM_NIPS/NIPS_PLAN.md`.

## Evaluating a Trained Policy (Optional)
To run policy-only evaluation outside the training loop, load the model checkpoint and call `evaluate_plan_policy` (strict success) or `evaluate_plan_policy_with_scores` (mean score + success):
```python
import torch
from evaluators.rl_plan_evaluator import evaluate_plan_policy, evaluate_plan_policy_with_scores
from rl.envs.plan_edit_env import PlanEditEnvConfig
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.sudoku_checkers import dummy_checker
from rl.training_setup import DummyPuzzleDataset

model = TinyRecursiveReasoningModel_ACTV1({...})  # load weights/checkpoint
model.load_state_dict(torch.load("path/to/checkpoint.pt"))
model.to("cuda" if torch.cuda.is_available() else "cpu")

dataset = DummyPuzzleDataset()
env_cfg = PlanEditEnvConfig(max_edits=8, gamma=0.99, reward_shaping=True)
success = evaluate_plan_policy(
    model=model,
    dataset=dataset,
    checker=dummy_checker,
    env_cfg=env_cfg,
    num_episodes=50,
    inner_unroll_n=RLConfig().inner_unroll_n,
)
print(f"Success rate: {success:.3f}")

mean_score, success_rate = evaluate_plan_policy_with_scores(
    model=model,
    dataset=dataset,
    checker=dummy_checker,
    env_cfg=env_cfg,
    num_episodes=50,
    inner_unroll_n=RLConfig().inner_unroll_n,
)
print(f"Mean checker score: {mean_score:.3f} | Success rate: {success_rate:.3f}")
```

## Experimental Notes
- **Latent evaluator** \(U_n(s)\): `TinyRecursiveReasoningModel_ACTV1.used_value()` unrolls the latent state and applies the `value_head`. Note: value-head spectral norm is optional and must be disabled (`disable_value_head_norm: true`) for stability-dial experiments due to the known collapse mode.
- **K-step operator**: `UPITrmTrainer.value_update()` pulls trajectories from replay and mixes 1-step / K-step returns via `RLConfig.K`.
- **CPI mixture**: `_mixed_policy_dist()` blends the frozen and candidate policies, while `_sync_policy_old_towards_candidate()` softly updates the target head. These are tested in `tests/test_cpi_mixture_policy_smoke.py`.
- **Logging / evaluation**: `UPITrmTrainer.evaluate_policy_metrics()` (and the success-rate-only alias) surface both strict solves and mean checker scores via `upi_trm_train.py`; `RLConfig` toggles log/eval cadence.
- **K-step sweeps**: tweak `RLConfig.K` (either directly in `rl/config.py` or by editing the dataclass instantiation) before running `scripts/run_k_step_experiment.sh`.

## Reproducibility & Random Seeds
- `DummyPuzzleDataset` builds tokens with `torch.randint`, so set `torch.manual_seed(seed)` before instantiating it for determinism.
- RL replay sampling and action sampling rely on PyTorch’s RNG; seed `torch` and `random` at the start of `upi_trm_train.py` (or your custom entry point) for reproducible smoke tests.

## Quick Helper Scripts
- `./scripts/run_rl_dummy.sh` – ~200 training steps with batch size 16.
- `./scripts/run_tests.sh` – convenience wrapper around `pytest -v` (forwards extra CLI args).
- `./scripts/run_k_step_experiment.sh` – same as the dummy run but intended for adjusting `RLConfig.K` between launches.
- `./scripts/run_sudoku_rl_full.sh` – full-featured training with puzzle embeddings, checkpointing, and WandB.

---

## Legacy Supervised TRM Instructions
The original README for the "Less is More: Recursive Reasoning with Tiny Networks" paper is preserved below for context on the supervised pretraining pipelines.

### Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

#### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

#### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

#### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

#### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme (9x9)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Sudoku 4x4 (mixed curriculum)
buck2 run //buiksat_trm:build_4x4_sudoku -- \
  --output-dir buiksat_trm/data/sudoku-4x4 \
  --num-puzzles 1000

# True ultra-easy 4x4 (1-4 empties, 500 puzzles)
buck2 run //buiksat_trm:build_4x4_trivial -- \
  --output-dir buiksat_trm/data/sudoku-4x4-trivial \
  --num-puzzles 500 \
  --seed 42

# Hard 4x4 paper split (6-8 empties, 1000 puzzles)
buck2 run //buiksat_trm:build_4x4_sudoku -- \
  --output-dir buiksat_trm/data/sudoku-4x4-easy_6to8empties \
  --num-easy 1000 --num-medium 0 --num-hard 0 \
  --seed 42

# Legacy compatibility path used by older scripts / eval metadata
buck2 run //buiksat_trm:build_4x4_sudoku -- \
  --output-dir buiksat_trm/data/sudoku-4x4-ultra-easy \
  --num-easy 500 --num-medium 0 --num-hard 0 \
  --seed 42

# Validate all 4x4 splits
buck2 run //buiksat_trm:inspect_4x4_dataset

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

### Experiments

#### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

#### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

#### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

#### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

### Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).

## Quick Start: Working Sudoku Solver

For a working 4x4 Sudoku solver, use imitation learning from oracle:

```bash
# Train and demo (100% solve rate)
python imitation_train.py --dataset-paths data/sudoku-4x4-trivial --num-epochs 100

# Output: 50/50 puzzles solved, step-by-step solving demos
```

**Key Finding**: Pure RL exploration cannot discover Sudoku solutions from scratch. Imitation learning from oracle is required for this task.
