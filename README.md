# Unrolled Policy Iteration in Plan Space for Tiny Recursive Models (UPI–TRM)

This repository extends the Tiny Recursive Model (TRM) codebase with a plan-space reinforcement learning framework we call **UPI–TRM**. It adds latent value estimation (`U_n(s)`), a configurable K-step value operator, and conservative policy improvement (CPI) mixture updates so that TRMs can be trained and evaluated with lightweight RL loops.

## Repository Structure
- `models/recursive_reasoning/trm.py` – `TinyRecursiveReasoningModel_ACTV1` with RL-specific value/policy heads.
- `models/value_head.py` – latent value head \(V_\psi\) with spectral normalization utilities from `utils/lipschitz.py`.
- `models/edit_policy.py` – autoregressive edit policy head that proposes plan-space actions.
- `rl/config.py` – `RLConfig` for hyperparameters, logging cadence, CPI knobs, and evaluation intervals.
- `rl/envs/plan_edit_env.py` – plan-space meta-MDP describing edit actions over latent plans.
- `rl/upi_trm_trainer.py` – trainer implementing 1-step + K-step TD, CPI mixtures, and evaluation hooks.
- `upi_trm_train.py` – main entry point that wires the TRM, env, dummy dataset, and trainer together.
- `evaluators/rl_plan_evaluator.py` – helper to measure success rate of a trained plan policy.
- `tests/` – smoke tests and unit tests for heads, TD targets, CPI mixture, logging, and evaluator glue.
- `scripts/` – ready-to-run helpers: `run_rl_dummy.sh`, `run_tests.sh`, `run_k_step_experiment.sh`.

## Installation
```bash
git clone <repo-url>
cd trm_bellman
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt  # torch, tqdm, pytest, and the original TRM deps
```
If you prefer not to use `requirements.txt`, minimally install `torch`, `tqdm`, and `pytest`.

## Running Tests
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
This uses the in-memory `DummyPuzzleDataset`, collects short plan-edit episodes, and prints both policy/value losses plus `evaluate_policy_success_rate` estimates. The `scripts/run_rl_dummy.sh` helper runs an equivalent configuration with a smaller batch size (16) to finish even faster.

## Evaluating a Trained Policy (Optional)
To run policy-only evaluation outside the training loop, load the model checkpoint and call `evaluate_plan_policy`:
```python
import torch
from evaluators.rl_plan_evaluator import evaluate_plan_policy
from rl.envs.plan_edit_env import PlanEditEnvConfig
from upi_trm_train import DummyPuzzleDataset, dummy_checker
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig

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
```

## Experimental Notes
- **Latent evaluator** \(U_n(s)\): `TinyRecursiveReasoningModel_ACTV1.used_value()` unrolls the latent state and applies the spectral-normalized `value_head`.
- **K-step operator**: `UPITrmTrainer.value_update()` pulls trajectories from replay and mixes 1-step / K-step returns via `RLConfig.K`.
- **CPI mixture**: `_mixed_policy_dist()` blends the frozen and candidate policies, while `_sync_policy_old_towards_candidate()` softly updates the target head. These are tested in `tests/test_cpi_mixture_policy_smoke.py`.
- **Logging / evaluation**: `UPITrmTrainer.evaluate_policy_success_rate()` is surfaced via `upi_trm_train.py`, and `RLConfig` toggles log/eval cadence.
- **K-step sweeps**: tweak `RLConfig.K` (either directly in `rl/config.py` or by editing the dataclass instantiation) before running `scripts/run_k_step_experiment.sh`.

## Reproducibility & Random Seeds
- `DummyPuzzleDataset` builds tokens with `torch.randint`, so set `torch.manual_seed(seed)` before instantiating it for determinism.
- RL replay sampling and action sampling rely on PyTorch’s RNG; seed `torch` and `random` at the start of `upi_trm_train.py` (or your custom entry point) for reproducible smoke tests.

## Quick Helper Scripts
- `./scripts/run_rl_dummy.sh` – ~200 training steps with batch size 16.
- `./scripts/run_tests.sh` – convenience wrapper around `pytest -v` (forwards extra CLI args).
- `./scripts/run_k_step_experiment.sh` – same as the dummy run but intended for adjusting `RLConfig.K` between launches.

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

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

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
