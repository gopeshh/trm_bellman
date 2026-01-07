# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## User Preferences

**IMPORTANT: Always use Meta Buck2 for everything** (building, running, testing). Do not use pip/python directly.

```bash
# Standard Buck2 command pattern for this project
buck2 run //buiksat_trm:<target> \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    -- <args>
```

## Project Overview

UPI-TRM (Unrolled Policy Iteration for Tiny Recursive Models) extends the Tiny Recursive Model with reinforcement learning. The codebase implements a theory-exact RL framework based on an ICML 2026 submission, combining supervised learning and plan-space RL for constraint satisfaction problems (Sudoku, ARC-AGI, mazes).

**Context**: I'm extending the Tiny Recursive Models paper into a Reinforcement Learning framework, targeting ICML 2026 submission.

**Background**:
- Original TRM paper: https://arxiv.org/pdf/2510.04871v1
- Original repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- ICML 2026 paper: `$HOME/UPI_TRM/UPI_TRM_ICML/main.tex`

**Core Innovation**: A ~7M parameter model that recursively refines solutions through:
- Inner loop: Latent state recursion (z^(0) → z^(n)) for reasoning
- Outer loop: Plan editing via discrete actions in a meta-MDP

**Project Status** (as of baseline implementation):
- ✅ Task 1: Shaped-reward Sudoku infrastructure complete
  - 9 configs created (2 main + 7 ablations)
  - All configs validated
  - Scripts ready: `run_shaped_reward_experiments.sh`, `verify_configs.py`
- ✅ Task 2: Baseline algorithms (PPO, A2C) and NoRecursionEncoder
  - `rl/algos/ppo.py` - PPO trainer for comparison
  - `rl/algos/a2c.py` - A2C trainer for comparison
  - `models/norec_encoder.py` - MLP/Transformer baseline without TRM recursion
  - Configs in `configs/baselines/`
- ✅ Task 3: Paper figure/table generation scripts
  - `scripts/aggregate_runs.py` - WandB data export
  - `scripts/plot_learning_curves.py` - ICML-ready learning curves
  - `scripts/plot_ablations.py` - Ablation bar charts
  - `scripts/make_tables.py` - LaTeX table generation
  - `scripts/stats_tests.py` - Statistical significance tests
- ✅ Task 4: UNDO action and curriculum training
  - UNDO action in `rl/envs/plan_edit_env.py` (enable with `enable_undo: true`)
  - `scripts/run_curriculum.py` - 4x4 → 9x9 curriculum training
- 🔄 Task 5: Run experiments and generate paper figures
- 📚 Documentation: See `DOCUMENTATION_INDEX.md` for navigation guide

## Essential Commands

### Testing (Buck2)

```bash
# Navigate to fbcode directory first
cd ~/fbsource/fbcode

# Run all 22 test targets (117 tests total)
buck2 test //buiksat_trm:test_... \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --local-only

# Run a single test target
buck2 test //buiksat_trm:test_rl_k_step_targets \
    -c fbcode.nvcc_arch=a100 \
    -c fbcode.enable_gpu_sections=true \
    --local-only

# Clear Buck2 cache (if tests are stale after BUCK file changes)
buck2 clean
```

**Available test targets (22 total, 117 tests):**
- `test_rl_k_step_targets` - K-step bootstrapped target computation
- `test_theory_exact_components` - Theory-exact features (exact baseline, contraction)
- `test_refactored_modules` - Refactored RL modules (replay buffer, task configs)
- `test_upi_trm_trainer_smoke` - End-to-end training smoke test
- `test_cpi_mixture_policy_smoke` - CPI mixture mechanics
- `test_upi_trm_logging_smoke` - Metric logging
- `test_plan_edit_env` - Environment dynamics
- `test_plan_edit_env_reward_shaping` - Reward shaping
- `test_gae` - GAE computation
- `test_trm_latent_unroll` - Inner recursion APIs
- `test_trm_rl_heads` - RL head modules
- `test_edit_policy_head` - Edit policy head
- `test_lipschitz_spectral_norm` - Lipschitz/spectral norm utilities
- `test_theory_metrics` - Theory metrics computation
- `test_rl_plan_evaluator_smoke` - Plan policy evaluation
- `test_rl_k_step_value_update_trainer` - K-step value update
- `test_z_init_encoder` - Z-init encoder tests
- `test_baselines` - Baseline algorithms (PPO, A2C, NoRecursionEncoder)
- `test_undo_and_sequences` - UNDO action and sequence sampling
- `test_sudoku_checkers` - Sudoku checker functions (constraint/progress)
- `test_config_integrity` - Config file validation
- `test_convergence_smoke` - Training convergence smoke test

### RL Training

**Quick smoke test (no dataset required):**
```bash
python upi_trm_train.py --train-steps 100 --batch-size 16 --max-edits 8 --seed 0
./scripts/run_rl_dummy.sh
```

**4×4 Sudoku (recommended for development):**
```bash
# Ultra-easy variant (fastest convergence, 1-4 empty cells)
python upi_trm_train.py \
    --dataset-paths data/sudoku-4x4-ultra-easy \
    --config configs/rl_sudoku_4x4_ultra_easy.yaml \
    --seed 42

# Full-featured with puzzle embeddings and WandB
./scripts/run_sudoku_rl_full.sh 4x4
```

**9×9 Sudoku (extreme difficulty):**
```bash
# Baseline K=1
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1.yaml \
    --seed 42

# Theory-exact K=1
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1_theory_exact.yaml \
    --seed 42

# K=3 multi-step
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k3_baseline.yaml \
    --seed 42
```

**Shaped Reward Experiments (ICML 2026 Submission):**
```bash
# Verify all configs are valid
python scripts/verify_configs.py

# Quick test (single experiment)
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_shaped_theory_exact.yaml \
    --seed 42

# Full experimental suite (9 configs × 3 seeds = 27 experiments)
# Dataset: data/sudoku-extreme-1k-aug-1000
# W&B project: UPI-TRM-ICML-Shaped-Rewards
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 20000

# Pilot run (faster, for testing)
./scripts/run_shaped_reward_experiments.sh --seeds 1 --steps 5000

# Parallel execution (requires GNU parallel)
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 20000 --parallel
```

**Bahram's Experimental Requirements** (from paper lines 1426-1429):
- ✅ Shaped-reward Sudoku setting (checker + potential shaping + absorbing normalization)
- ✅ Explicit action space, masking, termination, horizon T, discount γ
- ✅ Ablations removing: (i) contraction, (ii) exact centering, (iii) conservative mixture
- ⏭️ Run experiments and collect results for paper Section 6

**Baseline Algorithms (PPO, A2C):**
```bash
# Use --baseline flag to select algorithm and --backbone for model architecture
# Available baselines: ppo, a2c
# Available backbones: trm (default), norec-mlp, norec-transformer

# PPO with MLP backbone (simplest baseline)
python upi_trm_train.py \
    --baseline ppo \
    --backbone norec-mlp \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --train-steps 10000 \
    --seed 42

# A2C with Transformer backbone
python upi_trm_train.py \
    --baseline a2c \
    --backbone norec-transformer \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --train-steps 10000 \
    --seed 42

# PPO with TRM backbone (tests if improvement comes from algorithm vs architecture)
python upi_trm_train.py \
    --baseline ppo \
    --backbone trm \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --train-steps 10000 \
    --seed 42

# Available baseline configs (alternative to CLI flags):
# - configs/baselines/ppo_trm_sudoku.yaml   (PPO + TRM backbone)
# - configs/baselines/a2c_trm_sudoku.yaml   (A2C + TRM backbone)
# - configs/baselines/ppo_norec_sudoku.yaml (PPO + MLP encoder)
# - configs/baselines/a2c_norec_sudoku.yaml (A2C + MLP encoder)
```

**UNDO Action Variant:**
```bash
# Enable UNDO action that lets agent revert to previous plan states
python upi_trm_train.py \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_undo.yaml \
    --seed 42
```

**Curriculum Training (4×4 → 9×9):**
```bash
# Two-stage curriculum: start on 4×4, transfer to 9×9
python scripts/run_curriculum.py \
    --stage1-dataset data/sudoku-4x4-ultra-easy \
    --stage2-dataset data/sudoku-extreme-1k-aug-1000 \
    --stage1-steps 2000 \
    --total-steps 10000 \
    --config configs/rl_sudoku_shaped_theory_exact.yaml
```

**Imitation learning pretraining:**
```bash
python imitation_train.py --dataset-paths data/sudoku-4x4-ultra-easy --num-epochs 100
```

### Supervised Pretraining

**Single GPU (Sudoku):**
```bash
python pretrain.py \
    arch=trm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
    +run_name=pretrain_sudoku ema=True
```

**Multi-GPU (ARC-AGI with torchrun):**
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    pretrain.py \
    arch=trm \
    data_paths="[data/arc1concept-aug-1000]" \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
    +run_name=pretrain_arc1 ema=True
```

### Dataset Preparation

```bash
# 4×4 Sudoku (curriculum learning)
python dataset/build_4x4_sudoku.py --output-dir data/sudoku-4x4 --num-puzzles 1000
python dataset/build_4x4_sudoku.py --output-dir data/sudoku-4x4-ultra-easy --num-easy 500 --num-medium 0 --num-hard 0

# 9×9 Sudoku (extreme)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# ARC-AGI-1
python -m dataset.build_arc_dataset \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/arc1concept-aug-1000 \
    --subsets training evaluation concept \
    --test-set-name evaluation

# Mazes
python dataset/build_maze_dataset.py
```

### Checkpoint Management

```bash
# Load pretrained checkpoint and fine-tune with RL
python upi_trm_train.py \
    --load-checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1/checkpoint.pt \
    --dataset-paths data/sudoku-extreme-1k-aug-1000 \
    --config configs/rl_sudoku_k1.yaml

# Save RL checkpoints during training
python upi_trm_train.py \
    --checkpoint-dir checkpoints/rl_run \
    --save-interval 1000 \
    --train-steps 10000
```

### Paper Figure Generation (ICML 2026)

```bash
# Step 1: Aggregate runs from WandB
python scripts/aggregate_runs.py \
    --project UPI-TRM-ICML-Shaped-Rewards \
    --output artifacts/summary.parquet \
    --compute-stats

# Step 2: Generate learning curve figures
python scripts/plot_learning_curves.py \
    --input artifacts/summary.parquet \
    --output paper/figures/

# Step 3: Generate ablation bar chart
python scripts/plot_ablations.py \
    --input artifacts/summary.parquet \
    --output paper/figures/ablation_bar_chart.pdf

# Step 4: Generate LaTeX tables
python scripts/make_tables.py \
    --input artifacts/summary.parquet \
    --output paper/tables/

# Step 5: Run statistical significance tests
python scripts/stats_tests.py \
    --input artifacts/summary.parquet \
    --output paper/tables/significance.tex \
    --test bootstrap \
    --alpha 0.05
```

## Architecture Overview

### Dual Recursion Model

The TRM backbone (`models/recursive_reasoning/trm.py:TinyRecursiveReasoningModel_ACTV1`) implements two recursive loops:

1. **Inner Recursion (Latent Reasoning)**:
   - `init_latent(x, y) → z^(0)`: Initialize latent state from puzzle and plan
   - `update_latent(z, y, x) → z^(t+1)`: One step of latent refinement
   - `unroll_latent(x, y, n) → z^(n)`: Run n inner steps
   - Enforces contraction via spectral normalization (Lipschitz L_z < 1)

2. **Outer Loop (Plan Editing)**:
   - Meta-MDP state: `s = (x, y)` where x=puzzle, y=candidate solution
   - Actions: Discrete edits (position + value) or STOP
   - Policy: `π(a|s)` from `EditPolicyHead`
   - Value: `U_n(s) = V_ψ(z^(n)(s), x)` from `LatentValueHead`

### RL Framework

The UPI-TRM algorithm (`rl/upi_trm_trainer.py:UPITrmTrainer`) implements:

1. **K-step Bootstrapped Targets** (`rl/value_targets.py`):
   - `G^(K)(s) = Σ_k γ^k r_k + γ^K U_n(s_K)`
   - With configurable K ∈ {1, 3, 5} for bias-variance tradeoff

2. **Conservative Policy Improvement (CPI)**:
   - Policy-space mixture: `π_new = (1-α)π_old + α π_candidate`
   - Parameter-space interpolation: `param.lerp_(candidate_param, α)`
   - Controlled by `mixture_alpha` (typically 0.01-0.1)
   - See "CPI Mixture Policy Modes" section below for theory details

3. **Plan-Edit Environment** (`rl/envs/plan_edit_env.py`):
   - Potential-based reward shaping: `r = r_0 + γΦ(s') - Φ(s)`
   - Potential Φ derived from constraint checker function
   - STOP action modes: "terminal", "noop", "disabled"

### Theory-Exact Features

The `RLConfig` dataclass (`rl/config.py`) provides "theory dials" corresponding to ICML paper theorems:

**Critical for Theorem 5.9 (O(α·ε_A) bound):**
- `exact_baseline_summation=True`: Computes `E_{a~π}[Q̂(s,a)]` via exact summation over ALL discrete actions. This ensures `E_{a~π}[Â(s,a)] = 0` exactly per state, enabling the tight O(α·ε_A) bound instead of O(ε_A/(1-γ)).
- `exact_k_step_targets=True`: Uses fixed-horizon γ^K bootstrap
- `theory_exact_mixture=True`: Policy-space mixture (not parameter-space)

**For contraction guarantees (Assumption 4.2):**
- `enable_contraction=True`: Apply spectral normalization to latent map and value head
- `target_Lz < 1.0`: Target Lipschitz constant for latent recursion (typically 0.9)
- `latent_ball_radius > 0`: Forward-invariant projection (Assumption 4.1, typically 10.0)

**Heuristics (not covered by theory):**
- `distill_mixture_policy=True`: Distill mixture into single network (Section 6.5)
- `batch_centered_advantage=True`: Batch-level mean subtraction (NOT exact centering)
- `use_gae=True`: Generalized Advantage Estimation (Section 6.2)

Check theory alignment: `RLConfig.validate_theory_alignment()` or `RLConfig.is_theory_exact()`

## Key Implementation Details

### Latent State Modes

**Episodic (default)**: `episodic_latent=True`
- z reinitialized from (x, y) at every step
- Stateless reasoning per action

**Persistent**: `episodic_latent=False`
- z initialized once per episode, updated across steps
- RNN-like behavior, critical for harder tasks
- See `configs/rl_sudoku_k1_persistent_z.yaml`

### STOP Action Control

Addresses "rush-to-fail" problem (Remark 2.6):

- `stop_action_mode="terminal"`: STOP ends episode (standard RL)
- `stop_action_mode="noop"`: STOP is no-op with penalty, prevents early collapse
- `stop_action_mode="disabled"`: STOP masked out, forced to use full budget

### Reward Shaping

**Shaped (default)**: `reward_shaping=True`
- Dense feedback: `r = r_0 + γ·checker(x,y') - checker(x,y)` (Paper Equation 4)
- Terminal rewards: `fail_terminal_reward`, `solve_terminal_reward`
- For Sudoku: set `fail_terminal_reward = -10.0` (= -C_max) to prevent rush-to-fail (Remark 2.6)
- See `configs/rl_sudoku_shaped_theory_exact.yaml` for theory-exact setup

**Sparse**: `reward_shaping=False`
- Only terminal reward from checker function
- Harder credit assignment (slower learning)
- See `configs/rl_sudoku_sparse_theory_exact.yaml` for baseline comparison

**Rush-to-Fail Mitigation** (Paper Remark 2.6):
- Problem: With shaped rewards, agent may learn to terminate early on failures
- Solution: Set `fail_terminal_reward ≤ -γ·C_max` to penalize early termination
- For Sudoku with γ=0.99 and C_max=10.0: use `fail_terminal_reward = -10.0`
- This ensures continuing to improve is always better than giving up

### Checker Functions

Three checker types are available for 4×4 Sudoku:

**1. Solution Checker** (default): `use_constraint_checker=False, use_progress_checker=False`
- Compares plan to known solution
- Score = percentage match × 10 (range 0-10)
- Dense signal but requires solution labels

**2. Constraint Checker**: `use_constraint_checker=True`
- Counts row/column/box violations
- Score = `10 × (1 - violations/24)` (range 0-10)
- Initial score = 10.0 (empty cells ignored)
- Limitation: Can't distinguish partial from solved (both = 10.0)

**3. Progress Checker** (NEW): `use_progress_checker=True`
- Score = filled_cells (if no violations) or filled_cells - penalty
- Initial score = number of clue cells (e.g., 12)
- Solved score = 16 (all cells filled for 4×4)
- Most informative: clearly distinguishes progress

**Config example for progress checker**:
```yaml
use_progress_checker: true
solved_threshold: 16.0  # All 16 cells filled
fail_terminal_reward: -16.0  # Rush-to-fail mitigation
```

### Puzzle Embeddings

Per-instance learnable vectors (inspired by test-time training):

```bash
# Enable with puzzle-emb-ndim (typically equals hidden_size)
python upi_trm_train.py \
    --puzzle-emb-ndim 128 \
    --puzzle-emb-len 16 \
    --puzzle-emb-lr 0.01 \
    --puzzle-emb-weight-decay 0.1
```

Uses SignSGD optimizer (`models/sparse_embedding.py`). Disabled by default (ndim=0).

### Spectral Normalization

Enforces Lipschitz bounds for theoretical guarantees (`utils/lipschitz.py`):

- `apply_spectral_norm_to_trm()`: Apply to all z→z path layers
- `estimate_Lz()`: Compute empirical Lipschitz constant
- `compute_unrolling_term()`: Finite unrolling bias (Equation 10)

Tracked when `track_theory_metrics=True`: logs `hat_Cz`, `hat_Lz`, `hat_Lv`, `unrolling_term`.

### CPI Mixture Policy Modes

The implementation supports three CPI (Conservative Policy Improvement) modes with different theoretical properties. **For strict theoretical correctness, use `theory_exact_mixture=True`**.

**Why Importance Sampling Weights Are NOT Required** (in `theory_exact_mixture=True` mode):

A common concern is that data collected from the mixture policy `π_mix = (1-α)π_old + α·π_cand` should require importance sampling (IS) weights `ρ = π_cand(a|s) / π_mix(a|s)` when training `π_cand`. This is **not** the case in CPI because:

1. **The deployed policy IS the mixture**: We don't try to make `π_cand` behave like `π_mix`. Instead, we train `π_cand` with standard policy gradient, keep `π_old` fixed, and deploy the explicit mixture.

2. **CPI's guarantee is about the mixture**: The improvement bound `V^{π_new} ≥ V^{π_old} - O(α·ε_A)` applies to the **deployed mixture policy**, not to `π_cand` in isolation.

3. **The "bias" is intentional**: When we update `π_cand` using data from `π_mix`, we're improving `π_cand` relative to the current state distribution. CPI theory accounts for this.

**Three CPI Modes** (in `rl/upi_trm_trainer.py:policy_update()`):

| Mode | Config | Theory Status | Description |
|------|--------|---------------|-------------|
| **Theory-Exact** | `theory_exact_mixture=True` | ✅ CPI bound applies | `π_old` is **not updated**; behavior policy is explicit mixture from `_mixed_policy_dist()` |
| **Distillation** | `distill_mixture_policy=True` | ⚠️ Heuristic | Mixture is distilled into `π_old` via KL minimization; introduces projection error |
| **Default** | Both `False` | ⚠️ Heuristic | Parameter-space interpolation `param.lerp_(candidate, α)`; NOT equivalent to probability mixing |

**Recommendation**: For ICML 2026 experiments, always use `theory_exact_mixture=True` in theory-aligned configs. The default parameter interpolation mode is faster but does not satisfy CPI guarantees.

**Implementation Details** (`rl/upi_trm_trainer.py`):
- Lines 1085-1110: Mode selection logic
- Lines 222-284: `_mixed_policy_dist()` computes explicit mixture
- Lines 286-327: `_sync_policy_old_towards_candidate()` implements parameter interpolation

**Key Files**: `rl/upi_trm_trainer.py`, `rl/config.py:theory_exact_mixture`

## Configuration System

### YAML Configs vs CLI Args

**YAML files** (`configs/*.yaml`): Override `RLConfig` fields
**CLI args**: Override model architecture, paths, and top-level training params

Example precedence:
```bash
# YAML sets K=3, CLI overrides to train-steps=5000
python upi_trm_train.py \
    --config configs/rl_sudoku_k3_baseline.yaml \  # K=3 from YAML
    --train-steps 5000                             # Overrides YAML
```

### Common Config Combinations

**Fast debugging** (4×4, 80 actions):
- `configs/rl_sudoku_4x4_ultra_easy.yaml`
- Small action space, shaped rewards, noop STOP

**Theory-exact baseline** (9×9):
- `configs/rl_sudoku_k1_theory_exact.yaml`
- All paper features: exact baseline, exact targets, theory-exact mixture

**Shaped rewards experiments** (ICML 2026 submission):
- `configs/rl_sudoku_shaped_theory_exact.yaml` - Main contribution (all features ON)
  - `reward_shaping=true`, `fail_terminal_reward=-10.0` (rush-to-fail mitigation)
  - All theory-exact features enabled
- `configs/rl_sudoku_sparse_theory_exact.yaml` - Sparse baseline
  - `reward_shaping=false`, terminal-only rewards
- `configs/ablations/ablation_*.yaml` - 7 ablation configs
  - `ablation_no_exact_baseline.yaml` - **Tests Theorem 5.9 (KEY)**
  - `ablation_no_contraction.yaml` - Tests Assumption 4.2
  - `ablation_no_conservative_mixture.yaml` - Tests CPI benefit (α=1.0)
  - And 4 more systematic ablations

**Stable training** (persistent latent):
- `configs/rl_sudoku_k1_persistent_z.yaml`
- High entropy (0.1), disabled STOP, constant LR

**Multi-step unrolling** (K=5):
- `configs/rl_sudoku_k5_theory_exact.yaml`
- Higher variance but less bias

### Environment Variables

For `run_sudoku_rl_full.sh`:
```bash
HIDDEN_SIZE=128 \
PUZZLE_EMB_NDIM=128 \
TRAIN_STEPS=20000 \
SEED=42 \
./scripts/run_sudoku_rl_full.sh 4x4
```

## Critical Code Paths

### Data Collection Flow

1. `UPITrmTrainer.collect_rollouts()` samples puzzle x from dataset
2. `PlanEditEnv.reset()` initializes plan y (empty or heuristic)
3. For each step:
   - `model.unroll_latent(x, y, n)` → z^(n)
   - `model.policy_dist(x, y, n)` → π(a|s) via EditPolicyHead
   - Sample action a, `env.step(a)` → (s', r, done)
   - Store transition in `ReplayBuffer`

### Value Update

1. `UPITrmTrainer.value_update()` samples batch from replay
2. `compute_k_step_bootstrapped_target()` computes G^(K)
3. `model.used_value(x, y, n)` computes U_n(s) via LatentValueHead
4. MSE loss: `(U_n - stopgrad(G^(K)))^2`
5. Update value_head parameters ψ

### Policy Update

1. `UPITrmTrainer.policy_update()` samples batch
2. Compute advantages: `A = r + γU_n(s') - U_n(s)`
3. If `exact_baseline_summation=True`:
   - Enumerate ALL actions: `[a_0, a_1, ..., a_N]`
   - Compute `Q̂(s,a) = r(s,a) + γU_n(s')` for all a
   - Exact baseline: `b(s) = Σ_a π(a|s)Q̂(s,a)`
   - Centered advantage: `Â(s,a) = Q̂(s,a) - b(s)` (guarantees E[Â]=0 per state)
4. Policy gradient: `-log π(a|s) · Â(s,a) - β·H(π)`
5. If `theory_exact_mixture=True`: Deploy `π_new = (1-α)π_old + α π_candidate`
6. Else: Parameter interpolation `param.lerp_(candidate, α)`

### Evaluation

1. `UPITrmTrainer.evaluate_policy_metrics()` runs greedy rollouts
2. `evaluate_plan_policy_with_scores()` computes success rate and mean checker score
3. Logs to WandB/stdout every `eval_interval` steps

## Common Patterns

### Adding a New Task

1. **Dataset builder** (`dataset/build_<task>_dataset.py`):
   - Implement `generate_puzzles()` → list of (x, y_solution) pairs
   - Save as `{train,val,test}.jsonl`

2. **Checker function** (in `upi_trm_train.py` or separate module):
   - Signature: `checker(x: Tensor, y: Tensor) → float`
   - Return constraint satisfaction score (higher = better)

3. **Edit actions** (in `PlanEditEnv` or custom subclass):
   - Define action space (e.g., grid edits, sequence insertions)
   - Implement `_apply_action(y, action)` → y'

4. **Config**:
   - Create `configs/rl_<task>_k1.yaml` with task-specific hyperparams
   - Set `task_name`, `max_edits`, `solved_threshold`

### Debugging Training Issues

**Policy collapse to STOP**:
- Set `stop_action_mode="disabled"` or `"noop"`
- Increase `entropy_coef` (try 0.1)
- Use shaped rewards: `reward_shaping=True`

**Value divergence**:
- Enable `value_target_clip` (default 10.0)
- Lower `value_lr` (try 1e-4)
- Increase `target_ema_tau` (try 0.999)

**Theory metrics violations**:
- Check `hat_Lz < 1.0`: if not, lower `target_Lz` or increase spectral norm power iterations
- Check `latent_ball_radius > 0`: projection must be enabled
- Validate with `config.validate_theory_alignment()`

### Working with Checkpoints

**Load pretrained backbone, freeze, train only RL heads**:
```python
# In upi_trm_train.py
load_checkpoint(model, pretrained_path, strict=False)
# Freeze backbone
for param in model.inner.parameters():
    param.requires_grad = False
# RL heads remain trainable
```

**Resume RL training**:
```bash
python upi_trm_train.py \
    --resume-checkpoint checkpoints/rl_run/step_5000.pt \
    --train-steps 10000  # Continue for 5000 more steps
```

## Testing Strategy

### Smoke Tests
Fast sanity checks (< 1 min each):
- `test_upi_trm_trainer_smoke.py`: End-to-end training loop
- `test_cpi_mixture_policy_smoke.py`: CPI mixture mechanics
- `test_upi_trm_logging_smoke.py`: Metric logging

### Unit Tests
Component correctness:
- `test_rl_k_step_targets.py`: K-step target computation
- `test_theory_exact_components.py`: Exact baseline, centered advantages
- `test_trm_latent_unroll.py`: Inner recursion APIs

### Integration Tests
- `test_plan_edit_env.py`: Environment dynamics, reward shaping
- `test_rl_evaluator.py`: Policy evaluation functions

Run full suite before submitting changes:
```bash
pytest -v --tb=short
```

## Paper-to-Code Mapping

| Paper Section | Implementation | Key Files |
|---------------|----------------|-----------|
| Section 2: Meta-MDP | Plan-edit environment | `rl/envs/plan_edit_env.py` |
| Section 3-4: Latent evaluator | U_n(s) = V_ψ(z^(n), x) | `models/value_head.py`, `trm.py:used_value()` |
| Assumption 4.1: Forward-invariant | Latent ball projection | `latent_ball_radius` in config |
| Assumption 4.2: Contraction | Spectral normalization | `utils/lipschitz.py` |
| Section 5: K-step operator | Bootstrapped targets | `rl/value_targets.py` |
| Eq. 12 (lines 677-678): V(s_abs) = -C_max | Terminal bootstrap | `C_max` in RLConfig, `value_targets.py` |
| Theorem 5.9: Exact baseline | Exact summation for discrete actions | `exact_baseline_summation` flag |
| Remark 2.6: Rush-to-fail | fail_terminal_reward ≤ -γC_max | `fail_terminal_reward` in RLConfig |
| Section 6: CPI | Mixture updates | `UPITrmTrainer._mixed_policy_dist()` |
| Algorithm 1: UPI-TRM | Main training loop | `rl/upi_trm_trainer.py` |

Equation references in docstrings link to paper equations (e.g., "Equation 10" = finite unrolling bias).

## Theory-Aligned Defaults

The codebase now uses theory-aligned defaults matching the ICML 2026 paper:

- **`C_max = 10.0`**: Maximum checker score (Sudoku scale [0, 10])
- **`fail_terminal_reward = -10.0`**: Satisfies Remark 2.6 (≤ -γC_max ≈ -9.9)
- **K-step terminal bootstrap**: Uses V(s_abs) = -C_max instead of 0 (Eq. 12)
- **`latent_ball_radius = 10.0`**: Forward-invariant projection enabled (Assumption 4.1)

To check theory alignment: `RLConfig.validate_theory_alignment()` or `RLConfig.is_theory_exact()`

## Shaped Reward Experiments Documentation

For the ICML 2026 submission shaped-reward experiments (Bahram's guidance from paper lines 1426-1429):

**Documentation Hub**: See `DOCUMENTATION_INDEX.md` for complete navigation guide to all documentation.

**Quick Start**: See `docs/QUICKSTART_SHAPED_REWARDS.md` for commands and expected results.

**Theory Alignment**: See `docs/shaped_rewards_theory.md` for:
- Paper Equation 4 implementation details
- Rush-to-fail mitigation (Remark 2.6)
- C_max = 10.0 for Sudoku checker
- Theory-exact feature checklist

**Full Details**: See `docs/shaped_rewards_implementation_complete.md` for:
- Expected results by config
- Metrics to track
- Experimental validation plan

**Key Implementation Details**:
- Shaped rewards already implemented in `PlanEditEnv.compute_transition_reward()` (rl/envs/plan_edit_env.py:375-443)
- Sudoku checker returns [0, 10] range (upi_trm_train.py:212-228)
- Set `fail_terminal_reward = -10.0` to enable rush-to-fail mitigation
- All 9 configs validated via `scripts/verify_configs.py`
- Run experiments via `scripts/run_shaped_reward_experiments.sh`
- Results logged to W&B project: `UPI-TRM-ICML-Shaped-Rewards`

**Configs Summary**:
- 2 main configs: shaped theory-exact (best), sparse theory-exact (baseline)
- 7 ablations: tests for contraction, exact baseline (KEY), conservative mixture, projection, etc.
- Total: 9 configs × 3 seeds = 27 experiments for publication

## Development Workflow

1. **Start with 4×4 Sudoku**: Fast iteration, easy debugging
2. **Use dummy dataset** for API testing: `python upi_trm_train.py --train-steps 100`
3. **Enable theory metrics** when tuning: `track_theory_metrics=True` in config
4. **Checkpoint frequently** during long runs: `--save-interval 1000`
5. **Compare K values** with ablation configs: `configs/ablations/upi_trm_K{1,3,5}.yaml`
6. **Monitor WandB**: `--wandb-project` for experiment tracking

## WandB Integration

```bash
# First-time setup
wandb login

# Run with WandB (generic experiments)
python upi_trm_train.py \
    --wandb-project UPI-TRM-RL \
    --wandb-run-name experiment-name

# Shaped reward experiments (ICML 2026 submission)
python upi_trm_train.py \
    --wandb-project UPI-TRM-ICML-Shaped-Rewards \
    --wandb-run-name shaped-theory-exact-seed42

# Offline mode (sync later)
python upi_trm_train.py --wandb-offline

# Disable WandB
python upi_trm_train.py --no-wandb
```

**W&B Project Naming Convention**:
- Generic experiments: `UPI-TRM-RL`
- Shaped reward experiments: `UPI-TRM-ICML-Shaped-Rewards`

Logged metrics:
- `loss/policy`, `loss/value`: Training losses
- `eval_success_rate`: % puzzles solved
- `eval_mean_score`: Average checker score
- `theory/hat_Lz`, `theory/hat_Cz`: Lipschitz/contraction estimates (if enabled)
- `policy/entropy`: Policy entropy (monitor for collapse)
