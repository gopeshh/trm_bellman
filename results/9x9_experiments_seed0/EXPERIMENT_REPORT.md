# UPI-TRM 9x9 Sudoku Experiment Report

**Date:** 2026-01-30
**Seed:** 0
**Training Steps:** 50,000

---

## 1. Experiment Configuration

### Algorithm: UPI-TRM (Unified Policy Iteration with Thinking Recursive Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `algorithm` | `upi_trm` | Main algorithm |
| `gamma` | 0.99 | Discount factor |
| `K` | 1 | Number of Bellman backup steps |
| `inner_unroll_n` | 1 | Latent unrolling steps per action |
| `max_edits` | 81 | Maximum edit steps per episode (full grid) |
| `num_train_steps` | 50,000 | Total training steps |
| `batch_size` | 64 | Batch size for training |
| `rollout_episodes_per_step` | 2 | Episodes collected per training step |

### Learning Rates & Optimization

| Parameter | Value |
|-----------|-------|
| `value_lr` | 0.0003 |
| `policy_lr` | 0.0001 |
| `value_grad_clip` | 1.0 |
| `policy_grad_clip` | 0.5 |

### Policy & Value Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `mixture_alpha` | 0.1 | CPI mixture coefficient |
| `policy_epsilon` | 0.1 | Exploration epsilon |
| `target_ema_tau` | 0.99 | Target network EMA coefficient |
| `entropy_coef` | 0.05 | Entropy regularization |

### Stability Settings (Per CLAUDE.md Guidelines)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `enable_contraction` | **false** | Contraction OFF for this experiment |
| `disable_value_head_norm` | **true** | Value-head spectral norm OFF (prevents collapse) |
| `episodic_latent` | **false** | Persistent-z mode (latent carried across steps) |
| `latent_ball_radius` | 10.0 | Forward-invariant projection radius |

### Reward Shaping

| Parameter | Value |
|-----------|-------|
| `reward_shaping` | true |
| `solve_terminal_reward` | 1.0 |
| `fail_terminal_reward` | -81.0 |
| `C_max` | 81.0 |

### Checker & Constraints

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_feasibility_checker` | **true** | Primary checker (per CLAUDE.md) |
| `feasibility_violation_weight` | 2.0 | Penalty weight for constraint violations |
| `feasibility_zerocand_weight` | 5.0 | Penalty weight for zero-candidate cells |

### Evaluation Settings

| Parameter | Value |
|-----------|-------|
| `eval_interval` | 500 steps |
| `eval_num_episodes` | 50 |
| `log_interval` | 100 steps |

---

## 2. Dataset

**Path:** `data/sudoku-9x9/`

| Split | Size | Usage |
|-------|------|-------|
| Train | 10,000 puzzles | RL training + in-training evaluation |
| Validation | 1,000 puzzles | Not used in this experiment |
| Test | 1,000 puzzles | Not used in this experiment |

**Difficulty Distribution:**
- Easy: 30%
- Medium: 40%
- Hard: 30%

**Grid Size:** 9x9 (81 cells)

**Note:** Both training rollouts and periodic evaluation sample from the train split. The validation and test sets exist for future held-out evaluation but were not used during RL training.

---

## 3. Model Architecture

| Component | Value |
|-----------|-------|
| `model_type` | `trm` (Thinking Recursive Model) |
| `latent_ball_radius` | 10.0 |
| Action Masking | Enabled (prevents editing given cells) |

---

## 4. Training Progress

### Key Milestones

| Step | Event |
|------|-------|
| 0 | Training started |
| 16,500 | **First puzzle solved** (1/50 = 2%) |
| 20,500 | First 4% success rate (2/50) |
| 23,000 | First 6% success rate (3/50) |
| 46,000 | **Peak success rate: 12%** (6/50) |
| 47,500 | **Peak mean score: 57.02** |
| 50,000 | Training completed |

### Evaluation Results (Every 500 Steps)

#### Early Training (Steps 500-15,000)
- Success Rate: 0%
- Mean Score: 31-36 (starting from initial ~27)
- Model learning to fill cells but not solving puzzles

#### Mid Training (Steps 15,000-30,000)
- First solves appear at step 16,500
- Success Rate: 0-6%
- Mean Score: 32-45
- Rapid improvement in both metrics

#### Late Training (Steps 30,000-50,000)
- Success Rate: 6-12%
- Mean Score: 50-57
- More stable performance with occasional dips

### Detailed Results Table (Selected Checkpoints)

| Step | Success Rate | Mean Score | Solved/Total | Score Range | Initial Score |
|------|-------------|------------|--------------|-------------|---------------|
| 500 | 0.0% | 31.58 | 0/50 | 22-42 | 26.84 |
| 5,000 | 0.0% | 35.90 | 0/50 | 22-58 | 26.84 |
| 10,000 | 0.0% | 31.54 | 0/50 | 18-69 | 26.84 |
| 15,000 | 0.0% | 31.84 | 0/50 | 18-65 | 26.84 |
| 16,500 | **2.0%** | 33.12 | **1/50** | 18-81 | 26.84 |
| 20,000 | 2.0% | 41.20 | 1/50 | 18-81 | 26.84 |
| 25,000 | 4.0% | 43.12 | 2/50 | 18-81 | 26.84 |
| 30,000 | 6.0% | 47.68 | 3/50 | 18-81 | 26.84 |
| 35,000 | 8.0% | 50.06 | 4/50 | 18-81 | 26.84 |
| 40,000 | 10.0% | 52.38 | 5/50 | 18-81 | 26.84 |
| 45,000 | 8.0% | 53.06 | 4/50 | 19-81 | 26.84 |
| **46,000** | **12.0%** | 55.50 | **6/50** | 19-81 | 26.84 |
| **47,500** | 10.0% | **57.02** | 5/50 | 27-81 | 26.84 |
| 50,000 | 8.0% | 54.08 | 4/50 | 24-81 | 26.84 |

---

## 5. Final Results

### Summary Statistics

| Metric | Initial | Final (50k) | Peak | Peak Step |
|--------|---------|-------------|------|-----------|
| Success Rate | 0% | 8% | **12%** | 46,000 |
| Mean Score | 26.84 | 54.08 | **57.02** | 47,500 |
| Score Improvement | - | +27.24 | +30.18 | - |

### Performance Breakdown

- **Puzzles Solved:** 4/50 at final checkpoint (8%)
- **Best Performance:** 6/50 puzzles solved (12%) at step 46,000
- **Score Range at Final:** 24-81 out of 81
- **Mean Score Gain:** 54.08 - 26.84 = **+27.24 points** (doubled from initial)

---

## 6. Key Observations

### What Worked

1. **Persistent Latent Mode (`episodic_latent: false`):**
   - Latent state carried across edit steps within an episode
   - Allows model to accumulate information about the puzzle

2. **Feasibility Checker:**
   - Scores based on filled cells minus weighted violations
   - Provides dense reward signal for learning

3. **Value-Head Norm Disabled:**
   - Prevented target saturation issues
   - Stable training throughout 50k steps

4. **Action Masking:**
   - Prevented invalid edits to given cells
   - Used during both training and evaluation

### Challenges Observed

1. **High Variance in Success Rate:**
   - Success rate fluctuates between 0-12%
   - Even late in training, some eval windows show 0% success
   - Likely due to small eval sample size (50 episodes)

2. **Mean Score Plateau:**
   - Score improvement slows after step 40k
   - May need longer training or hyperparameter tuning

3. **No Solves Until Step 16.5k:**
   - ~33% of training before first solve
   - 9x9 Sudoku is significantly harder than 4x4

### Comparison to Initial State

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Filled Cells (mean) | ~27 | ~54-56 | +27-29 cells |
| Violations (mean) | - | ~0 | Clean solutions |
| Solve Rate | 0% | 8-12% | Learned to solve |

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `upi_trm_50k_s0.log` | Full training log |
| `upi_trm_50k_training_progress.png` | Training curves plot |
| `EXPERIMENT_REPORT.md` | This report |

**Checkpoint Location:**
`checkpoints/rl_sudoku-9x9_seed0/rl_checkpoint_step_50000.pt`

---

## 8. Configuration File

**Path:** `configs/sudoku9x9/upi_trm_9x9_50k.yaml`

```yaml
algorithm: "upi_trm"
gamma: 0.99
K: 1
inner_unroll_n: 1
max_edits: 81
num_train_steps: 50000
batch_size: 64
rollout_episodes_per_step: 2
value_lr: 0.0003
policy_lr: 0.0001
value_grad_clip: 1.0
policy_grad_clip: 0.5
mixture_alpha: 0.1
policy_epsilon: 0.1
target_ema_tau: 0.99
entropy_coef: 0.05
enable_contraction: false
target_Lz: 0.9
disable_value_head_norm: true
value_target_clip: 100.0
advantage_clip: 20.0
batch_centered_advantage: true
log_interval: 100
eval_interval: 500
eval_num_episodes: 50
episodic_latent: false
use_feasibility_checker: true
feasibility_violation_weight: 2.0
feasibility_zerocand_weight: 5.0
reward_shaping: true
solve_terminal_reward: 1.0
fail_terminal_reward: -81.0
stop_action_mode: disabled
C_max: 81.0
dataset_dir: "data/sudoku-9x9"
task_name: "sudoku"
model_type: "trm"
latent_ball_radius: 10.0
track_theory_metrics: true
```

---

## 9. Next Steps (Recommendations)

1. **Run Additional Seeds:** Run seeds 1, 2 for statistical significance
2. **Baseline Comparisons:** Run PPO and DQN baselines with same config
3. **Longer Training:** Consider 100k steps to see if performance continues improving
4. **Hyperparameter Tuning:**
   - Try different `entropy_coef` values
   - Experiment with `mixture_alpha`
5. **Ablation Studies:**
   - Compare episodic vs persistent latent
   - Test with contraction enabled (`enable_contraction: true`)

---

*Report generated automatically from training logs.*
