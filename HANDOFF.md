# HANDOFF.md - Session Summary for UPI-TRM Project

**Date:** 2026-01-20
**Branch:** `feature/upi-trm-clean`
**Latest Commit:** See git log for current state

---

## Current Work

### Active Task: Hard 4×4 Sudoku Baseline Experiments

**Status:** IN PROGRESS (Batch 2/5 completing)
**Started:** 2026-01-20 ~08:37
**Estimated Completion:** ~4-5 more hours for remaining batches

#### What's Running

18 experiments (6 methods × 3 seeds) on harder 4×4 Sudoku (6-8 empty cells) with 20k training steps. Running on 4 A100 GPUs in 5 sequential batches.

**Script:** `/home/buiksat/trm_bellman/scripts/run_table3_hard.sh`
**Results:** `/home/buiksat/trm_bellman/results/table3_hard_6to8/`

#### Current Progress (as of last check)

| Method | Seed 42 | Seed 123 | Seed 456 | Status |
|--------|---------|----------|----------|--------|
| persistent_nc | 60.0% | 52.0% | 58.0% | ✅ COMPLETE |
| episodic_nc | 42.0% | 54.0% | 48.0% | ✅ COMPLETE |
| episodic_c_clean | ~36% | ~38% | pending | 🔄 Batch 2 |
| ppo | - | - | - | ⏳ Batch 3 |
| a2c | - | - | - | ⏳ Batch 4 |
| dqn | - | - | - | ⏳ Batch 4-5 |

#### Key Observation

Contraction variant (`episodic_c_clean`) shows lower success on harder puzzles:
- **Trivial (1-4 empties):** 90.7% ± 5.0%
- **Hard (6-8 empties):** ~36-38% (preliminary)

No-contraction variants maintain higher success on hard puzzles (~50-57%).

### Decisions Made This Session

1. **Table 3 Data Verification:** Discovered paper had incorrect baseline numbers. Re-ran all 18 experiments (6 methods × 3 seeds) and updated `main.tex`.

2. **Missing persistent_nc Experiments:** Found that `persistent_nc` logs didn't exist - ran them and confirmed results match `episodic_nc` (93.3% ± 2.3% both).

3. **Config Asymmetry Documented:** Noted that:
   - No-contraction configs have **projection ON** (R=10, default)
   - Clean contraction config has **projection OFF** (R=0)
   - This is intentional for matching historical Table 3 setup

4. **Paper Updates Made:**
   - Table 3 (lines 1583-1588): Correct experimental values
   - Abstract (line 151): "90–93%" success range
   - Key observation (lines 1594-1601): Clean contraction note added
   - Appendix (line 2094): Baseline comparison updated

5. **Figure Regenerated:** `trivial_baselines_vs_no_contraction_success_vs_steps.pdf` with paper style (legend below, individual seed curves, thick mean lines).

### Relevant Codebase Details

#### Config Locations
```
configs/
├── ablations/
│   ├── upi_trm_feasibility_no_contraction.yaml        # episodic_nc
│   └── upi_trm_feasibility_persistent_z_no_contraction.yaml  # persistent_nc
├── baselines/
│   ├── ppo_trm_feasibility.yaml
│   ├── a2c_trm_feasibility.yaml
│   └── dqn_trm_feasibility.yaml
└── exp3_projection_ablation/
    └── c_rdis.yaml                                     # episodic_c_clean (contraction ON, vhead OFF, R=0)
```

#### Key Config Differences

| Config | `enable_contraction` | `latent_ball_radius` | `disable_value_head_norm` | `episodic_latent` |
|--------|---------------------|---------------------|--------------------------|-------------------|
| episodic_nc | false | 10.0 (default) | not set | true |
| persistent_nc | false | 10.0 (default) | not set | false |
| episodic_c_clean | true | 0 (disabled) | true | true |

#### Dataset Locations
```
data/
├── sudoku-4x4-trivial           # 1-4 empty cells (Table 3 primary)
├── sudoku-4x4-easy_6to8empties  # 6-8 empty cells (harder experiments)
└── sudoku-4x4-ultra-easy        # Mostly easy puzzles
```

#### Scripts Created This Session
```
scripts/
├── run_table3_baselines.sh      # Trivial 4×4, 5k steps
├── run_table3_hard.sh           # Hard 4×4, 20k steps (RUNNING)
├── run_persistent_nc.sh         # Fixed missing persistent_nc runs
└── plot_table3_baselines.py     # Paper-style learning curves
```

### Next Steps (When Experiments Complete)

1. **Aggregate hard 4×4 results:** Compute mean±std for all 6 methods
2. **Generate learning curves figure:** Modify `plot_table3_baselines.py` for hard dataset
3. **Update paper (optional):** Add hard 4×4 results to appendix if significant
4. **Consider 9×9 experiments:** Instructions in HANDOFF.md for generating dataset and running

### Monitoring Commands

```bash
# Overall progress
tail -20 /home/buiksat/trm_bellman/results/table3_hard_6to8/run_all.log

# Individual experiment status
for f in /home/buiksat/trm_bellman/results/table3_hard_6to8/*.log; do
    if [[ "$f" != *"run_all"* ]]; then
        echo "=== $(basename $f) ===";
        grep "eval_success" "$f" | tail -1;
    fi
done
```

---

## Project Overview

**UPI-TRM (Unified Policy Iteration with Thinking Recursive Model)** is a research project exploring contraction-based stability for recursive latent reasoning in neural networks. The work targets ICML submission.

### Key Hypothesis
Spectral-norm contraction on the z→z update map should provide stable value function learning with bounded error propagation.

---

## Current State: Table 3 Experiments (2026-01-20)

### Active Experiments

**Harder 4×4 Sudoku (6-8 empties, 20k steps):**
- **Status:** RUNNING on 4 A100 GPUs
- **Location:** `results/table3_hard_6to8/`
- **Script:** `scripts/run_table3_hard.sh`
- **Design:** 6 methods × 3 seeds = 18 experiments in 5 batches

**Methods being tested:**
| Method | Config | Key Settings |
|--------|--------|--------------|
| persistent_nc | `ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml` | contraction OFF, projection ON (R=10) |
| episodic_nc | `ablations/upi_trm_feasibility_no_contraction.yaml` | contraction OFF, projection ON (R=10) |
| episodic_c_clean | `exp3_projection_ablation/c_rdis.yaml` | contraction ON, vhead OFF, projection OFF |
| ppo | `baselines/ppo_trm_feasibility.yaml` | Standard PPO |
| a2c | `baselines/a2c_trm_feasibility.yaml` | Standard A2C |
| dqn | `baselines/dqn_trm_feasibility.yaml` | Standard DQN |

### Completed Table 3 Experiments (Trivial 4×4 Sudoku, 5k steps)

**Results saved to:** `results/table3_baselines/`

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| persistent_nc | 96.0% | 92.0% | 92.0% | **93.3% ± 2.3%** |
| episodic_nc | 92.0% | 96.0% | 92.0% | **93.3% ± 2.3%** |
| episodic_c_clean | 96.0% | 86.0% | 90.0% | **90.7% ± 5.0%** |
| PPO | 30.0% | 30.0% | 30.0% | 30.0% ± 0.0% |
| A2C | 30.0% | 34.0% | 30.0% | 31.3% ± 2.3% |
| DQN | 24.0% | 30.0% | 30.0% | 28.0% ± 3.5% |
| Random baseline | - | - | - | 52.0% |

### ⚠️ Important Config Notes

**Projection status in no-contraction configs:**
- `episodic_nc` and `persistent_nc` have **projection ON** (default `latent_ball_radius: 10.0`)
- `episodic_c_clean` has **projection OFF** (`latent_ball_radius: 0`)

This is an intentional asymmetry in Table 3. If you need consistent projection settings:
- Use `exp3_projection_ablation/nc_rdis.yaml` for no-contraction with projection OFF
- Use `exp3_projection_ablation/c_r10.yaml` for contraction with projection ON

---

## Paper Assets Updated (2026-01-20)

**Table 3 in paper (`main.tex`):**
- Lines 1583-1588: Updated with correct experimental values
- Abstract (line 151): "90–93%" success range
- Key observation (lines 1594-1601): Updated with clean contraction note
- Appendix (line 2094): Updated baseline comparison

**Figure regenerated:**
- `figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf` - Learning curves with paper style

---

## Running Experiments on 9×9 Sudoku

### Step 1: Generate 9×9 Dataset

```bash
cd ~/fbsource/fbcode/buiksat_trm

# Generate 9×9 Sudoku dataset (easy - fewer empty cells)
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-9x9-easy \
    --subsample-size 1000 \
    --num-aug 100 \
    --min-empty 10 \
    --max-empty 30

# For harder 9×9 (more empty cells):
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-9x9-medium \
    --subsample-size 1000 \
    --num-aug 100 \
    --min-empty 30 \
    --max-empty 50
```

**Note:** 9×9 Sudoku is significantly harder. Expect:
- Much longer training (50k-100k+ steps)
- Lower success rates initially
- Potentially need architecture changes (more unroll steps, larger latent dim)

### Step 2: Create Experiment Script

Adapt `scripts/run_table3_hard.sh` for 9×9:

```bash
#!/bin/bash
cd ~/fbsource/fbcode

DATA_PATH="buiksat_trm/data/sudoku-9x9-easy"
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_9x9_easy"
TRAIN_STEPS=50000  # 9x9 needs more training

mkdir -p "$RESULTS_DIR"

# Configs remain the same - they work for any Sudoku size
declare -A CONFIGS
CONFIGS["persistent_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
CONFIGS["episodic_nc"]="buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml"
CONFIGS["episodic_c_clean"]="buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml"
CONFIGS["ppo"]="buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml"
CONFIGS["a2c"]="buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml"
CONFIGS["dqn"]="buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml"

SEEDS=(42 123 456)

run_experiment() {
    local gpu=$1
    local method=$2
    local seed=$3
    local config="${CONFIGS[$method]}"
    local logfile="$RESULTS_DIR/${method}_s${seed}.log"

    echo "[GPU $gpu] Starting $method seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu buck2 run //buiksat_trm:upi_trm_train \
        -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true \
        -- --config "$config" \
        --seed "$seed" \
        --dataset-paths "$DATA_PATH" \
        --train-steps "$TRAIN_STEPS" \
        --no-wandb \
        > "$logfile" 2>&1
}

# Run experiments in batches of 4
# ... (same batch structure as run_table3_hard.sh)
```

### Step 3: Run and Monitor

```bash
chmod +x scripts/run_table3_9x9.sh
nohup ./scripts/run_table3_9x9.sh > results/table3_9x9_easy/run_all.log 2>&1 &

# Monitor progress:
tail -f results/table3_9x9_easy/run_all.log
```

### Step 4: Generate Figure

After experiments complete, update `scripts/plot_table3_baselines.py`:
- Change `results_dir` to `Path("/home/buiksat/trm_bellman/results/table3_9x9_easy")`
- Change output filename to include "9x9"
- Adjust x-axis limit to match training steps

```bash
buck2 run //buiksat_trm:plot_table3_baselines
```

---

## Monitoring Current Hard 4×4 Experiments

```bash
# Check overall progress:
tail -20 /home/buiksat/trm_bellman/results/table3_hard_6to8/run_all.log

# Check specific experiment progress:
for f in /home/buiksat/trm_bellman/results/table3_hard_6to8/*.log; do
    if [[ "$f" != *"run_all"* ]]; then
        echo "=== $(basename $f) ===";
        grep "eval_success" "$f" | tail -1;
    fi
done

# Estimated completion: ~4-5 hours per batch, 5 batches total (~20-25 hours)
```

---

## Previous Experiment Status (All Complete)

### Phase 4: 2×2 Norm Ablation (COMPLETE ✅)
- **Location:** `results/paper_ready/phase4_2x2_norm_ablation/`
- **Finding:** z→z contraction is the primary stabilizer (~97% vs ~78% argmax agreement)

### Exp1-5: Contraction & Stability Experiments (COMPLETE ✅)
- See `EXPERIMENT_PLAN_ICML.md` for full details
- All audits passed

---

## Key Technical Findings

### 1. Table 3 Results Match Paper
All baseline comparisons now have verified experimental data across 3 seeds.

### 2. Config Asymmetry (Document This)
- No-contraction configs use **projection ON** (R=10, default)
- Clean contraction config uses **projection OFF** (R=0)
- This is intentional for Table 3 but worth noting

### 3. Random Baseline Differences by Dataset
- Trivial 4×4 (1-4 empties): 52% random success
- Hard 4×4 (6-8 empties): ~0% random success
- 9×9: Near 0% random success

---

## Important Configuration Rules

### From CLAUDE.md (Must Follow)
1. **Checker:** Use `use_feasibility_checker: true` only
2. **Value-head normalization:** `disable_value_head_norm: true` for stability experiments
3. **One variable at a time:** Don't mix episodic_latent with contraction changes

### Config Template for Stability Experiments
```yaml
use_feasibility_checker: true
enable_contraction: true
target_Lz: 0.9
disable_value_head_norm: true   # CRITICAL
episodic_latent: true
latent_ball_radius: 10.0        # or 0 for projection-free
```

---

## Commands Reference

### Run Table 3 Baselines (Trivial)
```bash
cd ~/fbsource/fbcode
nohup /home/buiksat/trm_bellman/scripts/run_table3_baselines.sh &
```

### Run Table 3 Hard (6-8 empties)
```bash
cd ~/fbsource/fbcode
nohup /home/buiksat/trm_bellman/scripts/run_table3_hard.sh &
```

### Generate Learning Curves Figure
```bash
buck2 run //buiksat_trm:plot_table3_baselines \
  -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true
```

### Aggregate Results
```bash
for method in persistent_nc episodic_nc episodic_c_clean ppo a2c dqn; do
    echo "$method:"
    for seed in 42 123 456; do
        logfile="results/table3_baselines/${method}_s${seed}.log"
        if [ -f "$logfile" ]; then
            final_success=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            echo "  seed $seed: $final_success"
        fi
    done
done
```

---

## Files to Read First

1. `CLAUDE.md` - Operational guidance, non-negotiables
2. `EXPERIMENT_PLAN_ICML.md` - Full experiment plan with status
3. `results/table3_baselines/` - Latest baseline experiment logs
4. `scripts/run_table3_hard.sh` - Currently running experiment

---

## Session Log

| Date | Description |
|------|-------------|
| 2026-01-20 | Table 3 baseline experiments, paper figure regeneration, hard dataset experiments |
| 2026-01-19 | Phase 4 2×2 Norm Ablation, Exp5 tradeoff curve |
| 2026-01-18 | Exp3-4 projection ablation experiments |

---

*Generated: 2026-01-20*
