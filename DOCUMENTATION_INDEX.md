# Documentation Index: Shaped Reward Experiments

This index helps you quickly find the right documentation for your ICML 2026 submission.

## Quick Navigation

### 🚀 **Want to run experiments NOW?**
→ Read `docs/QUICKSTART_SHAPED_REWARDS.md`

### 📖 **Want to understand the theory-to-code mapping?**
→ Read `docs/shaped_rewards_theory.md`

### 📝 **Want full implementation details?**
→ Read `docs/shaped_rewards_implementation_complete.md`

### ✅ **Want a summary of what was implemented?**
→ Read `SHAPED_REWARDS_COMPLETE.md` (this directory)

### 🔬 **Want to run experiments?**
→ Execute `./scripts/run_shaped_reward_experiments.sh --help`

### 🐛 **Want to verify configs are valid?**
→ Execute `python scripts/verify_configs.py`

## Documentation Files

### Main Documentation

1. **`SHAPED_REWARDS_COMPLETE.md`** (you are here)
   - Executive summary
   - What was implemented
   - How to run experiments
   - Expected results
   - Files created

### Detailed Guides

2. **`docs/shaped_rewards_theory.md`**
   - Paper specification (Eq. 4, Remark 2.6)
   - Code implementation details
   - C_max = 10.0 for Sudoku
   - Theory alignment checklist
   - How to enable theory-exact mode
   - STOP action interaction
   - Verification tests

3. **`docs/shaped_rewards_implementation_complete.md`**
   - Comprehensive implementation summary
   - How configs map to paper features
   - Experimental validation plan
   - Metrics to track
   - Next steps
   - Full file list

4. **`docs/QUICKSTART_SHAPED_REWARDS.md`**
   - Quick start commands
   - Single experiment examples
   - Full suite runner
   - Expected results table
   - Validation checklist

### Other Documentation

5. **`docs/BASELINE_IMPLEMENTATION.md`** (NEW - Dec 27, 2024)
   - PPO/A2C baseline trainers
   - NoRecursionEncoder architecture
   - CLI flags: `--baseline`, `--backbone`
   - Experimental comparison matrix
   - Usage examples for running baselines

6. **`docs/THEORY_NOTES.md`** (NEW - Dec 27, 2024)
   - CPI mixture policy analysis
   - Why IS weights are NOT required in theory-exact mode
   - Three CPI modes comparison table
   - Episodic vs persistent latent z
   - Exact baseline summation (Theorem 5.9)
   - Key config flags for theory alignment

7. **`README.md`** (main repository README)
   - Full UPI-TRM documentation
   - Installation instructions
   - All training modes (not just shaped rewards)

7. **`docs/rl_upi_trm.md`**
   - Paper section → code file mapping
   - Implementation guide

8. **`CLAUDE.md`**
   - Guide for future Claude Code sessions
   - Essential commands
   - Architecture overview

## Config Files

### Main Configs (2 files)

Located in `configs/`:

1. **`rl_sudoku_shaped_theory_exact.yaml`**
   - All theory features enabled
   - Shaped rewards + rush-to-fail mitigation
   - Main contribution config
   - Expected: Best performance

2. **`rl_sudoku_sparse_theory_exact.yaml`**
   - Sparse (terminal-only) rewards
   - All theory features except shaped rewards
   - Baseline for comparison
   - Expected: Slower learning

### Ablation Configs (7 files)

Located in `configs/ablations/`:

1. **`ablation_no_contraction.yaml`**
   - Tests Assumption 4.2
   - `enable_contraction: false`

2. **`ablation_no_exact_baseline.yaml`** ⭐ KEY
   - Tests Theorem 5.9 (main contribution)
   - `exact_baseline_summation: false`

3. **`ablation_no_conservative_mixture.yaml`**
   - Tests CPI benefit
   - `mixture_alpha: 1.0` (greedy)

4. **`ablation_no_projection.yaml`**
   - Tests Assumption 4.1
   - `latent_ball_radius: 0.0`

5. **`ablation_no_theory_exact_mixture.yaml`**
   - Tests policy-space vs parameter-space
   - `theory_exact_mixture: false`

6. **`ablation_no_theory_features.yaml`**
   - All features OFF
   - Baseline RL

7. **`ablation_sparse_no_theory.yaml`**
   - Hardest baseline
   - Sparse + no theory features

## Scripts

### Experiment Runners

1. **`scripts/run_shaped_reward_experiments.sh`**
   - Runs full suite (8 configs × N seeds)
   - Supports parallel execution
   - Logs to W&B with organized groups
   - Usage: `./scripts/run_shaped_reward_experiments.sh --help`

2. **`scripts/generate_ablation_configs.py`**
   - Auto-generates all ablation configs
   - Already run - configs exist
   - Rerun if you need to regenerate

3. **`scripts/verify_configs.py`**
   - Validates all configs
   - Checks theory alignment
   - Run before experiments

## How to Use This Documentation

### Scenario 1: "I want to run experiments now"

1. Read `docs/QUICKSTART_SHAPED_REWARDS.md` (5 min)
2. Run `python scripts/verify_configs.py` (verify setup)
3. Run `./scripts/run_shaped_reward_experiments.sh --seeds 1 --steps 5000` (pilot)
4. Monitor in W&B
5. If successful, run full suite with `--seeds 3 --steps 20000`

### Scenario 2: "I need to understand the theory alignment"

1. Read `docs/shaped_rewards_theory.md` (detailed theory mapping)
2. Check `rl/envs/plan_edit_env.py:compute_transition_reward()` (implementation)
3. Review config `configs/rl_sudoku_shaped_theory_exact.yaml` (settings)
4. Run `python scripts/verify_configs.py` (verification)

### Scenario 3: "I need to write the paper experiments section"

1. Read `SHAPED_REWARDS_COMPLETE.md` (this file) for expected results
2. Run experiments: `./scripts/run_shaped_reward_experiments.sh`
3. Collect results from W&B
4. Refer to `docs/shaped_rewards_implementation_complete.md` for:
   - Metrics to report
   - Figures to create
   - Claims to make

### Scenario 4: "Reviewers asked about implementation details"

1. Point them to `docs/shaped_rewards_theory.md` (theory-to-code mapping)
2. Reference specific line numbers:
   - Checker: `upi_trm_train.py:212-228`
   - Reward computation: `rl/envs/plan_edit_env.py:375-443`
   - Config validation: `rl/config.py:208-301`
3. Share config: `configs/rl_sudoku_shaped_theory_exact.yaml`

### Scenario 5: "I want to add a new ablation"

1. Edit `scripts/generate_ablation_configs.py`
2. Add new `create_ablation()` call
3. Run `python scripts/generate_ablation_configs.py`
4. Verify with `python scripts/verify_configs.py`
5. Add to `scripts/run_shaped_reward_experiments.sh` EXPERIMENTS array

## Quick Reference

### Commands

```bash
# Verify setup
python scripts/verify_configs.py

# Quick test (1 seed, fast)
python upi_trm_train.py --config configs/rl_sudoku_shaped_theory_exact.yaml --seed 42

# Pilot run (1 seed, all configs)
./scripts/run_shaped_reward_experiments.sh --seeds 1 --steps 5000 --dry-run
./scripts/run_shaped_reward_experiments.sh --seeds 1 --steps 5000

# Full run (3 seeds, publication quality)
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 20000

# Parallel (faster)
./scripts/run_shaped_reward_experiments.sh --seeds 3 --steps 20000 --parallel
```

### Key Files to Know

| File | Purpose | When to Use |
|------|---------|-------------|
| `configs/rl_sudoku_shaped_theory_exact.yaml` | Main config (all features ON) | Always (baseline) |
| `configs/ablations/ablation_no_exact_baseline.yaml` | Tests Theorem 5.9 | Critical ablation |
| `scripts/run_shaped_reward_experiments.sh` | Run all experiments | Publication runs |
| `scripts/verify_configs.py` | Validate configs | Before experiments |
| `docs/shaped_rewards_theory.md` | Theory details | Understanding |
| `docs/QUICKSTART_SHAPED_REWARDS.md` | Quick start | Getting started |

## Paper Integration

When writing Section 6 (Experiments) in `main.tex`, reference:

1. **Shaped vs Sparse:**
   - Configs: `rl_sudoku_shaped_theory_exact.yaml` vs `rl_sudoku_sparse_theory_exact.yaml`
   - Expected: Shaped learns 2× faster
   - Claim: "Potential-based shaping (Eq. 4) accelerates learning by..."

2. **Ablation Study:**
   - Configs: All `configs/ablations/ablation_*.yaml`
   - Critical: `ablation_no_exact_baseline.yaml` (tests Theorem 5.9 - KEY)
   - Claim: "Exact baseline summation provides X% improvement, confirming the O(α·ε_A) bound..."

3. **Theory Validation:**
   - Metrics: `theory/hat_Lz`, `theory/unrolling_term`, `theory/bellman_residual`
   - Claim: "Empirical measurements confirm theoretical predictions: L_z stays < 1, unrolling bias decays geometrically..."

## FAQ

**Q: Do I need to modify any code?**
A: No! Everything is config-based. Just run experiments with different YAML files.

**Q: Which config should I run first?**
A: `configs/rl_sudoku_shaped_theory_exact.yaml` (main contribution)

**Q: Which ablation is most important?**
A: `configs/ablations/ablation_no_exact_baseline.yaml` (tests Theorem 5.9 - your KEY contribution)

**Q: How long do experiments take?**
A: ~30 min per run (5K steps), ~2-4 hours per run (20K steps), on a single GPU

**Q: How many seeds should I use?**
A: 3 seeds for publication (standard), 1 seed for quick testing

**Q: Where are results logged?**
A: W&B project `UPI-TRM-ICML-Shaped-Rewards`, organized by config groups

**Q: What if a config fails validation?**
A: Run `python scripts/verify_configs.py` to see specific issues. Most "failures" are intentional warnings for ablations.

**Q: Can I run on CPU?**
A: Yes, but much slower. Reduce `--batch-size` to 64 or 32.

---

**Everything is ready. Start with the quickstart guide and run your first experiment!** 🚀

See: `docs/QUICKSTART_SHAPED_REWARDS.md`
