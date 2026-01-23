# Figure 2 + Table 3 Provenance Bundle (Post-Evaluator-Fix Rerun)

**Generated:** $(date)
**Purpose:** Reproducibility artifacts for ICML submission

## Contents

- `logs/` - Training logs for all 18 runs (6 methods × 3 seeds)
- `configs/` - YAML configs used for each method
- `figures/` - Generated Figure 2 PDF and PNG
- `scripts/` - Scripts used to run experiments and generate outputs
- `table3_summary.md` - Table 3 summary with final success rates

## Methods

| Key | Config | Description |
|-----|--------|-------------|
| persistent_nc | upi_trm_feasibility_persistent_z_no_contraction.yaml | UPI-TRM Persistent-z, no contraction |
| episodic_nc | upi_trm_feasibility_no_contraction.yaml | UPI-TRM Episodic-z, no contraction |
| episodic_c_clean | c_rdis.yaml | UPI-TRM Episodic-z with contraction |
| ppo | ppo_trm_feasibility.yaml | PPO baseline |
| a2c | a2c_trm_feasibility.yaml | A2C baseline |
| dqn | dqn_trm_feasibility.yaml | DQN baseline |

## Seeds

42, 123, 456

## Reproduction Commands

```bash
# Run training (from fbcode directory)
bash scripts/run_table3_fig2_rerun.sh

# Generate Figure 2
python scripts/plot_table3_baselines.py \
    --results-dir logs/ \
    --output-dir figures/

# Generate Table 3 summary
python scripts/generate_table3_summary.py \
    --results-dir logs/
```
