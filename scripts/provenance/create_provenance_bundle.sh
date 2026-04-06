#!/bin/bash
# Create provenance bundle for Figure 2 and Table 3 rerun
# Contains all artifacts needed to reproduce paper results

set -e

DATE=$(date +%Y_%m_%d)
RESULTS_DIR="/home/buiksat/trm_bellman/results/table3_baselines_rerun_evalfix_2026_01_22"
PROVENANCE_DIR="/home/buiksat/trm_bellman/results/paper_ready/fig2_provenance_rerun_evalfix_${DATE}"

echo "============================================================"
echo "Creating Provenance Bundle for Figure 2 + Table 3 Rerun"
echo "============================================================"
echo "Source: $RESULTS_DIR"
echo "Target: $PROVENANCE_DIR"
echo ""

# Create directory structure
mkdir -p "$PROVENANCE_DIR"/{logs,configs,figures,scripts}

# 1. Copy training logs
echo "=== Copying training logs ==="
cp "$RESULTS_DIR"/*.log "$PROVENANCE_DIR/logs/" 2>/dev/null || echo "  No logs found"
echo "  Copied $(ls "$PROVENANCE_DIR/logs/"*.log 2>/dev/null | wc -l) log files"

# 2. Copy configs used
echo ""
echo "=== Copying configs ==="
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/ablations/upi_trm_feasibility_no_contraction.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/exp3_projection_ablation/c_rdis.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/baselines/ppo_trm_feasibility.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/baselines/a2c_trm_feasibility.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
cp /home/buiksat/fbsource/fbcode/buiksat_trm/configs/baselines/dqn_trm_feasibility.yaml "$PROVENANCE_DIR/configs/" 2>/dev/null || true
echo "  Copied $(ls "$PROVENANCE_DIR/configs/"*.yaml 2>/dev/null | wc -l) config files"

# 3. Copy figures
echo ""
echo "=== Copying figures ==="
cp /home/buiksat/trm_bellman/figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf "$PROVENANCE_DIR/figures/" 2>/dev/null || echo "  PDF not found"
cp /home/buiksat/trm_bellman/figures/trivial_baselines_vs_no_contraction_success_vs_steps.png "$PROVENANCE_DIR/figures/" 2>/dev/null || echo "  PNG not found"
echo "  Copied $(ls "$PROVENANCE_DIR/figures/"* 2>/dev/null | wc -l) figure files"

# 4. Copy scripts
echo ""
echo "=== Copying scripts ==="
cp /home/buiksat/trm_bellman/scripts/run_table3_fig2_rerun.sh "$PROVENANCE_DIR/scripts/"
cp /home/buiksat/trm_bellman/scripts/plot_table3_baselines.py "$PROVENANCE_DIR/scripts/"
cp /home/buiksat/trm_bellman/scripts/generate_table3_summary.py "$PROVENANCE_DIR/scripts/"
echo "  Copied $(ls "$PROVENANCE_DIR/scripts/"* 2>/dev/null | wc -l) script files"

# 5. Copy table summary
echo ""
echo "=== Copying table summary ==="
cp "$RESULTS_DIR/table3_summary.md" "$PROVENANCE_DIR/" 2>/dev/null || echo "  Table summary not found (generate with generate_table3_summary.py)"

# 6. Create README
echo ""
echo "=== Creating README ==="
cat > "$PROVENANCE_DIR/README.md" << 'EOFREADME'
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
EOFREADME

echo ""
echo "============================================================"
echo "Provenance bundle created: $PROVENANCE_DIR"
echo "============================================================"
ls -la "$PROVENANCE_DIR/"
