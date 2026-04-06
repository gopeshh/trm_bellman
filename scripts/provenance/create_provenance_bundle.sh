#!/usr/bin/env bash
# Create provenance bundle for Figure 2 and Table 3 rerun
# Contains all artifacts needed to reproduce paper results

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATE="$(date +%Y_%m_%d)"
GENERATED_AT="$(date -Is)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/table3_baselines_rerun_evalfix_2026_01_22}"
PROVENANCE_DIR="${PROVENANCE_DIR:-${REPO_ROOT}/results/paper_ready/fig2_provenance_rerun_evalfix_${DATE}}"
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configs}"
FIGURE_ROOT="${FIGURE_ROOT:-${REPO_ROOT}/figures}"
SCRIPT_ROOT="${SCRIPT_ROOT:-${REPO_ROOT}/scripts}"

echo "============================================================"
echo "Creating Provenance Bundle for Figure 2 + Table 3 Rerun"
echo "============================================================"
echo "Source: $RESULTS_DIR"
echo "Target: $PROVENANCE_DIR"
echo ""

# Create directory structure
mkdir -p "$PROVENANCE_DIR"/{logs,configs,figures,scripts}

config_files=(
  "ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml"
  "ablations/upi_trm_feasibility_no_contraction.yaml"
  "exp3_projection_ablation/c_rdis.yaml"
  "baselines/ppo_trm_feasibility.yaml"
  "baselines/a2c_trm_feasibility.yaml"
  "baselines/dqn_trm_feasibility.yaml"
)

figure_files=(
  "trivial_baselines_vs_no_contraction_success_vs_steps.pdf"
  "trivial_baselines_vs_no_contraction_success_vs_steps.png"
)

script_files=(
  "run_table3_fig2_rerun.sh"
  "plot_table3_baselines.py"
  "generate_table3_summary.py"
)

# 1. Copy training logs
echo "=== Copying training logs ==="
cp "$RESULTS_DIR"/*.log "$PROVENANCE_DIR/logs/" 2>/dev/null || echo "  No logs found"
echo "  Copied $(ls "$PROVENANCE_DIR/logs/"*.log 2>/dev/null | wc -l) log files"

# 2. Copy configs used
echo ""
echo "=== Copying configs ==="
for relative_path in "${config_files[@]}"; do
  cp "${CONFIG_ROOT}/${relative_path}" "$PROVENANCE_DIR/configs/" 2>/dev/null || true
done
echo "  Copied $(ls "$PROVENANCE_DIR/configs/"*.yaml 2>/dev/null | wc -l) config files"

# 3. Copy figures
echo ""
echo "=== Copying figures ==="
for filename in "${figure_files[@]}"; do
  cp "${FIGURE_ROOT}/${filename}" "$PROVENANCE_DIR/figures/" 2>/dev/null || echo "  Missing ${filename}"
done
echo "  Copied $(ls "$PROVENANCE_DIR/figures/"* 2>/dev/null | wc -l) figure files"

# 4. Copy scripts
echo ""
echo "=== Copying scripts ==="
for filename in "${script_files[@]}"; do
  cp "${SCRIPT_ROOT}/${filename}" "$PROVENANCE_DIR/scripts/"
done
echo "  Copied $(ls "$PROVENANCE_DIR/scripts/"* 2>/dev/null | wc -l) script files"

# 5. Copy table summary
echo ""
echo "=== Copying table summary ==="
cp "$RESULTS_DIR/table3_summary.md" "$PROVENANCE_DIR/" 2>/dev/null || echo "  Table summary not found (generate with generate_table3_summary.py)"

# 6. Create README
echo ""
echo "=== Creating README ==="
cat > "$PROVENANCE_DIR/README.md" <<EOFREADME
# Figure 2 + Table 3 Provenance Bundle (Post-Evaluator-Fix Rerun)

**Generated:** ${GENERATED_AT}
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
