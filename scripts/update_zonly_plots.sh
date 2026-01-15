#!/bin/bash
# Script to update plots with latest z-only contraction experiment data
# Run this after experiments complete to regenerate the final figure

set -e

echo "=== Parsing z-only contraction logs ==="
python3 /tmp/parse_zonly_logs.py > /tmp/zonly_data.csv 2>&1 || echo "Parser ran"

# Check if we have enough data points
POINTS=$(wc -l < /tmp/zonly_data.csv)
echo "Found $POINTS data points"

# Update the CSV
echo "=== Updating CSV ==="
# First, remove any existing contraction_zonly data
grep -v "contraction_zonly" ~/fbsource/fbcode/results/plot_data/plot_data_feasibility_learning_curves.csv > /tmp/base.csv
# Append new data
cat /tmp/base.csv /tmp/zonly_data.csv > ~/fbsource/fbcode/results/plot_data/plot_data_feasibility_learning_curves.csv

echo "=== Generating plots ==="
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:plot_feasibility_curves -c fbcode.nvcc_arch=a100

echo "=== Copying to paper repo ==="
cp ~/fbsource/fbcode/results/plots/feasibility_success_vs_steps.png \
   ~/UPI_TRM/UPI_TRM_ICML/figures/trivial_baselines_vs_no_contraction_success_vs_steps.png
cp ~/fbsource/fbcode/results/plots/feasibility_success_vs_steps.pdf \
   ~/UPI_TRM/UPI_TRM_ICML/figures/trivial_baselines_vs_no_contraction_success_vs_steps.pdf

echo "=== Done ==="
ls -la ~/UPI_TRM/UPI_TRM_ICML/figures/trivial_baselines_vs_no_contraction_success_vs_steps.*
