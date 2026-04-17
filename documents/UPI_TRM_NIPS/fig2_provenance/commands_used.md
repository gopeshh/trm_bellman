# Commands Used to Generate Figure 2 Provenance

## 1. Locate Figure Generation Script

```bash
# Found via glob search
ls scripts/plot*.py
# Result: scripts/plot_table3_baselines.py
```

## 2. Verify Figure Location

```bash
ls /home/buiksat/UPI_TRM/UPI_TRM_NIPS/figures/trivial_baselines_vs_no_contraction_success_vs_steps.png
# Result: trivial_baselines_vs_no_contraction_success_vs_steps.png
```

## 3. Identify Source Log Files

```bash
ls -la results/table3_baselines/*.log
# Shows 18 log files (6 methods x 3 seeds)
```

## 4. Extract Final Success Rates

```bash
cd results/table3_baselines
for method in persistent_nc episodic_nc episodic_c_clean ppo a2c dqn; do
    echo "=== $method ==="
    for seed in 42 123 456; do
        logfile="${method}_s${seed}.log"
        if [ -f "$logfile" ]; then
            final=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "eval_success_rate=\K[0-9.]+")
            step=$(grep "eval_success_rate" "$logfile" | tail -1 | grep -oP "^\[step \K[0-9]+")
            echo "  seed $seed: step=$step, success=$final"
        fi
    done
done
```

## 5. Check Training Protocol in Log Headers

```bash
head -100 results/table3_baselines/persistent_nc_s42.log
head -100 results/table3_baselines/episodic_c_clean_s42.log
head -100 results/table3_baselines/ppo_s42.log
```

## 6. Verify Config Settings

```bash
cat configs/ablations/upi_trm_feasibility_persistent_z_no_contraction.yaml
cat configs/ablations/upi_trm_feasibility_no_contraction.yaml
cat configs/exp3_projection_ablation/c_rdis.yaml
cat configs/baselines/ppo_trm_feasibility.yaml
cat configs/baselines/a2c_trm_feasibility.yaml
cat configs/baselines/dqn_trm_feasibility.yaml
```

## 7. Check Table 3 in Paper

```bash
grep -n "93\|90\|31\|28\|30" /home/buiksat/UPI_TRM/UPI_TRM_NIPS/main.tex | \
    grep -E "(persistent|episodic|PPO|A2C|DQN|93\.3|90\.7|31\.3|28\.0|30\.0)"
```

## 8. Verify Horizon T

```bash
# Check configs for max_edits
grep -E "max_edits" configs/ablations/*.yaml configs/baselines/*.yaml
# Result: All configs use max_edits: 16

# Check RLConfig default
grep "max_edits" rl/config.py
# Result: max_edits: int = 16
```

## 9. Get Git Commit

```bash
git log -1 --format='%H %s'
```

## 10. Regenerate Figure (if needed)

```bash
cd ~/fbsource/fbcode
buck2 run //buiksat_trm:plot_table3_baselines \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true
```

## 11. Update main.tex (2026-01-20)

Fixed horizon discrepancy: paper incorrectly stated T=20, actual value is T=16.
Changed all occurrences in main.tex:
- Line 151: abstract
- Line 1540: Setup description
- Line 1578: Success definition
- Line 1581: "within 16 attempts" (was 20)
- Line 1585: Table 3 caption
- Line 1622: Figure 2 caption
- Line 2108: Appendix baseline comparison
- Line 2119: Appendix figure caption
- Line 2285: Exp5 figure caption
- Line 2309: 6-8 empties figure caption

---

## Provenance Summary Generated

- `fig2_protocol.md` - Paper-safe protocol description (no run IDs)
- `fig2_table3_consistency.md` - Verification that Figure 2 and Table 3 match
- `fig2_inputs.json` - Full provenance with exact paths, settings, and values
- `commands_used.md` - This file
