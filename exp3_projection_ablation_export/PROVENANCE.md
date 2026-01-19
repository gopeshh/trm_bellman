# Exp3 Provenance

**Generated:** 2026-01-18T13:29:30.623049

## Experiment Configuration

- **Experiment:** Exp3 (Training-time projection ablation)
- **Conditions:** 2×3 matrix (contraction × projection radius)
- **Seeds:** 42 (single seed diagnostic)
- **Training steps:** 5000
- **Dataset:** sudoku-4x4-trivial

## Config Files

- `configs/exp3_projection_ablation/nc_r10.yaml`
- `configs/exp3_projection_ablation/nc_r100.yaml`
- `configs/exp3_projection_ablation/nc_rdis.yaml`
- `configs/exp3_projection_ablation/c_r10.yaml`
- `configs/exp3_projection_ablation/c_r100.yaml`
- `configs/exp3_projection_ablation/c_rdis.yaml`

## Checkpoint Paths

- `results/exp3/nc_r10_s42/model_step_5000.pt`
- `results/exp3/nc_r100_s42/model_step_5000.pt`
- `results/exp3/nc_rdis_s42/model_step_5000.pt`
- `results/exp3/c_r10_s42/model_step_5000.pt`
- `results/exp3/c_r100_s42/model_step_5000.pt`
- `results/exp3/c_rdis_s42/model_step_5000.pt`

## Commands to Reproduce

```bash
# Training (run all 6 conditions)
for config in nc_r10 nc_r100 nc_rdis c_r10 c_r100 c_rdis; do
  buck2 run //buiksat_trm:upi_trm_train -- \
    --config buiksat_trm/configs/exp3_projection_ablation/${config}.yaml \
    --seed 42 \
    --dataset-paths buiksat_trm/data/sudoku-4x4-trivial \
    --checkpoint-dir buiksat_trm/results/exp3/${config}_s42
done

# Evaluation
buck2 run //buiksat_trm:eval_exp3_projection_ablation
```
