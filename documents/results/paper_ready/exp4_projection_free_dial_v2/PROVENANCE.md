# Exp4 Provenance: Projection-free Contraction Dial (Final v2)

**Generated:** 2026-01-18T21:46:46.262096
**Git SHA:** 6fede631455e

## Checkpoints

| Seed | Path |
|------|------|
| 41 | /home/buiksat/trm_bellman/results/exp3_v2/nc_rdis_s41/model_step_5000.pt |
| 42 | /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt |
| 43 | /home/buiksat/trm_bellman/results/exp3_v2/nc_rdis_s43/model_step_5000.pt |

## YAML Config

- **Path:** /home/buiksat/trm_bellman/configs/exp3_projection_ablation/nc_rdis.yaml
- **Config Source:** yaml (required, no inference)

## Non-Negotiables Verified

- `disable_value_head_norm: true` ✓
- `latent_ball_radius: 0.0` (projection disabled) ✓

## Dial Implementation

Contraction scaling is applied at **inference/evaluation time** by multiplying
the weight matrices of L_level (z→z) layers by the scaling factor:

```python
for module in L_level.modules():
    if hasattr(module, 'weight'):
        module.weight.data.mul_(scale_factor)
```

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Dial scales | [1.0, 0.85, 0.7, 0.55] |
| Checkpoint seeds | [41, 42, 43] |
| N (independent samples) | 12 |
| n_train | 2 |
| Eval depths | [4, 8, 16] |
| B0 composition | 70 easy + 30 hard |
| B1 cap | 1500 |

## Batch Provenance

- **B0:** 100 states, hash: `9ceab78310f3`
- **B1:** 1048 states, hash: `fca64be3b53c`
- B1 uses union of top-5 actions from ALL checkpoints × ALL scales + random actions

## Training Commands (if checkpoints were generated)

```bash
# For each seed in [41, 42, 43]:
buck2 run //buiksat_trm:upi_trm_train \
    -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true --local-only \
    -- --config configs/exp3_projection_ablation/nc_rdis.yaml \
    --seed <SEED> --output_dir results/exp3/nc_rdis_s<SEED>
```

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_final_v2 -- \
    --checkpoint_glob "results/exp3/nc_rdis_s*/model_step_5000.pt" \
    --config_yaml configs/exp3_projection_ablation/nc_rdis.yaml \
    --out_dir /home/buiksat/trm_bellman/results/paper_ready/exp4_projection_free_dial_v2
```
