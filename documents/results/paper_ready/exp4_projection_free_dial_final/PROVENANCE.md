# Exp4 Provenance: Projection-free Contraction Dial (Final)

**Generated:** 2026-01-18T17:30:11.650029
**Git SHA:** None

## Checkpoint and Config

- **Checkpoint:** /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt
- **YAML Config:** /home/buiksat/trm_bellman/configs/exp3_projection_ablation/nc_rdis.yaml
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

This directly modulates the Lipschitz constant of the z→z mapping.

## Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Dial scales | [1.0, 0.85, 0.7, 0.55] |
| Seeds | [41, 42, 43] |
| n_train | 2 |
| Eval depths | [4, 8, 16] |
| B0 composition | 70 easy + 30 hard |
| B1 cap | 1500 |

## Batch Provenance

- **B0:** 100 states, hash: `9ceab78310f3`
- **B1:** 789 states, hash: `4a3bd6f6fd5b`
- B1 uses union of top-5 actions from all dial scales + 5 random actions per B0 state

## Regeneration Command

```bash
buck2 run //buiksat_trm:exp4_final -- \
    --checkpoint /home/buiksat/trm_bellman/results/exp3/nc_rdis_s42/model_step_5000.pt \
    --config_yaml /home/buiksat/trm_bellman/configs/exp3_projection_ablation/nc_rdis.yaml \
    --out_dir /home/buiksat/trm_bellman/results/paper_ready/exp4_projection_free_dial_final
```
