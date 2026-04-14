# Exp1 Value-Head Lipschitz Diagnostic: Provenance

Generated: 2026-04-14T02:00:08.020355+00:00

## Inputs

- Checkpoint base: `/home/buiksat/trm_bellman/checkpoints/exp1_v4_refreeze`
- Batch path: `/home/buiksat/trm_bellman/artifacts/eval_batches/exp1_v4_refreeze/b0.pt`
- Seeds: `41,42`
- n_eval: `8`
- eps grid: `1e-04,1e-03`
- trials per eps: `2`
- directions per trial: `4`

## Regeneration

```bash
cd /home/buiksat/fbsource/fbcode
buck2 run //buiksat_trm:exp1_value_head_lipschitz -- --n_eval 8 --eps_list 0.0001,0.001 --num_trials 2 --num_directions 4
```