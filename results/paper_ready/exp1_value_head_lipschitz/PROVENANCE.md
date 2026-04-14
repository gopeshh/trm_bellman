# Exp1 Value-Head Lipschitz Diagnostic: Provenance

Generated: 2026-04-14T02:01:36.978528+00:00

## Inputs

- Checkpoint base: `/home/buiksat/trm_bellman/checkpoints/exp1_v4_refreeze`
- Batch path: `/home/buiksat/trm_bellman/artifacts/eval_batches/exp1_v4_refreeze/b0.pt`
- Seeds: `41,42,43,44,45,46,47,48,49,50`
- n_eval: `8`
- eps grid: `1e-05,3e-05,1e-04,3e-04,1e-03`
- trials per eps: `8`
- directions per trial: `16`

## Regeneration

```bash
cd /home/buiksat/fbsource/fbcode
buck2 run //buiksat_trm:exp1_value_head_lipschitz -- --n_eval 8 --eps_list 1e-05,3e-05,0.0001,0.0003,0.001 --num_trials 8 --num_directions 16
```