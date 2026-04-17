# Hard 4x4 Trusted Baseline Summary

Paper-facing hard-4x4 aggregate for `T2.1`: UPI-TRM main anchor plus the trusted external PPO/A2C/DQN/DQN (n=5) reruns on the no-mask 6-8-empties suite.

## Aggregate Table

| Method | Steps | Seeds | Success Rate | 95% CI | Mean Return | Invalid Action Rate |
|--------|-------|-------|--------------|--------|-------------|---------------------|
| UPI-TRM | 20k | 10 | 0.574 ± 0.122 | [0.487, 0.661] | -- | -- |
| SB3 PPO | 20k | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -17.462 ± 0.000 | 1.000 ± 0.000 |
| SB3 A2C | 20k | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -18.092 ± 0.520 | 0.783 ± 0.127 |
| SB3 DQN | 20k | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -17.915 ± 0.173 | 0.843 ± 0.054 |
| SB3 DQN (n=5) | 20k | 10 | 0.000 ± 0.000 | [0.000, 0.000] | -18.027 ± 0.278 | 0.842 ± 0.043 |

## Gap Analysis

- Best external baseline by mean success: SB3 PPO, SB3 A2C, SB3 DQN, SB3 DQN (n=5) (tie at 0.0%).
- Seed-bootstrap 95% CI for `UPI-TRM - best external baseline`: [0.502, 0.644] success-rate points.
- Final-eval Wilson 95% upper bound for each external baseline's solve rate: SB3 PPO <= 0.76%, SB3 A2C <= 0.76%, SB3 DQN <= 0.76%, SB3 DQN (n=5) <= 0.76%.
- All checkpointed evals stayed at zero success: SB3 PPO 2000 / 2000, SB3 A2C 2000 / 2000, SB3 DQN 2000 / 2000, SB3 DQN (n=5) 2000 / 2000 zero-success eval points.

## Paper-Ready Interpretation

The trusted external rerun confirms the hard-4x4 no-mask capability story rather than softening it. UPI-TRM stays nontrivial at 57.4% ± 12.2% over 10 seeds, while SB3 PPO, SB3 A2C, SB3 DQN, SB3 DQN (n=5) remain at 0.0% success over 10 seeds on the same no-mask suite. The bootstrap 95% CI for the UPI-vs-best-baseline gap is [50.2, 64.4] percentage points, well above zero. Across all 40 external runs and all 8000 checkpointed evaluations during training, no trusted baseline reached nonzero success. This is enough to resolve the reviewer-trust objection with a known external codebase rather than our in-house trainer.
