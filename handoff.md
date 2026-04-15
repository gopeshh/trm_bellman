# Handoff

Date: 2026-04-15
Repo: `/home/buiksat/trm_bellman`
Branch: `feature/upi-trm-clean`

## Current State

- Working tree was clean at handoff time before adding this note.
- Canonical manuscript repository: `/home/buiksat/UPI_TRM/UPI_TRM_NIPS`
- Canonical multi-month experiment plan: `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/NIPS_PLAN.md`
- Locked experiment status remains:
  - no-mask hard `4x4` is the main capability result
  - `M1` no-mask hard `4x4` is complete: UPI `0.574` mean (`std 0.122`) over seeds `0..9`; in-house A2C `0.000` over seeds `0..3`
  - controlled no-mask hard `4x4` `2x2` is complete: `nc_r0 = 0.350`, `nc_r10 = 0.482`, `c_r0 = 0.374`, `c_r10 = 0.502`
  - interpretation: projection is the primary stabilizer on hard `4x4`; contraction adds only a small average lift and does not reduce variance under projection
- Another-domain expansion was scoped and explicitly dropped as too expensive before the May 4 deadline.

## Next Priorities

1. Finish `PF2` / `PF3` in `/home/buiksat/UPI_TRM/UPI_TRM_NIPS/PHASE0_EXPERIMENT_QUEUE.md` for the `exp1_v4` depth-mismatch package.
2. Keep code work focused on the existing Sudoku pipeline, not new domains.
3. If multi-domain work is revisited later, `ARC` is the cheapest candidate, but it still needs checker/env/train wiring.
