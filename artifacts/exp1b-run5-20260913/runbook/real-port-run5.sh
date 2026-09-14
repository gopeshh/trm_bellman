#!/bin/bash
# Sync the real parity fix into the fbsource cell and rebuild the real PARs.
#
# Run this only after the scratch chain-c confirmation has finished: it rebuilds
# the mode/opt launcher, whose digest the scratch chain also registers.
set -euo pipefail

REAL=/home/buiksat/trm_bellman
EVAL=/home/buiksat/trm_bellman_evaluator
CELL=/data/repos/fbsource/fbcode/buiksat_trm
OWNER=/home/buiksat/upi-trm-owner-20260911

FILES=(
  policy_improvement_full_backend.py
  scripts/policy_improvement_exp1b_schema.py
  configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json
  tests/test_policy_improvement_exp1b_unittest.py
)

COMMIT=$(git -C "$REAL" rev-parse HEAD)
echo "=== real producer commit $COMMIT ==="
[ -z "$(git -C "$REAL" status --porcelain)" ] || { echo "producer tree is dirty" >&2; exit 1; }

echo "=== $(date '+%H:%M:%S') sync the evaluator clone ==="
git -C "$EVAL" fetch --quiet origin
git -C "$EVAL" checkout --quiet --detach "$COMMIT"
[ "$(git -C "$EVAL" rev-parse HEAD)" = "$COMMIT" ] || { echo "evaluator not at $COMMIT" >&2; exit 1; }
[ -z "$(git -C "$EVAL" status --porcelain)" ] || { echo "evaluator tree is dirty" >&2; exit 1; }
echo "evaluator at $COMMIT, clean"

echo "=== $(date '+%H:%M:%S') sync the fbsource cell ==="
for f in "${FILES[@]}"; do
  cp "$REAL/$f" "$CELL/$f"
  cmp -s "$REAL/$f" "$CELL/$f" || { echo "CELL SYNC MISMATCH: $f" >&2; exit 1; }
  echo "  synced $f"
done

echo "=== $(date '+%H:%M:%S') rebuild the real runtimes (dev-nosan) ==="
cd /data/repos/fbsource
buck2 build @fbcode//mode/dev-nosan \
  fbcode//buiksat_trm:upi_trm_train \
  fbcode//buiksat_trm:policy_improvement_full \
  fbcode//buiksat_trm:policy_improvement_theory_bridge \
  fbcode//buiksat_trm:policy_improvement_audit \
  fbcode//buiksat_trm:policy_improvement_analysis \
  fbcode//buiksat_trm:policy_improvement_exp1b_auditor_bin \
  2>&1 | tail -30
RC=${PIPESTATUS[0]}
echo "BUCK_DEV_EXIT=$RC"
[ "$RC" -eq 0 ] || exit "$RC"

echo "=== $(date '+%H:%M:%S') rebuild the launcher (mode/opt) ==="
buck2 build @fbcode//mode/opt fbcode//buiksat_trm:phase4_runtime_launcher 2>&1 | tail -15
RC=${PIPESTATUS[0]}
echo "BUCK_OPT_EXIT=$RC"
[ "$RC" -eq 0 ] || exit "$RC"

ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
echo "=== $(date '+%H:%M:%S') new real PAR digests ==="
for n in upi_trm_train policy_improvement_audit policy_improvement_analysis \
         policy_improvement_full policy_improvement_theory_bridge; do
  printf '%s  %s\n' "$(sha256sum "$RT/__${n}__/${n}.par" | cut -c1-64)" "$n"
done
printf '%s  %s\n' \
  "$(sha256sum "$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par" | cut -c1-64)" \
  launcher

echo "$COMMIT" > "$OWNER/.run5_commit"
echo "=== $(date '+%H:%M:%S') real port complete; run real-mint-run5.sh next ==="
