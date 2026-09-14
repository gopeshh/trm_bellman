#!/bin/bash
# Audit a published Experiment 1B result with the standalone auditor binary.
#   $1 = cell            buiksat_trm | buiksat_trm_scratch
#   $2 = project root     the tree whose registered documents the result cites
#   $3 = evidence generation directory  (holds result/ and provenance.json)
#   $4 = execution admission path
#
# The auditor reads a published result plus four registered documents. No
# checkpoint, no dataset, no evidence payload, no writes.
set -u
CELL=$1
ROOT=$2
GENDIR=$3
ADMISSION=$4

ART=/data/repos/fbsource/buck-out/v2/art/fbcode/c59829718afa7235/$CELL
BIN=$ART/__policy_improvement_exp1b_auditor_bin__/policy_improvement_exp1b_auditor_bin.par
[ -f "$BIN" ] || { echo "auditor binary not built: $BIN" >&2; exit 2; }

RESULT=$(find "$GENDIR" -maxdepth 2 -name '*.json' -path '*result*' | head -1)
[ -n "$RESULT" ] || { echo "no published result under $GENDIR" >&2; exit 2; }
echo "=== auditing $RESULT ==="

"$BIN" \
  --result "$RESULT" \
  --reduced-study-protocol "$ROOT/configs/policy_improvement_exp1b/protocol.json" \
  --reduced-study-registry "$ROOT/configs/policy_improvement_exp1b/registry.json" \
  --reduced-study-amendment "$ROOT/configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json" \
  --substitute-provenance "$GENDIR/provenance.json" \
  --parent-populations "$ROOT/configs/policy_improvement_v2/populations.json" \
  --execution-admission "$ADMISSION"
echo "AUDIT EXIT=$?"
