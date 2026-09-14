#!/bin/bash
# Real Stage A lane driver, chain run5: $1=gpu, rest = seed positions.
gpu=$1; shift
REG=/home/buiksat/trm_bellman/configs/policy_improvement_exp1b/registry.json
for pos in "$@"; do
  read -r rid seed < <(python3 -c "
import json
rows={r['seed_position']:r for r in json.load(open('$REG'))['rows']}
r=rows[int('$pos')]
print(r['run_id'], r['seed'])
")
  [ -n "$rid" ] && [ -n "$seed" ] || { echo "registry lookup failed for position $pos"; exit 1; }
  echo "=== lane gpu$gpu seed_position $pos ($rid) start $(date '+%H:%M:%S') ==="
  /home/buiksat/real-stage-a-run5.sh "$pos" "$rid" "$seed" "$gpu" \
    > /tmp/run5-seed$pos.log 2>&1
  rc=$?
  echo "=== lane gpu$gpu seed_position $pos done rc=$rc $(date '+%H:%M:%S') ==="
  [ $rc -eq 0 ] || { echo "LANE $gpu ABORT at position $pos"; exit $rc; }
done
echo "LANE $gpu COMPLETE"
