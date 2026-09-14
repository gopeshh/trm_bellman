#!/bin/bash
# Real Stage B driver, chain run5: args = seed positions, in registered order.
for pos in "$@"; do
  echo "=== stage B position $pos start $(date '+%H:%M:%S') ==="
  /home/buiksat/real-stage-b-run5.sh "$pos" > /tmp/run5-stageb-$pos.log 2>&1
  rc=$?
  echo "=== stage B position $pos done rc=$rc $(date '+%H:%M:%S') ==="
  [ $rc -eq 0 ] || { echo "STAGE B ABORT at position $pos"; exit $rc; }
done
echo "STAGE B COMPLETE"
