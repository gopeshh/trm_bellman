#!/bin/bash
while true; do
  echo "=== $(date) ===" >> ~/trm_bellman/results/9x9_monitor.log
  for f in ~/trm_bellman/results/9x9_experiments_seed0/*50k*.log; do
    name=$(basename "$f" .log)
    last_eval=$(grep "eval_success" "$f" 2>/dev/null | tail -1)
    if [ -n "$last_eval" ]; then
      step=$(echo "$last_eval" | grep -oP '\[step \K\d+')
      success=$(echo "$last_eval" | grep -oP 'eval_success_rate=\K[0-9.]+')
      score=$(echo "$last_eval" | grep -oP 'eval_mean_score=\K[0-9.]+')
      pct=$((step * 100 / 50000))
      printf "%-18s %5s/50k (%3d%%)  success=%s  score=%s\n" "$name" "$step" "$pct" "$success" "$score"
    fi
  done >> ~/trm_bellman/results/9x9_monitor.log
  echo "" >> ~/trm_bellman/results/9x9_monitor.log
  sleep 3600
done
