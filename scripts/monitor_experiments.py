#!/usr/bin/env python3
"""Monitor experiment progress."""

import os
import re
import glob
from datetime import datetime

EXP_DIR = "/home/buiksat/trm_bellman/runs/constraint_v2_20260107_160240"

print(f"\n{'='*60}")
print(f"EXPERIMENT STATUS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print(f"{'='*60}\n")

logs = sorted(glob.glob(f"{EXP_DIR}/*.log"))
print(f"Total experiments: {len(logs)}\n")

results = []
for log_path in logs:
    name = os.path.basename(log_path).replace('.log', '')
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Find last step
        step_matches = re.findall(r'\[step (\d+)\]', content)
        step = step_matches[-1] if step_matches else '0'

        # Find last eval
        success_matches = re.findall(r'eval_success_rate=(\d+\.\d+)', content)
        success = success_matches[-1] if success_matches else 'N/A'

        score_matches = re.findall(r'eval_mean_score=(\d+\.\d+)', content)
        score = score_matches[-1] if score_matches else 'N/A'

        results.append((name, step, success, score))
    except Exception as e:
        results.append((name, '0', 'error', str(e)[:20]))

# Sort by success rate (descending)
def sort_key(x):
    try:
        return float(x[2])
    except:
        return -1

results.sort(key=sort_key, reverse=True)

print(f"{'Experiment':<50} {'Step':<8} {'Success':<10} {'Score':<10}")
print(f"{'-'*50} {'-'*8} {'-'*10} {'-'*10}")
for name, step, success, score in results:
    print(f"{name:<50} {step:<8} {success:<10} {score:<10}")

print(f"\n{'='*60}")
