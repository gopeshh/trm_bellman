#!/usr/bin/env python3
"""Batch inspection for validation."""

import sys
from pathlib import Path
from collections import Counter

import torch

def inspect_batch(batch_path: str) -> dict:
    """Inspect a batch file and return statistics."""
    data = torch.load(batch_path, map_location="cpu")

    states = data.get("states", [])
    metadata = data.get("metadata", {})

    info = {
        "path": batch_path,
        "num_states": len(states),
        "metadata": metadata if isinstance(metadata, dict) else "present" if metadata else "missing",
    }

    if states:
        # Count empties
        empties = []
        state_ids = []
        parent_ids = []
        action_ids = []
        action_sources = []

        for s in states:
            empties.append(s.empties)
            state_ids.append(s.state_id)
            if hasattr(s, 'parent_id') and s.parent_id is not None:
                parent_ids.append(s.parent_id)
            if hasattr(s, 'action_id') and s.action_id is not None:
                action_ids.append(s.action_id)
            if hasattr(s, 'action_source') and s.action_source is not None:
                action_sources.append(s.action_source)

        info["empties_distribution"] = dict(Counter(empties))
        info["unique_state_ids"] = len(set(state_ids))
        info["duplicate_check"] = "PASS" if len(set(state_ids)) == len(states) else f"FAIL: {len(states) - len(set(state_ids))} duplicates"

        if parent_ids:
            info["has_parent_info"] = f"Yes ({len(parent_ids)}/{len(states)})"
        else:
            info["has_parent_info"] = "No"

        if action_sources:
            info["action_source_distribution"] = dict(Counter(action_sources))
        else:
            info["action_source_distribution"] = "Not available"

    return info

def main():
    batch_paths = [
        "artifacts/eval_batches/b0.pt",
        "artifacts/eval_batches/b1.pt",
    ]

    print("=" * 80)
    print("BATCH INSPECTION REPORT")
    print("=" * 80)

    for path in batch_paths:
        if not Path(path).exists():
            print(f"\n{path}: NOT FOUND")
            continue

        print(f"\n{'='*60}")
        print(f"Batch: {path}")
        print("=" * 60)

        info = inspect_batch(path)
        for key, value in info.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
