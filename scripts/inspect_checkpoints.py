#!/usr/bin/env python3
"""Inspect checkpoint contents for validation."""

import sys
import torch
from pathlib import Path

def inspect_checkpoint(path: str) -> dict:
    """Inspect a checkpoint file and extract key information."""
    ckpt = torch.load(path, map_location="cpu")

    info = {
        "path": path,
        "keys": list(ckpt.keys()) if isinstance(ckpt, dict) else ["state_dict_only"],
    }

    if isinstance(ckpt, dict):
        # Check for rl_config
        if "rl_config" in ckpt:
            info["rl_config"] = ckpt["rl_config"]
        else:
            info["rl_config"] = "NOT FOUND"

        # Check for model_state_dict
        if "model_state_dict" in ckpt:
            model_state = ckpt["model_state_dict"]
        else:
            model_state = ckpt

        # Get step if available
        if "step" in ckpt:
            info["step"] = ckpt["step"]
    else:
        model_state = ckpt

    # Infer architecture from state dict
    if isinstance(model_state, dict):
        # Check for contraction-related keys (Lipschitz scaling)
        lip_keys = [k for k in model_state.keys() if "_lip_scale" in k]
        info["has_lipschitz_scaling"] = len(lip_keys) > 0
        info["lipschitz_key_count"] = len(lip_keys)

        # Check for value head
        value_keys = [k for k in model_state.keys() if "value_head" in k]
        info["has_value_head"] = len(value_keys) > 0

        # Check for policy head
        policy_keys = [k for k in model_state.keys() if "edit_policy" in k]
        info["has_policy_head"] = len(policy_keys) > 0

        # Get hidden size from embed layer
        for key in model_state.keys():
            if "embed_inputs.weight" in key or "inner.embed_inputs.weight" in key:
                info["hidden_size"] = model_state[key].shape[1]
                info["vocab_size"] = model_state[key].shape[0]
                break

        # Get action dim from policy head
        for key in model_state.keys():
            if "edit_policy.mlp.2.weight" in key:
                info["num_actions"] = model_state[key].shape[0]
                break

    return info

def main():
    checkpoints = [
        "checkpoints/model_a/model_step_5000.pt",
        "checkpoints/model_a/rl_checkpoint_step_5000.pt",
        "checkpoints/model_b/model_step_5000.pt",
        "checkpoints/model_b/rl_checkpoint_step_5000.pt",
    ]

    print("=" * 80)
    print("CHECKPOINT INSPECTION REPORT")
    print("=" * 80)

    for ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"\n{ckpt_path}: NOT FOUND")
            continue

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 60)

        info = inspect_checkpoint(ckpt_path)
        for key, value in info.items():
            if key == "keys":
                print(f"  {key}: {value[:10]}{'...' if len(value) > 10 else ''}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
