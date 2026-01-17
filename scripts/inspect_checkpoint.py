#!/usr/bin/env python3
"""Quick checkpoint inspection tool."""
import sys
import torch

def main():
    # Hardcoded paths for inspection
    paths = [
        "/home/buiksat/trm_bellman/checkpoints/exp2_contraction_sweep/lz_095/seed41/model_step_5000.pt",
        "/home/buiksat/trm_bellman/checkpoints/exp1_v4/model_b/seed41/model_step_5000.pt",
    ]

    for path in paths:
        print(f"\n{'='*60}")
        print(f"Loading: {path}")
        print(f"{'='*60}")
        try:
            ckpt = torch.load(path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"Error loading: {e}")
            continue

        keys = list(ckpt.keys())
        print(f"\nTop-level keys ({len(keys)}): {keys[:10]}...")

        if "config" in ckpt:
            print(f"\nConfig: {ckpt['config']}")

        if "model_state_dict" in ckpt:
            print("\nFirst 10 model weights (from model_state_dict):")
            for k in list(ckpt["model_state_dict"].keys())[:10]:
                shape = ckpt["model_state_dict"][k].shape
                print(f"  {k}: {list(shape)}")
            # Check embedding size to infer vocab_size
            for k, v in ckpt["model_state_dict"].items():
                if "embed" in k.lower() and "weight" in k.lower():
                    print(f"\nEmbedding layer '{k}': {list(v.shape)} -> vocab_size = {v.shape[0]}")
                    break
        else:
            print("\nFirst 10 weights (direct keys):")
            for k in keys[:10]:
                if isinstance(ckpt[k], torch.Tensor):
                    print(f"  {k}: {list(ckpt[k].shape)}")
            # Check embedding size to infer vocab_size
            for k in keys:
                if "embed" in k.lower() and "weight" in k.lower():
                    if isinstance(ckpt[k], torch.Tensor):
                        print(f"\nEmbedding layer '{k}': {list(ckpt[k].shape)} -> vocab_size = {ckpt[k].shape[0]}")
                        break

    return 0

if __name__ == "__main__":
    sys.exit(main())
