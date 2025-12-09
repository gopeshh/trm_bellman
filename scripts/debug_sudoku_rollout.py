#!/usr/bin/env python3
"""
Debug script to trace through a single Sudoku rollout end-to-end.

This script helps diagnose whether:
1. The environment correctly loads puzzles and solutions
2. The checker scores are computed correctly
3. The success detection works when puzzles are solved
4. The policy is actually making reasonable edits

Usage:
    # Run with random actions (sanity check env):
    python scripts/debug_sudoku_rollout.py --dataset-paths data/sudoku-4x4-ultra-easy --random

    # Run with a trained checkpoint:
    python scripts/debug_sudoku_rollout.py \
        --dataset-paths data/sudoku-4x4-ultra-easy \
        --checkpoint checkpoints/rl_sudoku-4x4-ultra-easy_seed42/model_step_2000.pt

    # Run with oracle (cheat) to verify env works when puzzle is solved correctly:
    python scripts/debug_sudoku_rollout.py --dataset-paths data/sudoku-4x4-ultra-easy --oracle
"""

import argparse
import os
import sys
import torch

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from upi_trm_train import build_dataset_from_paths, sudoku_checker, DummyPuzzleDataset
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def format_4x4_grid(tokens, vocab_size=6):
    """Format a 16-token sequence as a 4x4 grid for display."""
    # Encoding: 0=PAD, 1=empty, 2-5=digits 1-4
    lines = []
    for row in range(4):
        cells = []
        for col in range(4):
            tok = tokens[row * 4 + col].item() if torch.is_tensor(tokens) else tokens[row * 4 + col]
            if tok == 1:
                cells.append(".")
            elif 2 <= tok <= 5:
                cells.append(str(tok - 1))  # Convert back to digit
            else:
                cells.append("?")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def action_to_string(action, seq_len, vocab_size, stop_action_id):
    """Convert action index to human-readable string."""
    if action == stop_action_id:
        return "STOP"
    pos = action // vocab_size
    tok = action % vocab_size
    row = pos // 4
    col = pos % 4
    if tok == 1:
        digit = "empty"
    elif 2 <= tok <= 5:
        digit = str(tok - 1)
    else:
        digit = f"tok{tok}"
    return f"(row={row}, col={col}, digit={digit})"


def run_debug_rollout(
    dataset_path: str,
    checkpoint_path: str = None,
    config_path: str = None,
    puzzle_idx: int = 0,
    max_edits: int = 16,
    use_oracle: bool = False,
    use_random: bool = False,
    hidden_size: int = 64,
):
    """
    Run a single episode with detailed logging.
    
    Args:
        dataset_path: Path to Sudoku dataset
        checkpoint_path: Optional path to trained model checkpoint
        config_path: Optional path to YAML config
        puzzle_idx: Which puzzle to test
        max_edits: Maximum number of edits
        use_oracle: If True, always pick the correct action (cheat mode)
        use_random: If True, use random actions (no model needed)
        hidden_size: Model hidden size (must match checkpoint if loading)
    """
    print("=" * 60)
    print("DEBUG SUDOKU ROLLOUT")
    print("=" * 60)
    
    # Load dataset
    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=[dataset_path] if dataset_path else None,
        pool_size=max(puzzle_idx + 1, 8),
    )
    print(f"\nDataset: {dataset_path}")
    print(f"  Size: {len(dataset)}")
    print(f"  seq_len: {seq_len}, vocab_size: {vocab_size}")
    
    # Create environment
    env_cfg = PlanEditEnvConfig(
        max_edits=max_edits,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=vocab_size,
        solved_threshold=10.0,
        stop_action_mode="disabled",
    )
    env = PlanEditEnv(dataset, sudoku_checker, env_cfg)
    num_actions = seq_len * vocab_size + 1
    stop_action_id = num_actions - 1
    env.set_stop_action_id(stop_action_id)
    
    # Load model if checkpoint provided
    model = None
    if checkpoint_path and not use_random and not use_oracle:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        trm_cfg_dict = dict(
            batch_size=1,
            seq_len=seq_len,
            puzzle_emb_ndim=0,
            puzzle_emb_len=0,
            num_puzzle_identifiers=num_identifiers,
            vocab_size=vocab_size,
            H_cycles=2,
            L_cycles=2,
            H_layers=0,
            L_layers=1,
            hidden_size=hidden_size,
            expansion=2.0,
            num_heads=max(4, hidden_size // 16),
            pos_encodings="rope",
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            halt_max_steps=2,
            halt_exploration_prob=0.0,
            forward_dtype="float32",
            mlp_t=False,
            no_ACT_continue=True,
            rl_enable_value_head=True,
            rl_enable_contraction=True,
            rl_target_Lz=0.9,
            rl_target_Lv=5.0,
            rl_enable_policy_head=True,
            rl_num_actions=num_actions,
            rl_latent_ball_radius=10.0,
        )
        model = TinyRecursiveReasoningModel_ACTV1(trm_cfg_dict)
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle potential prefix issues
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace("_orig_mod.", "").replace("model.", "")
            cleaned_state_dict[clean_key] = value
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        print("  Model loaded successfully")
    
    # Reset environment
    x, y = env.reset(idx=puzzle_idx)
    
    print(f"\n{'=' * 60}")
    print(f"EPISODE {puzzle_idx}")
    print("=" * 60)
    
    # Show initial state
    print("\n[INITIAL STATE]")
    print(f"Puzzle (inputs):\n{format_4x4_grid(x['inputs'])}")
    print(f"\nInitial plan (y):\n{format_4x4_grid(y)}")
    
    solution = x.get("solution")
    if solution is not None:
        print(f"\nSolution:\n{format_4x4_grid(solution)}")
        max_score = sudoku_checker(x, solution)
        print(f"\nMax possible score: {max_score}")
    else:
        print("\nNo solution available in dataset!")
        max_score = None
    
    initial_score = sudoku_checker(x, y)
    print(f"Initial score: {initial_score}")
    
    # Count empty cells
    empty_cells = [(i, solution[i].item() if solution is not None else None) 
                   for i in range(seq_len) if y[i].item() == 1]
    print(f"Empty cells to fill: {len(empty_cells)}")
    
    # Get action mask
    action_mask = env.get_action_mask()
    if action_mask is not None:
        valid_count = action_mask.sum().item()
        print(f"Valid actions: {valid_count} out of {num_actions}")
    
    print("\n" + "-" * 60)
    print("ROLLOUT")
    print("-" * 60)
    
    total_reward = 0.0
    done = False
    step = 0
    
    # For oracle mode, build a map of which cells need to be filled
    oracle_actions = []
    if use_oracle and solution is not None:
        for pos in range(seq_len):
            if y[pos].item() != solution[pos].item():
                action = pos * vocab_size + solution[pos].item()
                oracle_actions.append(action)
        print(f"Oracle has {len(oracle_actions)} actions planned")
    
    z = None  # For persistent latent mode
    
    while not done and step < max_edits:
        # Select action
        if use_oracle and oracle_actions:
            action = oracle_actions.pop(0)
            action_source = "ORACLE"
        elif use_random:
            # Random valid action
            if action_mask is not None:
                valid_actions = torch.where(action_mask)[0]
                action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
            else:
                action = torch.randint(num_actions, (1,)).item()
            action_source = "RANDOM"
        elif model is not None:
            # Use model policy
            batched = state_is_batched(x)
            batch_x = prepare_batch_x(x, device=torch.device("cpu"), batched=batched)
            plan = prepare_plan(y, device=torch.device("cpu"), batched=batched)
            
            mask = action_mask
            with torch.no_grad():
                dist, z_new = model.policy_dist(batch_x, plan, n=4, action_mask=mask, z=z)
                # Greedy
                action = dist.logits.argmax(dim=-1).item()
                z = z_new  # Carry forward for persistent mode
            
            # Show top 3 actions
            probs = dist.probs[0]
            top3 = probs.topk(3)
            top3_str = ", ".join([
                f"{action_to_string(a.item(), seq_len, vocab_size, stop_action_id)}:{p:.3f}"
                for a, p in zip(top3.indices, top3.values)
            ])
            action_source = f"MODEL (top3: {top3_str})"
        else:
            print("ERROR: No action source (need --checkpoint, --random, or --oracle)")
            break
        
        # Take step
        (x_next, y_next), reward, done, info = env.step(action)
        total_reward += reward
        
        action_str = action_to_string(action, seq_len, vocab_size, stop_action_id)
        print(f"\nStep {step}:")
        print(f"  Action: {action} = {action_str}")
        print(f"  Source: {action_source}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        if info.get("done_reason"):
            print(f"  Done reason: {info['done_reason']}")
        if info.get("phi_new") is not None:
            print(f"  New score: {info['phi_new']:.3f}")
        
        x, y = x_next, y_next
        step += 1
    
    # Final state
    print("\n" + "-" * 60)
    print("FINAL STATE")
    print("-" * 60)
    print(f"\nFinal plan:\n{format_4x4_grid(y)}")
    
    final_score = sudoku_checker(x, y)
    print(f"\nFinal score: {final_score}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Steps taken: {step}")
    
    is_solved = max_score is not None and abs(final_score - max_score) < 1e-6
    print(f"\nIS SOLVED: {is_solved}")
    
    if not is_solved and solution is not None:
        # Show differences
        print("\nDifferences from solution:")
        for pos in range(seq_len):
            if y[pos].item() != solution[pos].item():
                row, col = pos // 4, pos % 4
                y_digit = y[pos].item() - 1 if y[pos].item() >= 2 else "empty"
                sol_digit = solution[pos].item() - 1
                print(f"  Position ({row},{col}): got {y_digit}, expected {sol_digit}")
    
    print("\n" + "=" * 60)
    
    return is_solved, final_score, total_reward


def main():
    parser = argparse.ArgumentParser(description="Debug Sudoku rollout")
    parser.add_argument("--dataset-paths", nargs="+", default=["data/sudoku-4x4-ultra-easy"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--puzzle-idx", type=int, default=0, help="Which puzzle to test")
    parser.add_argument("--max-edits", type=int, default=16, help="Maximum edits per episode")
    parser.add_argument("--oracle", action="store_true", help="Use oracle (cheat) actions")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    parser.add_argument("--hidden-size", type=int, default=64, help="Model hidden size")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()
    
    if not args.oracle and not args.random and not args.checkpoint:
        print("Note: Running with --random since no checkpoint specified.")
        print("Use --oracle to verify env works with correct actions.")
        print("Use --checkpoint PATH to test a trained model.")
        args.random = True
    
    solved_count = 0
    for ep in range(args.num_episodes):
        idx = args.puzzle_idx + ep
        is_solved, _, _ = run_debug_rollout(
            dataset_path=args.dataset_paths[0] if args.dataset_paths else None,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            puzzle_idx=idx,
            max_edits=args.max_edits,
            use_oracle=args.oracle,
            use_random=args.random,
            hidden_size=args.hidden_size,
        )
        if is_solved:
            solved_count += 1
    
    if args.num_episodes > 1:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: Solved {solved_count}/{args.num_episodes} episodes")
        print("=" * 60)


if __name__ == "__main__":
    main()

