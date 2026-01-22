#!/usr/bin/env python3
"""Evaluate random policy on 4x4 Sudoku to establish baseline success rate."""

import argparse
import random
from pathlib import Path

import torch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sudoku_dataset import SudokuDataset
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.checkers import make_checker


def evaluate_random_policy(
    dataset_path: str,
    num_episodes: int = 50,
    max_edits: int = 16,
    seed: int = 42,
    use_action_mask: bool = True,
) -> dict:
    """
    Evaluate a random policy on Sudoku.

    Args:
        dataset_path: Path to sudoku dataset
        num_episodes: Number of evaluation episodes
        max_edits: Maximum edits per episode
        seed: Random seed
        use_action_mask: If True, only sample from valid actions (action masking)

    Returns:
        Dict with success_rate, mean_score, etc.
    """
    random.seed(seed)

    # Load dataset
    dataset = SudokuDataset(dataset_path)
    print(f"Loaded dataset with {len(dataset)} puzzles")

    # Create checker
    checker = make_checker(
        "feasibility",
        violation_weight=2.0,
        zerocand_weight=5.0,
    )

    # Create environment config
    env_cfg = PlanEditEnvConfig(
        max_edits=max_edits,
        reward_shaping=True,
        fail_terminal_reward=-16.0,
        solve_terminal_reward=1.0,
        C_max=16.0,
        solved_threshold=None,
    )

    # Create environment
    env = PlanEditEnv(dataset=dataset, checker=checker, config=env_cfg)

    # Track metrics
    num_solved = 0
    total_score = 0.0
    all_final_scores = []

    for episode_idx in range(num_episodes):
        x, y = env.reset(idx=episode_idx % len(dataset))
        done = False

        while not done:
            # Get valid action mask
            if use_action_mask:
                action_mask = env.get_action_mask()
                # Find valid actions (where mask is True/1)
                valid_actions = torch.where(action_mask)[0].tolist()
                if len(valid_actions) == 0:
                    # No valid actions, episode done
                    break
                action = random.choice(valid_actions)
            else:
                # Sample from all actions (excluding stop action if present)
                num_actions = env.num_actions
                action = random.randint(0, num_actions - 1)

            x, reward, done, info = env.step(action)

        # Get final score
        final_score = info.get("final_score", info.get("score", 0.0))
        all_final_scores.append(final_score)
        total_score += final_score

        # Check if solved
        if info.get("solved", False):
            num_solved += 1

    success_rate = num_solved / num_episodes
    mean_score = total_score / num_episodes

    return {
        "success_rate": success_rate,
        "mean_score": mean_score,
        "solved_count": num_solved,
        "total_episodes": num_episodes,
        "score_min": min(all_final_scores),
        "score_max": max(all_final_scores),
        "use_action_mask": use_action_mask,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate random policy on Sudoku")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sudoku-4x4-trivial",
        help="Path to sudoku dataset",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=16,
        help="Maximum edits per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-action-mask",
        action="store_true",
        help="Disable action masking (sample from all actions)",
    )
    args = parser.parse_args()

    print(f"Evaluating random policy on {args.dataset}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max edits: {args.max_edits}")
    print(f"  Seed: {args.seed}")
    print(f"  Action masking: {not args.no_action_mask}")
    print()

    results = evaluate_random_policy(
        dataset_path=args.dataset,
        num_episodes=args.num_episodes,
        max_edits=args.max_edits,
        seed=args.seed,
        use_action_mask=not args.no_action_mask,
    )

    print(f"Results:")
    print(f"  Success rate: {results['success_rate']*100:.1f}% ({results['solved_count']}/{results['total_episodes']})")
    print(f"  Mean score: {results['mean_score']:.2f}")
    print(f"  Score range: {results['score_min']:.2f} - {results['score_max']:.2f}")


if __name__ == "__main__":
    main()
