"""
Policy evaluation utilities for RL plan editing.

This module provides functions to evaluate trained TRM policies on puzzle datasets.
"""
from typing import Any, Callable, Optional, Tuple

import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_utils import sudoku_is_solved, sudoku_get_stats


def evaluate_plan_policy_with_scores(
    model: TinyRecursiveReasoningModel_ACTV1,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
    episodic_latent: bool = True,
    greedy: bool = True,
    use_sudoku_solved_criterion: bool = True,
) -> Tuple[float, float, dict]:
    """
    Evaluate a TRM + policy head in plan space on a given dataset.

    Args:
        model: TRM model with policy head
        dataset: Dataset providing puzzle instances
        checker: Function (x, y) -> score
        env_cfg: Environment configuration
        num_episodes: Number of evaluation episodes
        inner_unroll_n: Number of latent unrolling steps (default: 4)
        episodic_latent: If True (default), reinitialize z from (x,y) at every step.
            If False (persistent mode), initialize z once per episode and carry it
            forward across steps, allowing the model to accumulate information.
        greedy: If True (default), use argmax action selection for deterministic evaluation.
            If False, sample from the policy distribution (stochastic evaluation).
        use_sudoku_solved_criterion: If True (default), use sudoku_is_solved() to determine
            success (filled==N and violations==0). If False, use score matching.

    Returns:
        Tuple of (mean_checker_score, success_rate, detailed_stats)
        detailed_stats contains:
            - solved_count: number of solved episodes
            - total_episodes: total episodes run
            - score_min: minimum final score
            - score_max: maximum final score
            - max_possible_score: max score from solution (or None if no solution)
            - initial_score_mean: mean initial score before any edits
            - mean_return: mean cumulative episode reward under the internal env
            - mean_steps: mean number of steps per episode
            - invalid_action_rate: invalid actions / total steps during eval
            - final_filled_mean: mean number of filled cells at episode end (Sudoku only)
            - final_violations_mean: mean violations at episode end (Sudoku only)
            - final_zero_cand_mean: mean zero-candidate cells at episode end (Sudoku only)
    """

    device = next(model.parameters()).device
    model.eval()
    env = PlanEditEnv(dataset=dataset, checker=checker, config=env_cfg)

    if env.stop_action_id is None:
        num_actions = getattr(model.config, "rl_num_actions", None)
        if num_actions is None or num_actions <= 0:
            raise ValueError("Model config must define `rl_num_actions` > 0 for policy evaluation.")
        env.set_stop_action_id(stop_id=num_actions - 1)

    if inner_unroll_n is None:
        inner_unroll_n = 4

    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0.0, 0.0, {}

    num_solved = 0
    total_score = 0.0
    episodes_ran = 0
    total_return = 0.0
    total_steps = 0
    total_invalid_actions = 0
    all_final_scores: list = []
    all_initial_scores: list = []
    max_possible_score: Optional[float] = None

    # Sudoku-specific tracking
    all_filled: list = []
    all_violations: list = []
    all_zero_cand: list = []
    is_sudoku_task = False

    with torch.no_grad():
        for episode_idx in range(num_episodes):
            x, y = env.reset(idx=episode_idx % dataset_size)
            done = False
            episode_return = 0.0
            episode_steps = 0
            episode_invalid_actions = 0

            # Track initial score before any edits
            initial_score = float(checker(x, y))
            all_initial_scores.append(initial_score)

            # Get the optimal solution for computing max reward
            # Priority: "solution" > "labels" (both represent the solved state)
            # DO NOT fall back to "inputs" - those are puzzle clues, not solutions!
            # Note: x may be a dict or a raw tensor depending on dataset format
            if isinstance(x, dict):
                optimal_plan = x.get("solution")
                if optimal_plan is None:
                    optimal_plan = x.get("labels")  # Common alternative name in supervised datasets
            else:
                # x is not a dict (e.g., raw tensor from simple datasets)
                optimal_plan = None

            if optimal_plan is not None:
                episode_max_reward = float(checker(x, optimal_plan))
                if max_possible_score is None:
                    max_possible_score = episode_max_reward
            else:
                # No solution available - cannot compute meaningful success rate
                # Set max_reward to None to skip success counting for this episode
                episode_max_reward = None

            # === Persistent latent mode ===
            # In persistent mode, z is initialized once per episode and carried forward.
            # In episodic mode, z=None causes policy_dist to reinitialize from (x,y) each step.
            z = None
            if not episodic_latent:
                # Initialize z from (x, y) at episode start
                batched = state_is_batched(x)
                batch_x = prepare_batch_x(x, device=device, batched=batched)
                plan = prepare_plan(y, device=device, batched=batched)
                z = model.init_latent(batch_x, plan)

            for _ in range(env_cfg.max_edits):
                batched = state_is_batched(x)
                batch_x = prepare_batch_x(x, device=device, batched=batched)
                plan = prepare_plan(y, device=device, batched=batched)

                # Get action mask to prevent editing "given" cells
                action_mask = env.get_action_mask()
                if action_mask is not None:
                    action_mask = action_mask.to(device)

                # Pass z for persistent mode; z=None for episodic mode (reinitializes each step)
                dist, z_new = model.policy_dist(batch_x, plan, n=inner_unroll_n, action_mask=action_mask, z=z)

                # Greedy (argmax) for deterministic evaluation; sample for stochastic
                if greedy:
                    action = dist.logits.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()

                if action_mask is not None:
                    if action < 0 or action >= action_mask.numel() or not bool(action_mask[action].item()):
                        episode_invalid_actions += 1

                # Carry forward the updated latent in persistent mode
                if not episodic_latent:
                    z = z_new

                (x_next, y_next), reward, done, _ = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                x, y = x_next, y_next

                if done:
                    break

            final_score = float(checker(x, y))
            total_score += final_score
            all_final_scores.append(final_score)
            episodes_ran += 1
            total_return += episode_return
            total_steps += episode_steps
            total_invalid_actions += episode_invalid_actions

            # Get the final plan tensor for Sudoku-specific checks
            if isinstance(y, torch.Tensor):
                final_plan = y
            elif isinstance(y, dict) and "plan" in y:
                final_plan = y["plan"]
            else:
                final_plan = None

            # Track Sudoku-specific stats and use solution-independent success criterion
            if final_plan is not None and final_plan.numel() in (16, 81):
                is_sudoku_task = True
                total_cells, filled, violations, zero_cand = sudoku_get_stats(final_plan)
                all_filled.append(filled)
                all_violations.append(violations)
                all_zero_cand.append(zero_cand)

                # Use sudoku_is_solved for solution-independent success
                if use_sudoku_solved_criterion:
                    if sudoku_is_solved(final_plan):
                        num_solved += 1
                elif episode_max_reward is not None and abs(final_score - episode_max_reward) < 1e-6:
                    # Fallback: score-matching criterion (requires solution)
                    num_solved += 1
            else:
                # Non-Sudoku task: use score-matching criterion
                if episode_max_reward is not None and abs(final_score - episode_max_reward) < 1e-6:
                    num_solved += 1

    mean_score = total_score / float(max(episodes_ran, 1))
    success_rate = num_solved / float(max(episodes_ran, 1))

    # Compute detailed statistics
    detailed_stats = {
        "solved_count": num_solved,
        "total_episodes": episodes_ran,
        "score_min": min(all_final_scores) if all_final_scores else 0.0,
        "score_max": max(all_final_scores) if all_final_scores else 0.0,
        "max_possible_score": max_possible_score,
        "initial_score_mean": sum(all_initial_scores) / len(all_initial_scores) if all_initial_scores else 0.0,
        "mean_return": total_return / float(max(episodes_ran, 1)),
        "mean_steps": total_steps / float(max(episodes_ran, 1)),
        "invalid_action_rate": total_invalid_actions / float(max(total_steps, 1)),
    }

    # Add Sudoku-specific stats if applicable
    if is_sudoku_task and all_filled:
        detailed_stats["final_filled_mean"] = sum(all_filled) / len(all_filled)
        detailed_stats["final_violations_mean"] = sum(all_violations) / len(all_violations)
        detailed_stats["final_zero_cand_mean"] = sum(all_zero_cand) / len(all_zero_cand)

    return mean_score, success_rate, detailed_stats


def evaluate_plan_policy(
    model: TinyRecursiveReasoningModel_ACTV1,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
    episodic_latent: bool = True,
    greedy: bool = True,
) -> float:
    """
    Backwards-compatible wrapper that only returns the strict success rate.

    Args:
        episodic_latent: If True (default), reinitialize z each step.
            If False (persistent mode), carry z forward across steps.
        greedy: If True (default), use argmax for deterministic evaluation.
    """

    _, success_rate, _ = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker,
        env_cfg=env_cfg,
        num_episodes=num_episodes,
        inner_unroll_n=inner_unroll_n,
        episodic_latent=episodic_latent,
        greedy=greedy,
    )
    return success_rate
