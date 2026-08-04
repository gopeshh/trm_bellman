"""
Policy evaluation utilities for RL plan editing.

This module provides functions to evaluate trained TRM policies on puzzle datasets.
"""
import random
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_utils import sudoku_is_solved, sudoku_get_stats
from utils.dataset_provenance import sample_sha256
from utils.evaluation_artifacts import record_local_evaluation_seed
from utils.run_identity import canonical_json_sha256


@contextmanager
def _isolated_evaluation_state(
    models: Sequence[Any],
    *,
    seed: Optional[int],
) -> Iterator[None]:
    """Run evaluation without changing training RNG streams or module modes."""

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training_modes = [bool(model.training) for model in models]
    try:
        if seed is not None:
            if seed < 0 or seed > 2**32 - 1:
                raise ValueError("Evaluation seed must be in [0, 2**32 - 1].")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        for model in models:
            model.eval()
        yield
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        for model, training in zip(models, training_modes):
            model.train(training)


def _dataset_record_sha256(dataset: Any, index: int) -> str:
    sample = dataset[index]
    if not isinstance(sample, dict) or "inputs" not in sample:
        raise TypeError(
            "Per-instance evaluation records require dictionary dataset samples "
            "with an 'inputs' field."
        )
    solution = sample.get("solution", sample.get("labels"))
    return sample_sha256(sample["inputs"], solution)


def _plan_list(plan: torch.Tensor) -> list[int]:
    return [int(value) for value in plan.detach().cpu().reshape(-1).tolist()]


def evaluate_plan_policy_with_scores(
    model: Any,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    task_config: Optional[Any] = None,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
    episodic_latent: bool = True,
    greedy: bool = True,
    use_sudoku_solved_criterion: bool = True,
    policy_dist_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
    allow_cycle: bool = False,
    evaluation_seed: Optional[int] = None,
    collect_per_instance: bool = False,
    additional_models: Sequence[Any] = (),
    record_local_seeding: bool = False,
) -> Tuple[float, float, dict]:
    """
    Evaluate a TRM + policy head in plan space on a given dataset.

    Args:
        model: TRM model with policy head
        dataset: Dataset providing puzzle instances
        checker: Function (x, y) -> score
        env_cfg: Environment configuration
        task_config: Optional task-specific masking logic to keep eval aligned with training
        num_episodes: Number of evaluation episodes
        inner_unroll_n: Number of latent unrolling steps (default: 4)
        episodic_latent: If True (default), reinitialize z from (x,y) at every step.
            If False (persistent mode), initialize z once per episode and carry it
            forward across steps, allowing the model to accumulate information.
        greedy: If True (default), use argmax action selection for deterministic evaluation.
            If False, sample from the policy distribution (stochastic evaluation).
        use_sudoku_solved_criterion: If True (default), use sudoku_is_solved() to determine
            success (filled==N and violations==0). If False, use score matching.
        policy_dist_fn: Optional deployed-policy callback. When supplied, evaluation
            calls it instead of ``model.policy_dist``. This is required for an
            explicit probability-space mixture represented by two networks.
        allow_cycle: Permit repeated dataset records when ``num_episodes`` exceeds
            the dataset size. Disabled by default to prevent accidental reuse of a
            small evaluation pool.
        evaluation_seed: Optional private RNG seed. All Python, NumPy, Torch CPU,
            and Torch CUDA RNG states are restored after evaluation.
        collect_per_instance: Include one ordered record per evaluated dataset row.
        additional_models: Other modules used by ``policy_dist_fn`` whose training
            modes must be restored after evaluation.
        record_local_seeding: Derive an independent deterministic RNG seed from
            ``evaluation_seed`` and each record hash. This prevents an earlier
            episode's length from shifting later stochastic action draws.

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

    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0.0, 0.0, {}
    if num_episodes > dataset_size and not allow_cycle:
        raise ValueError(
            "Evaluation requested more episodes than distinct dataset records; "
            f"refusing to cycle the pool ({num_episodes} > {dataset_size})."
        )
    if record_local_seeding and evaluation_seed is None:
        raise ValueError("Record-local evaluation seeding requires evaluation_seed.")

    device = next(model.parameters()).device
    env = PlanEditEnv(dataset=dataset, checker=checker, config=env_cfg, task_config=task_config)

    if env.stop_action_id is None:
        num_actions = getattr(model.config, "rl_num_actions", None)
        if num_actions is None or num_actions <= 0:
            raise ValueError("Model config must define `rl_num_actions` > 0 for policy evaluation.")
        env.set_stop_action_id(stop_id=num_actions - 1)

    if inner_unroll_n is None:
        inner_unroll_n = 4

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
    per_instance: list[dict[str, Any]] = []

    models = (model, *tuple(additional_models))
    if len({id(item) for item in models}) != len(models):
        raise ValueError("Evaluation model inventory contains duplicate modules.")

    with _isolated_evaluation_state(models, seed=evaluation_seed), torch.no_grad():
        for episode_idx in range(num_episodes):
            dataset_index = episode_idx % dataset_size if allow_cycle else episode_idx
            record_sha256 = (
                _dataset_record_sha256(dataset, dataset_index)
                if collect_per_instance or record_local_seeding
                else None
            )
            episode_seed: Optional[int] = None
            if record_local_seeding:
                assert evaluation_seed is not None
                assert record_sha256 is not None
                episode_seed = record_local_evaluation_seed(
                    evaluation_seed, record_sha256
                )
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
            x, y = env.reset(idx=dataset_index)
            done = False
            episode_return = 0.0
            episode_steps = 0
            episode_invalid_actions = 0
            episode_actions: list[int] = []
            episode_rewards: list[float] = []
            done_reason: Optional[str] = None

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
                if policy_dist_fn is None:
                    dist, z_new = model.policy_dist(
                        batch_x,
                        plan,
                        n=inner_unroll_n,
                        action_mask=action_mask,
                        z=z,
                    )
                else:
                    dist, z_new = policy_dist_fn(
                        batch_x,
                        plan,
                        n=inner_unroll_n,
                        action_mask=action_mask,
                        z=z,
                    )

                # Greedy (argmax) for deterministic evaluation; sample for stochastic
                if greedy:
                    action = int(dist.logits.argmax(dim=-1).item())
                else:
                    action = int(dist.sample().item())

                if action_mask is not None:
                    if action < 0 or action >= action_mask.numel() or not bool(action_mask[action].item()):
                        episode_invalid_actions += 1

                # Carry forward the updated latent in persistent mode
                if not episodic_latent:
                    z = z_new

                (x_next, y_next), reward, done, info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                episode_actions.append(action)
                episode_rewards.append(float(reward))
                x, y = x_next, y_next

                if done:
                    done_reason = info.get("done_reason")
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

            episode_solved = False
            sudoku_stats: Optional[dict[str, int]] = None

            # Track Sudoku-specific stats and use solution-independent success criterion
            if final_plan is not None and final_plan.numel() in (16, 81):
                is_sudoku_task = True
                total_cells, filled, violations, zero_cand = sudoku_get_stats(final_plan)
                all_filled.append(filled)
                all_violations.append(violations)
                all_zero_cand.append(zero_cand)

                # Use sudoku_is_solved for solution-independent success
                if use_sudoku_solved_criterion:
                    episode_solved = bool(sudoku_is_solved(final_plan))
                elif episode_max_reward is not None and abs(final_score - episode_max_reward) < 1e-6:
                    # Fallback: score-matching criterion (requires solution)
                    episode_solved = True
                sudoku_stats = {
                    "total_cells": int(total_cells),
                    "filled": int(filled),
                    "violations": int(violations),
                    "zero_candidates": int(zero_cand),
                }
            else:
                # Non-Sudoku task: use score-matching criterion
                if episode_max_reward is not None and abs(final_score - episode_max_reward) < 1e-6:
                    episode_solved = True

            if episode_solved:
                num_solved += 1

            if collect_per_instance:
                if final_plan is None:
                    raise TypeError(
                        "Per-instance evaluation records require a tensor final plan."
                    )
                final_plan_values = _plan_list(final_plan)
                per_instance.append(
                    {
                        "record_index": dataset_index,
                        "record_sha256": record_sha256,
                        "evaluation_seed": episode_seed,
                        "success": episode_solved,
                        "initial_checker_score": initial_score,
                        "final_checker_score": final_score,
                        "undiscounted_shaped_return": episode_return,
                        "environment_interactions": episode_steps,
                        "invalid_action_count": episode_invalid_actions,
                        "termination_reason": done_reason or "not_terminated",
                        "actions": episode_actions,
                        "rewards": episode_rewards,
                        "final_plan": final_plan_values,
                        "final_plan_sha256": canonical_json_sha256(final_plan_values),
                        "sudoku": sudoku_stats,
                    }
                )

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
    if collect_per_instance:
        detailed_stats["per_instance"] = per_instance

    return mean_score, success_rate, detailed_stats


def evaluate_plan_policy(
    model: Any,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    task_config: Optional[Any] = None,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
    episodic_latent: bool = True,
    greedy: bool = True,
    allow_cycle: bool = False,
) -> float:
    """
    Backwards-compatible wrapper that only returns the strict success rate.

    Args:
        episodic_latent: If True (default), reinitialize z each step.
            If False (persistent mode), carry z forward across steps.
        greedy: If True (default), use argmax for deterministic evaluation.
        allow_cycle: Permit repeated dataset records. Disabled by default.
    """

    _, success_rate, _ = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker,
        env_cfg=env_cfg,
        task_config=task_config,
        num_episodes=num_episodes,
        inner_unroll_n=inner_unroll_n,
        episodic_latent=episodic_latent,
        greedy=greedy,
        allow_cycle=allow_cycle,
    )
    return success_rate
