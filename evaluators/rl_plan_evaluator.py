from typing import Any, Callable, Optional, Tuple

import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


def evaluate_plan_policy_with_scores(
    model: TinyRecursiveReasoningModel_ACTV1,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate a TRM + policy head in plan space on a given dataset.

    Returns:
        (mean_checker_score, success_rate)
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
        return 0.0, 0.0

    num_solved = 0
    total_score = 0.0
    episodes_ran = 0

    with torch.no_grad():
        for episode_idx in range(num_episodes):
            x, y = env.reset(idx=episode_idx % dataset_size)
            done = False

            optimal_plan = x.get("solution")
            if optimal_plan is None:
                optimal_plan = x.get("inputs")
            if optimal_plan is None:
                raise KeyError("Environment state must include `solution` or `inputs` for reward reference.")
            episode_max_reward = float(checker(x, optimal_plan))

            for _ in range(env_cfg.max_edits):
                batched = state_is_batched(x)
                batch_x = prepare_batch_x(x, device=device, batched=batched)
                plan = prepare_plan(y, device=device, batched=batched)
                
                # Get action mask to prevent editing "given" cells
                action_mask = env.get_action_mask()
                if action_mask is not None:
                    action_mask = action_mask.to(device)

                dist, _ = model.policy_dist(batch_x, plan, n=inner_unroll_n, action_mask=action_mask)
                action = dist.sample().item()

                (x_next, y_next), _, done, _ = env.step(action)
                x, y = x_next, y_next

                if done:
                    break

            final_score = float(checker(x, y))
            total_score += final_score
            episodes_ran += 1
            if abs(final_score - episode_max_reward) < 1e-6:
                num_solved += 1

    mean_score = total_score / float(max(episodes_ran, 1))
    success_rate = num_solved / float(max(episodes_ran, 1))
    return mean_score, success_rate


def evaluate_plan_policy(
    model: TinyRecursiveReasoningModel_ACTV1,
    dataset: Any,
    checker: Callable[[Any, Any], float],
    env_cfg: PlanEditEnvConfig,
    num_episodes: int = 100,
    inner_unroll_n: Optional[int] = None,
) -> float:
    """
    Backwards-compatible wrapper that only returns the strict success rate.
    """

    _, success_rate = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker,
        env_cfg=env_cfg,
        num_episodes=num_episodes,
        inner_unroll_n=inner_unroll_n,
    )
    return success_rate

