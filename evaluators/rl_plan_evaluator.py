"""
Backwards compatibility shim for evaluators.rl_plan_evaluator.

The actual implementation has moved to rl.evaluator to avoid circular
dependency issues with Buck2.
"""
from rl.evaluator import evaluate_plan_policy, evaluate_plan_policy_with_scores

__all__ = ["evaluate_plan_policy", "evaluate_plan_policy_with_scores"]
