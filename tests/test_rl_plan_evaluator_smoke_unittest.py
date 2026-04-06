"""
Tests for RL plan evaluator smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import math
import unittest
import torch

from rl.evaluator import evaluate_plan_policy, evaluate_plan_policy_with_scores
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.envs.plan_edit_env import PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from rl.training_setup import DummyPuzzleDataset


def _num_actions(seq_len: int, vocab_size: int) -> int:
    return seq_len * vocab_size + 1


def _tiny_trm_cfg(seq_len: int, vocab_size: int, num_identifiers: int, batch_size: int):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=max(num_identifiers, batch_size),
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,
        rl_enable_policy_head=True,
        rl_num_actions=_num_actions(seq_len, vocab_size),
    )


class TestRLPlanEvaluatorSmoke(unittest.TestCase):
    """Smoke tests for RL plan evaluator."""

    def test_evaluate_plan_policy_smoke(self):
        """Test that evaluate_plan_policy runs without errors."""
        torch.manual_seed(0)

        dataset = DummyPuzzleDataset(num_instances=8, seq_len=12, vocab_size=16)
        env_cfg = PlanEditEnvConfig(max_edits=4, gamma=0.99, reward_shaping=True, vocab_size=dataset.vocab_size)

        batch_size = 4
        cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(cfg)

        score = evaluate_plan_policy(
            model=model,
            dataset=dataset,
            checker=dummy_checker,
            env_cfg=env_cfg,
            num_episodes=10,
            inner_unroll_n=2,
        )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_plan_policy_with_scores_smoke(self):
        """Test that evaluate_plan_policy_with_scores runs without errors."""
        torch.manual_seed(1)

        dataset = DummyPuzzleDataset(num_instances=6, seq_len=10, vocab_size=8)
        env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.95, reward_shaping=True, vocab_size=dataset.vocab_size)

        batch_size = 4
        cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(cfg)

        mean_score, success_rate, details = evaluate_plan_policy_with_scores(
            model=model,
            dataset=dataset,
            checker=dummy_checker,
            env_cfg=env_cfg,
            num_episodes=12,
            inner_unroll_n=2,
        )

        self.assertIsInstance(mean_score, float)
        self.assertIsInstance(success_rate, float)
        self.assertIsInstance(details, dict)
        self.assertTrue(math.isfinite(mean_score))
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
