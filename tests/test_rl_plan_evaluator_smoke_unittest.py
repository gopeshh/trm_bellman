"""
Tests for RL plan evaluator smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import math
import random
import unittest

import numpy as np
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

        dataset = DummyPuzzleDataset(num_instances=10, seq_len=12, vocab_size=16)
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

        dataset = DummyPuzzleDataset(num_instances=12, seq_len=10, vocab_size=8)
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

    def test_evaluation_refuses_to_cycle_pool_by_default(self):
        dataset = DummyPuzzleDataset(num_instances=2, seq_len=4, vocab_size=4)
        env_cfg = PlanEditEnvConfig(
            max_edits=1,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )
        with self.assertRaisesRegex(ValueError, "refusing to cycle"):
            evaluate_plan_policy_with_scores(
                model=object(),
                dataset=dataset,
                checker=dummy_checker,
                env_cfg=env_cfg,
                num_episodes=3,
            )

    def test_evaluation_isolates_rng_modes_and_emits_deterministic_rows(self):
        torch.manual_seed(11)
        dataset = DummyPuzzleDataset(num_instances=3, seq_len=4, vocab_size=4)
        env_cfg = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )
        cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=2,
        )
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        additional_model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.train()
        additional_model.eval()

        def policy_dist(*args, **kwargs):
            random.random()
            np.random.random()
            return model.policy_dist(*args, **kwargs)

        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()

        first = evaluate_plan_policy_with_scores(
            model=model,
            dataset=dataset,
            checker=dummy_checker,
            env_cfg=env_cfg,
            num_episodes=3,
            inner_unroll_n=1,
            greedy=False,
            policy_dist_fn=policy_dist,
            evaluation_seed=1729,
            collect_per_instance=True,
            additional_models=(additional_model,),
            record_local_seeding=True,
        )

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        self.assertTrue(model.training)
        self.assertFalse(additional_model.training)

        second = evaluate_plan_policy_with_scores(
            model=model,
            dataset=dataset,
            checker=dummy_checker,
            env_cfg=env_cfg,
            num_episodes=3,
            inner_unroll_n=1,
            greedy=False,
            policy_dist_fn=policy_dist,
            evaluation_seed=1729,
            collect_per_instance=True,
            additional_models=(additional_model,),
            record_local_seeding=True,
        )
        self.assertEqual(first[2]["per_instance"], second[2]["per_instance"])
        rows = first[2]["per_instance"]
        self.assertEqual([row["record_index"] for row in rows], [0, 1, 2])
        self.assertEqual(len({row["record_sha256"] for row in rows}), 3)
        self.assertTrue(all(row["evaluation_seed"] is not None for row in rows))

        class ReversedDataset:
            def __init__(self, source):
                self.source = source
                self.vocab_size = source.vocab_size

            def __len__(self):
                return len(self.source)

            def __getitem__(self, index):
                return self.source[len(self.source) - index - 1]

        permuted = evaluate_plan_policy_with_scores(
            model=model,
            dataset=ReversedDataset(dataset),
            checker=dummy_checker,
            env_cfg=env_cfg,
            num_episodes=3,
            inner_unroll_n=1,
            greedy=False,
            policy_dist_fn=policy_dist,
            evaluation_seed=1729,
            collect_per_instance=True,
            additional_models=(additional_model,),
            record_local_seeding=True,
        )[2]["per_instance"]
        original_by_hash = {
            row["record_sha256"]: {
                key: value for key, value in row.items() if key != "record_index"
            }
            for row in rows
        }
        permuted_by_hash = {
            row["record_sha256"]: {
                key: value for key, value in row.items() if key != "record_index"
            }
            for row in permuted
        }
        self.assertEqual(original_by_hash, permuted_by_hash)

    def test_evaluation_restores_rng_and_modes_when_policy_raises(self):
        torch.manual_seed(12)
        dataset = DummyPuzzleDataset(num_instances=1, seq_len=4, vocab_size=4)
        env_cfg = PlanEditEnvConfig(
            max_edits=1,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )
        cfg = _tiny_trm_cfg(4, 4, 1, 1)
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.train()

        random.seed(201)
        np.random.seed(202)
        torch.manual_seed(203)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()

        def fail_policy(*args, **kwargs):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("deliberate evaluation failure")

        with self.assertRaisesRegex(RuntimeError, "deliberate evaluation failure"):
            evaluate_plan_policy_with_scores(
                model=model,
                dataset=dataset,
                checker=dummy_checker,
                env_cfg=env_cfg,
                num_episodes=1,
                policy_dist_fn=fail_policy,
                evaluation_seed=1729,
            )

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
