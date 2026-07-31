"""
Tests for RL K-step value update trainer - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import math
import unittest
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.replay import Transition
from rl.sudoku_checkers import dummy_checker
from rl.training_setup import DummyPuzzleDataset
from rl.upi_trm_trainer import UPITrmTrainer


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


class TestRLKStepValueUpdateTrainer(unittest.TestCase):
    """Tests for RL K-step value update trainer."""

    def test_k_step_value_update_runs(self):
        """Test that K-step value update runs without errors."""
        dataset = DummyPuzzleDataset(num_instances=8, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(max_edits=4, gamma=0.9, reward_shaping=True, vocab_size=dataset.vocab_size)
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

        rl_cfg = RLConfig(
            batch_size=4,
            num_train_steps=1,
            rollout_episodes_per_step=1,
            max_edits=4,
            K=3,
            gamma=env_cfg.gamma,
        )

        model_cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=rl_cfg.batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

        for _ in range(5):
            trainer.collect_episode()

        loss_result = trainer.value_update()

        self.assertIsInstance(loss_result, dict)
        self.assertIn("loss_value", loss_result)
        self.assertTrue(math.isfinite(loss_result["loss_value"]))

    def test_exact_k_step_sampler_rejects_truncated_nonterminal_segment(self):
        dataset = DummyPuzzleDataset(num_instances=2, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(
            max_edits=4,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
        env.set_stop_action_id(_num_actions(dataset.seq_len, dataset.vocab_size) - 1)
        rl_cfg = RLConfig(batch_size=1, K=3, gamma=env_cfg.gamma, exact_k_step_targets=True)
        model = TinyRecursiveReasoningModel_ACTV1(
            _tiny_trm_cfg(
                dataset.seq_len,
                dataset.vocab_size,
                dataset.num_identifiers,
                rl_cfg.batch_size,
            )
        )
        trainer = UPITrmTrainer(model, env, rl_cfg, torch.device("cpu"))

        x = {
            "inputs": torch.zeros(dataset.seq_len, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0),
        }
        y = torch.zeros(dataset.seq_len, dtype=torch.long)
        for timestep in range(2):
            trainer.replay.add(
                Transition(
                    x=x,
                    y=y,
                    action=torch.tensor(0),
                    reward=torch.tensor([0.0]),
                    x_next=x,
                    y_next=y,
                    done=torch.tensor([False]),
                    episode_id=0,
                    timestep=timestep,
                )
            )

        self.assertFalse(trainer._has_complete_k_step_segment(0, 3))
        with self.assertRaisesRegex(RuntimeError, "No replay segment"):
            trainer._sample_k_step_batch(1)


if __name__ == "__main__":
    unittest.main()
