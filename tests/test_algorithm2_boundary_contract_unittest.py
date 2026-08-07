"""Boundary tests for the executable Algorithm 2 contract."""

import math
import unittest
from unittest.mock import patch

import torch

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from rl.task_config import DummyTaskConfig
from rl.training_setup import DummyPuzzleDataset
from rl.upi_trm_trainer import UPITrmTrainer


_SEQ_LEN = 2
_VOCAB_SIZE = 3
_HIDDEN_SIZE = 8


def _num_actions() -> int:
    return _SEQ_LEN * _VOCAB_SIZE + 1


def _tiny_model_config(num_identifiers: int) -> dict[str, object]:
    return {
        "batch_size": 1,
        "seq_len": _SEQ_LEN,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": num_identifiers,
        "vocab_size": _VOCAB_SIZE,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": _HIDDEN_SIZE,
        "expansion": 2.0,
        "num_heads": 1,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
        "rl_enable_value_head": True,
        "rl_value_hidden_dim": 8,
        "rl_enable_contraction": False,
        "rl_enable_policy_head": True,
        "rl_num_actions": _num_actions(),
        "rl_latent_projection_mode": "disabled",
        "rl_latent_ball_radius": None,
    }


def _make_trainer(
    *,
    persistent: bool = False,
    inner_unroll_n: int = 1,
    k_steps: int = 1,
    max_edits: int = 1,
    target_tau: float = 0.5,
) -> tuple[UPITrmTrainer, DummyPuzzleDataset]:
    torch.manual_seed(0)
    dataset = DummyPuzzleDataset(
        num_instances=2,
        seq_len=_SEQ_LEN,
        vocab_size=_VOCAB_SIZE,
    )
    for sample in dataset.samples:
        sample["inputs"] = torch.ones(_SEQ_LEN, dtype=torch.long)
        sample["initial_plan"] = torch.zeros(_SEQ_LEN, dtype=torch.long)
        sample["solution"] = torch.ones(_SEQ_LEN, dtype=torch.long)

    env_config = PlanEditEnvConfig(
        max_edits=max_edits,
        gamma=0.9,
        reward_shaping=True,
        vocab_size=_VOCAB_SIZE,
        solved_threshold=None,
        stop_action_mode="terminal",
        C_max=1.0,
        fail_terminal_reward=-1.0,
    )
    env = PlanEditEnv(
        dataset=dataset,
        checker=dummy_checker,
        config=env_config,
        task_config=DummyTaskConfig(),
    )
    env.set_stop_action_id(_num_actions() - 1)

    rl_config = RLConfig(
        gamma=0.9,
        K=k_steps,
        inner_unroll_n=inner_unroll_n,
        max_edits=max_edits,
        batch_size=1,
        rollout_episodes_per_step=1,
        num_train_steps=1,
        episodic_latent=not persistent,
        stop_action_mode="terminal",
        solved_threshold=None,
        C_max=1.0,
        fail_terminal_reward=-1.0,
        target_ema_tau=target_tau,
        exact_k_step_targets=True,
        exact_baseline_summation=True,
        batch_centered_advantage=False,
        theory_exact_mixture=True,
        distill_mixture_policy=False,
        training_protocol="fixed_base_exact",
        policy_epsilon=0.0,
        value_target_clip=None,
        enable_contraction=False,
        opnorm_clamp_interval=0,
        latent_projection_mode="disabled",
        latent_ball_radius=None,
        lr_schedule="constant",
        entropy_coef=0.01,
        use_tqdm=False,
    )
    model = TinyRecursiveReasoningModel_ACTV1(
        _tiny_model_config(dataset.num_identifiers)
    )
    trainer = UPITrmTrainer(model, env, rl_config, torch.device("cpu"))
    trainer.set_checker_fn(dummy_checker)
    return trainer, dataset


def _carry(value: float) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    tensor = torch.full((1, _SEQ_LEN, _HIDDEN_SIZE), value)
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=tensor,
        z_L=tensor.clone(),
    )


class TestAlgorithm2BoundaryContract(unittest.TestCase):
    def test_zero_depth_fixed_base_runs_all_update_phases(self) -> None:
        trainer, _ = _make_trainer(inner_unroll_n=0)

        metrics = trainer.train_step()

        self.assertEqual(len(trainer.replay), 1)
        self.assertEqual(metrics["value_optimizer_step"], 1.0)
        self.assertEqual(metrics["policy_optimizer_step"], 1.0)
        self.assertEqual(metrics["optimization_performed"], 1.0)
        self.assertTrue(math.isfinite(metrics["loss_value"]))
        self.assertTrue(math.isfinite(metrics["loss_policy"]))

    def test_probability_mixture_endpoints_share_persistent_carry(self) -> None:
        trainer, _ = _make_trainer(persistent=True, max_edits=2)
        input_carry = _carry(1.0)
        base_output_carry = _carry(2.0)
        candidate_output_carry = _carry(3.0)
        base_probs = torch.zeros(1, _num_actions())
        candidate_probs = torch.zeros(1, _num_actions())
        base_probs[0, 0] = 1.0
        candidate_probs[0, 1] = 1.0
        x_batch = {
            "inputs": torch.ones(1, _SEQ_LEN, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
            "remaining_edits": torch.tensor([2]),
        }
        y_batch = torch.zeros(1, _SEQ_LEN, dtype=torch.long)

        for alpha, expected_probs in ((0.0, base_probs), (1.0, candidate_probs)):
            calls = []

            def base_policy_dist(x, y, *, n, action_mask, z):
                calls.append(("base", n, z))
                return (
                    torch.distributions.Categorical(probs=base_probs),
                    base_output_carry,
                )

            def candidate_policy_dist(x, y, *, n, action_mask, z):
                calls.append(("candidate", n, z))
                return (
                    torch.distributions.Categorical(probs=candidate_probs),
                    candidate_output_carry,
                )

            trainer.rl_cfg.mixture_alpha = alpha
            with patch.object(
                trainer.policy_model_old,
                "policy_dist",
                side_effect=base_policy_dist,
            ), patch.object(
                trainer.policy_model_candidate,
                "policy_dist",
                side_effect=candidate_policy_dist,
            ):
                mixture, output_carry = trainer._mixed_policy_dist(
                    x_batch,
                    y_batch,
                    n=2,
                    z=input_carry,
                )

            torch.testing.assert_close(
                mixture.probs,
                expected_probs,
                atol=0.0,
                rtol=0.0,
            )
            self.assertEqual(calls[0][:2], ("base", 2))
            self.assertIs(calls[0][2], input_carry)
            self.assertEqual(calls[1][:2], ("candidate", 0))
            self.assertIs(calls[1][2], base_output_carry)
            self.assertIs(output_carry, base_output_carry)

    def test_target_retention_coefficient_endpoints(self) -> None:
        trainer, _ = _make_trainer()
        assert trainer.model.value_head is not None
        assert trainer.target_model.value_head is not None

        with torch.no_grad():
            for parameter in trainer.model.value_head.parameters():
                parameter.fill_(3.0)
            for parameter in trainer.target_model.value_head.parameters():
                parameter.fill_(-2.0)
        trainer.rl_cfg.target_ema_tau = 0.0
        trainer._soft_update_target()
        for online, target in zip(
            trainer.model.value_head.parameters(),
            trainer.target_model.value_head.parameters(),
        ):
            torch.testing.assert_close(target, online, atol=0.0, rtol=0.0)

        with torch.no_grad():
            for parameter in trainer.model.value_head.parameters():
                parameter.fill_(-4.0)
            for parameter in trainer.target_model.value_head.parameters():
                parameter.fill_(7.0)
        retained = [
            parameter.detach().clone()
            for parameter in trainer.target_model.value_head.parameters()
        ]
        trainer.rl_cfg.target_ema_tau = 1.0
        trainer._soft_update_target()
        for target, expected in zip(
            trainer.target_model.value_head.parameters(), retained
        ):
            torch.testing.assert_close(target, expected, atol=0.0, rtol=0.0)

    def test_fractional_target_update_changes_only_value_head(self) -> None:
        trainer, _ = _make_trainer(target_tau=0.99)
        assert trainer.model.value_head is not None
        assert trainer.target_model.value_head is not None

        target_before = {
            name: value.detach().clone()
            for name, value in trainer.target_model.state_dict().items()
        }
        with torch.no_grad():
            for parameter in trainer.model.value_head.parameters():
                parameter.add_(3.0)

            recurrent_parameter = next(
                parameter
                for name, parameter in trainer.model.named_parameters()
                if name.startswith("inner.")
                and not name.startswith(("inner.lm_head.", "inner.q_head."))
            )
            recurrent_parameter.add_(5.0)
            recurrent_buffer = next(
                buffer
                for name, buffer in trainer.model.named_buffers()
                if name.startswith("inner.")
                and not name.startswith(("inner.lm_head.", "inner.q_head."))
                and torch.is_floating_point(buffer)
            )
            recurrent_buffer.add_(7.0)
            edit_parameter = next(trainer.model.edit_policy.parameters())
            edit_parameter.add_(11.0)

        source_value_parameters = dict(
            trainer.model.value_head.named_parameters()
        )
        expected_value_parameters = {}
        for name, source in source_value_parameters.items():
            expected = target_before[f"value_head.{name}"].clone()
            expected.mul_(0.99).add_(source.detach(), alpha=0.01)
            expected_value_parameters[name] = expected

        trainer._soft_update_target()

        for name, target in trainer.target_model.value_head.named_parameters():
            torch.testing.assert_close(
                target,
                expected_value_parameters[name],
                atol=0.0,
                rtol=0.0,
            )
        for name, target in trainer.target_model.state_dict().items():
            if name.startswith("value_head."):
                continue
            self.assertTrue(
                torch.equal(target, target_before[name]),
                msg=f"fixed-base target update mutated {name}",
            )

    def test_exact_k_larger_than_clock_stops_without_terminal_bootstrap(self) -> None:
        trainer, _ = _make_trainer(
            persistent=True,
            k_steps=3,
            max_edits=1,
        )
        edit_probs = torch.zeros(1, _num_actions())
        edit_probs[0, 0] = 1.0

        def edit_policy_dist(x, y, *, n, action_mask, z):
            self.assertIsNotNone(z)
            return torch.distributions.Categorical(probs=edit_probs), z

        with patch.object(
            trainer.policy_model_old,
            "policy_dist",
            side_effect=edit_policy_dist,
        ):
            steps = trainer.collect_episode()

        self.assertEqual(steps, 1)
        self.assertEqual(len(trainer.replay), 1)
        transition = trainer.replay.storage[0]
        self.assertTrue(bool(transition.done.item()))
        self.assertEqual(transition.terminal_reason, "budget")
        self.assertIsNotNone(transition.latent)
        self.assertIsNone(transition.next_latent)

        sample = trainer._sample_k_step_batch(1)
        rewards_k = sample[4]
        dones_k = sample[5]
        steps_taken = sample[6]
        end_latents = sample[8]
        self.assertEqual(steps_taken.item(), 1)
        self.assertTrue(bool(dones_k[0, 0].item()))
        self.assertEqual(end_latents, [None])

        with patch.object(
            trainer.target_model,
            "used_value",
            side_effect=AssertionError("terminal endpoint was bootstrapped"),
        ) as target_value:
            result = trainer.value_update()

        target_value.assert_not_called()
        self.assertEqual(result["value_optimizer_step"], 1.0)
        self.assertAlmostEqual(
            result["target_mean"],
            float(rewards_k[0, 0].item()),
            places=6,
        )

    def test_terminal_stop_stores_one_persistent_record_and_breaks(self) -> None:
        trainer, _ = _make_trainer(persistent=True, max_edits=4)
        stop_id = trainer.env.stop_action_id
        assert stop_id is not None
        stop_probs = torch.zeros(1, _num_actions())
        stop_probs[0, stop_id] = 1.0

        def stop_policy_dist(x, y, *, n, action_mask, z):
            self.assertIsNotNone(z)
            return torch.distributions.Categorical(probs=stop_probs), z

        with patch.object(
            trainer.policy_model_old,
            "policy_dist",
            side_effect=stop_policy_dist,
        ) as policy_dist, patch.object(
            trainer.env,
            "step",
            wraps=trainer.env.step,
        ) as env_step:
            steps = trainer.collect_episode()

        self.assertEqual(steps, 1)
        policy_dist.assert_called_once()
        env_step.assert_called_once_with(stop_id)
        self.assertEqual(len(trainer.replay), 1)
        transition = trainer.replay.storage[0]
        self.assertTrue(bool(transition.done.item()))
        self.assertEqual(transition.terminal_reason, "stop")
        self.assertIsNotNone(transition.latent)
        self.assertIsNone(transition.next_latent)
        self.assertIsNone(trainer._active_episode)


if __name__ == "__main__":
    unittest.main()
