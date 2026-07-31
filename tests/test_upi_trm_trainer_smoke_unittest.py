"""
Tests for UPI-TRM trainer smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import math
import unittest
from unittest.mock import patch
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from rl.task_config import DummyTaskConfig
from rl.training_setup import DummyPuzzleDataset
from rl.upi_trm_trainer import (
    UPITrmTrainer,
    _clip_and_recenter_advantages,
    _mean_categorical_kl,
)


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


class TestUPITrmTrainerSmoke(unittest.TestCase):
    """Smoke tests for UPI-TRM trainer."""

    def test_upi_trm_trainer_smoke(self):
        """Test that UPI-TRM trainer runs without errors."""
        dataset = DummyPuzzleDataset(num_instances=8, seq_len=12, vocab_size=16)
        env_cfg = PlanEditEnvConfig(max_edits=4, gamma=0.99, reward_shaping=True, vocab_size=dataset.vocab_size)
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

        rl_cfg = RLConfig(
            batch_size=4,
            num_train_steps=5,
            rollout_episodes_per_step=1,
            max_edits=4,
        )

        model_cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=rl_cfg.batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

        for _ in range(3):
            metrics = trainer.train_step()
            self.assertIn("loss_value", metrics)
            self.assertIn("loss_policy", metrics)
            self.assertTrue(math.isfinite(metrics["loss_value"]))
            self.assertTrue(math.isfinite(metrics["loss_policy"]))

    def test_policy_update_uses_task_config_batch_mask(self):
        class TrackingTaskConfig(DummyTaskConfig):
            def __init__(self):
                super().__init__()
                self.batch_called = False
                self.last_current_state_shape = None

            def compute_batch_action_mask(self, inputs, vocab_size, stop_action_id, current_state=None):
                if current_state is None:
                    raise AssertionError("current_state should be provided for batch masking")
                self.batch_called = True
                if torch.is_tensor(current_state):
                    self.last_current_state_shape = tuple(current_state.shape)
                return super().compute_batch_action_mask(inputs, vocab_size, stop_action_id, current_state=current_state)

        dataset = DummyPuzzleDataset(num_instances=6, seq_len=8, vocab_size=12)
        env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True, vocab_size=dataset.vocab_size)
        task_config = TrackingTaskConfig()
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg, task_config=task_config)
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

        rl_cfg = RLConfig(
            batch_size=2,
            num_train_steps=4,
            rollout_episodes_per_step=2,
            max_edits=3,
        )
        model_cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=rl_cfg.batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))
        for _ in range(3):
            trainer.train_step()

        self.assertTrue(task_config.batch_called)
        self.assertIsNotNone(task_config.last_current_state_shape)

    def _make_trainer(
        self,
        *,
        episodic_latent=True,
        stop_mode="noop",
        task_config=None,
        theory_exact_mixture=False,
        enable_kl_trust_region=False,
        distill_mixture_policy=False,
    ):
        dataset = DummyPuzzleDataset(num_instances=6, seq_len=8, vocab_size=12)
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.99,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
            stop_action_mode=stop_mode,
        )
        env = PlanEditEnv(
            dataset=dataset,
            checker=dummy_checker,
            config=env_cfg,
            task_config=task_config,
        )
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)
        rl_cfg = RLConfig(
            batch_size=2,
            rollout_episodes_per_step=1,
            max_edits=3,
            episodic_latent=episodic_latent,
            theory_exact_mixture=theory_exact_mixture,
            enable_kl_trust_region=enable_kl_trust_region,
            distill_mixture_policy=distill_mixture_policy,
        )
        model = TinyRecursiveReasoningModel_ACTV1(
            _tiny_trm_cfg(
                dataset.seq_len,
                dataset.vocab_size,
                dataset.num_identifiers,
                rl_cfg.batch_size,
            )
        )
        return UPITrmTrainer(model, env, rl_cfg, torch.device("cpu")), dataset

    def test_training_mask_keeps_disabled_stop_out_of_support(self):
        trainer, _ = self._make_trainer(
            stop_mode="disabled",
            task_config=DummyTaskConfig(),
        )
        trainer.collect_episode()
        transitions = list(trainer.replay.storage)[:2]
        x_batch, y_batch, *_ = trainer._stack_batch(transitions)
        mask = trainer._compute_training_action_mask(x_batch, y_batch)
        self.assertFalse(mask[:, trainer.env.stop_action_id].any().item())

    def test_persistent_replay_retains_latents_and_behavior_probability(self):
        trainer, _ = self._make_trainer(episodic_latent=False)
        trainer.collect_episode()
        transition = trainer.replay.storage[0]
        self.assertIsNotNone(transition.latent)
        self.assertIsNotNone(transition.next_latent)
        self.assertIsNotNone(transition.behavior_log_prob)
        self.assertTrue(torch.isfinite(transition.behavior_log_prob))

        x_batch, _, x_next_batch, *_ = trainer._stack_batch([transition])
        self.assertIn("solution", x_batch)
        self.assertIn("solution", x_next_batch)
        torch.testing.assert_close(x_batch["solution"][0], transition.x["solution"])

    def test_exact_mixture_evaluation_uses_deployed_policy_callback(self):
        trainer, dataset = self._make_trainer(theory_exact_mixture=True)

        def consume_evaluation_rng(*args, **kwargs):
            torch.rand(3)
            return 0.0, 0.0, {}

        rng_before = torch.random.get_rng_state().clone()
        with patch(
            "rl.evaluator.evaluate_plan_policy_with_scores",
            side_effect=consume_evaluation_rng,
        ) as evaluate:
            metrics = trainer.evaluate_policy_metrics(
                trainer.env_config,
                dataset,
                dummy_checker,
            )

        self.assertEqual(metrics["eval_policy_mode"], "stochastic_exact_mixture")
        self.assertEqual(metrics["eval_seed"], trainer.rl_cfg.eval_seed)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertFalse(evaluate.call_args.kwargs["greedy"])
        callback = evaluate.call_args.kwargs["policy_dist_fn"]
        self.assertIs(callback.__self__, trainer)
        self.assertIs(callback.__func__, trainer._mixed_policy_dist.__func__)

    def test_exact_actor_is_insulated_until_post_value_snapshot_sync(self):
        trainer, _ = self._make_trainer(theory_exact_mixture=True)
        trainer.collect_episode()

        parameter_name, critic_parameter = next(
            (name, parameter)
            for name, parameter in trainer.model.named_parameters()
            if not name.startswith("edit_policy")
        )
        old_parameter = dict(trainer.policy_model_old.named_parameters())[parameter_name]
        old_before = old_parameter.detach().clone()
        critic_before = critic_parameter.detach().clone()
        original_sync = trainer._sync_exact_policy_snapshot_from_model
        observed_insulation = []

        def mutate_critic():
            with torch.no_grad():
                critic_parameter.add_(0.25)

        def inspect_then_sync():
            torch.testing.assert_close(old_parameter, old_before)
            torch.testing.assert_close(
                critic_parameter,
                critic_before + 0.25,
            )
            observed_insulation.append(True)
            original_sync()

        with patch.object(trainer.value_opt, "step", side_effect=mutate_critic), patch.object(
            trainer,
            "_sync_exact_policy_snapshot_from_model",
            side_effect=inspect_then_sync,
        ) as snapshot_sync:
            trainer.value_update()

        snapshot_sync.assert_called_once_with()
        self.assertEqual(observed_insulation, [True])
        torch.testing.assert_close(old_parameter, critic_parameter)

    def test_exact_snapshot_copies_nonpolicy_state_and_preserves_policy_heads(self):
        trainer, _ = self._make_trainer(theory_exact_mixture=True)
        for actor in (
            trainer.model,
            trainer.policy_model_old,
            trainer.policy_model_candidate,
        ):
            actor.register_buffer(
                "snapshot_probe",
                torch.zeros(2),
                persistent=True,
            )

        parameter_name, critic_parameter = next(
            (name, parameter)
            for name, parameter in trainer.model.named_parameters()
            if not name.startswith("edit_policy")
        )
        old_head_before = {
            name: value.clone()
            for name, value in trainer.policy_model_old.edit_policy.state_dict().items()
        }
        candidate_head_before = {
            name: value.clone()
            for name, value in trainer.policy_model_candidate.edit_policy.state_dict().items()
        }
        with torch.no_grad():
            critic_parameter.add_(0.5)
            trainer.model.snapshot_probe.copy_(torch.tensor([3.0, 7.0]))

        trainer._sync_exact_policy_snapshot_from_model()

        for actor in (trainer.policy_model_old, trainer.policy_model_candidate):
            actor_parameter = dict(actor.named_parameters())[parameter_name]
            torch.testing.assert_close(actor_parameter, critic_parameter)
            torch.testing.assert_close(actor.snapshot_probe, trainer.model.snapshot_probe)
        for name, value in trainer.policy_model_old.edit_policy.state_dict().items():
            torch.testing.assert_close(value, old_head_before[name])
        for name, value in trainer.policy_model_candidate.edit_policy.state_dict().items():
            torch.testing.assert_close(value, candidate_head_before[name])

    def test_exact_clamp_precedes_target_snapshot_and_policy_update(self):
        trainer, _ = self._make_trainer(theory_exact_mixture=True)
        trainer.rl_cfg.enable_contraction = True
        trainer.rl_cfg.opnorm_clamp_interval = 1

        parameter_name, critic_parameter = next(
            (name, parameter)
            for name, parameter in trainer.model.named_parameters()
            if not name.startswith("edit_policy")
        )
        events = []
        original_target_sync = trainer._soft_update_target
        original_actor_sync = trainer._sync_exact_policy_snapshot_from_model

        def clamp_then_mark(*_args, **_kwargs):
            with torch.no_grad():
                critic_parameter.add_(0.75)
            events.append("clamp")
            return {"probe": 0.5}

        def target_sync_after_clamp():
            self.assertEqual(events, ["clamp"])
            original_target_sync()
            events.append("target")

        def actor_sync_after_target():
            self.assertEqual(events, ["clamp", "target"])
            original_actor_sync()
            events.append("actors")

        def policy_after_snapshot():
            self.assertEqual(events, ["clamp", "target", "actors"])
            for actor in (
                trainer.policy_model_old,
                trainer.policy_model_candidate,
            ):
                actor_parameter = dict(actor.named_parameters())[parameter_name]
                torch.testing.assert_close(actor_parameter, critic_parameter)
            events.append("policy")
            return {"loss_policy": 0.0}

        with patch(
            "rl.upi_trm_trainer.apply_opnorm_clamp_periodically",
            side_effect=clamp_then_mark,
        ) as clamp, patch.object(
            trainer,
            "_soft_update_target",
            side_effect=target_sync_after_clamp,
        ), patch.object(
            trainer,
            "_sync_exact_policy_snapshot_from_model",
            side_effect=actor_sync_after_target,
        ), patch.object(
            trainer,
            "policy_update",
            side_effect=policy_after_snapshot,
        ):
            trainer.train_step()

        clamp.assert_called_once_with(
            trainer.model.inner,
            per_layer_max=trainer.rl_cfg.opnorm_clamp_max_norm,
            num_power_iters=trainer.rl_cfg.opnorm_clamp_num_power_iters,
            restrict_to_reasoning_layers=True,
        )
        self.assertEqual(events, ["clamp", "target", "actors", "policy"])

    def test_advantage_clipping_preserves_statewise_centering(self):
        advantages = torch.tensor([[20.0, -180.0, 999.0]])
        policy_probs = torch.tensor([[0.9, 0.1, 0.0]])
        action_mask = torch.tensor([[True, True, False]])

        clipped = _clip_and_recenter_advantages(
            advantages,
            policy_probs,
            action_mask,
            clip_value=10.0,
        )

        torch.testing.assert_close(
            (policy_probs * clipped).sum(dim=-1),
            torch.zeros(1),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(clipped[0, 2].item(), 0.0)

    def test_masked_categorical_kl_is_finite_with_finite_gradients(self):
        reference_probs = torch.tensor([[1.0, 0.0]])
        reference_log_probs = reference_probs.clamp_min(1e-8).log()
        candidate_logits = torch.tensor([[0.0, float("-inf")]], requires_grad=True)
        candidate_log_probs = candidate_logits.log_softmax(dim=-1)

        kl = _mean_categorical_kl(
            reference_probs,
            reference_log_probs,
            candidate_log_probs,
        )
        kl.backward()

        self.assertTrue(torch.isfinite(kl).item())
        candidate_grad = candidate_logits.grad
        assert candidate_grad is not None
        self.assertTrue(torch.isfinite(candidate_grad).all().item())
        self.assertEqual(candidate_grad[0, 1].item(), 0.0)

    def test_categorical_kl_preserves_true_support_mismatch(self):
        reference_probs = torch.tensor([[0.5, 0.5]])
        reference_log_probs = reference_probs.log()
        candidate_log_probs = torch.tensor([[0.0, float("-inf")]])

        kl = _mean_categorical_kl(
            reference_probs,
            reference_log_probs,
            candidate_log_probs,
        )

        self.assertTrue(torch.isinf(kl).item())

    def test_policy_trust_region_is_finite_with_active_action_mask(self):
        trainer, _ = self._make_trainer(
            stop_mode="disabled",
            task_config=DummyTaskConfig(),
            enable_kl_trust_region=True,
        )
        trainer.collect_episode()

        transitions = list(trainer.replay.storage)[: trainer.rl_cfg.batch_size]
        x_batch, y_batch, *_ = trainer._stack_batch(transitions)
        action_mask = trainer._compute_training_action_mask(x_batch, y_batch)
        assert action_mask is not None
        self.assertFalse(action_mask[:, trainer.env.stop_action_id].any().item())

        result = trainer.policy_update()

        self.assertTrue(math.isfinite(result["loss_policy"]))
        self.assertTrue(math.isfinite(result["policy_kl"]))
        candidate_grads = [
            parameter.grad
            for parameter in trainer._policy_params
            if parameter.grad is not None
        ]
        self.assertTrue(candidate_grads)
        self.assertTrue(all(torch.isfinite(grad).all().item() for grad in candidate_grads))

    def test_distillation_kl_is_finite_with_active_action_mask(self):
        trainer, _ = self._make_trainer(
            stop_mode="disabled",
            task_config=DummyTaskConfig(),
            enable_kl_trust_region=True,
            distill_mixture_policy=True,
        )
        trainer.collect_episode()
        distill_optimizer = trainer.old_policy_distill_opt
        assert distill_optimizer is not None

        observed_finite_gradients = []
        original_step = distill_optimizer.step

        def assert_finite_distillation_gradients():
            gradients = [
                parameter.grad
                for parameter in trainer._old_policy_params
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all().item() for grad in gradients))
            observed_finite_gradients.append(True)
            return original_step()

        with patch.object(
            distill_optimizer,
            "step",
            side_effect=assert_finite_distillation_gradients,
        ):
            result = trainer.policy_update()

        self.assertTrue(math.isfinite(result["loss_policy"]))
        self.assertTrue(math.isfinite(result["policy_kl"]))
        self.assertEqual(observed_finite_gradients, [True])
        self.assertTrue(
            all(
                torch.isfinite(parameter).all().item()
                for parameter in trainer._old_policy_params
            )
        )

    def test_model_sync_copies_persistent_nonpolicy_buffers(self):
        trainer, _ = self._make_trainer()
        for model in (
            trainer.model,
            trainer.policy_model_candidate,
            trainer.target_model,
        ):
            model.register_buffer("sync_probe", torch.zeros(2), persistent=True)

        trainer.model.sync_probe.copy_(torch.tensor([3.0, 7.0]))
        trainer._sync_candidate_backbone_from_model()
        trainer._soft_update_target()

        torch.testing.assert_close(
            trainer.policy_model_candidate.sync_probe,
            trainer.model.sync_probe,
        )
        torch.testing.assert_close(trainer.target_model.sync_probe, trainer.model.sync_probe)

    def test_sparse_optimizer_steps_before_later_model_forwards(self):
        trainer, _ = self._make_trainer()
        trainer.collect_episode()

        class FakeSparseOptimizer:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

            def zero_grad(self):
                pass

        optimizer = FakeSparseOptimizer()
        trainer.set_puzzle_embedding_optimizer(optimizer)

        def assert_already_stepped(*_args, **_kwargs):
            self.assertEqual(optimizer.steps, 1)

        with patch.object(
            trainer,
            "_maybe_run_value_debug_checks",
            side_effect=assert_already_stepped,
        ):
            trainer.value_update()


if __name__ == "__main__":
    unittest.main()
