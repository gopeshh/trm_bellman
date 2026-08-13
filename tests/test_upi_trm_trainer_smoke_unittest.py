"""
Tests for UPI-TRM trainer smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

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
from rl.upi_trm_trainer import (
    UPITrmTrainer,
    _clip_and_recenter_advantages,
    _mean_categorical_kl,
    _validated_statewise_centering_defect,
)
from utils.dataset_provenance import sample_sha256
from utils.evaluation_artifacts import record_local_evaluation_seed


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
        rl_latent_projection_mode="enabled",
        rl_latent_ball_radius=10.0,
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
        exact_baseline_summation=False,
        exact_k_step_targets=False,
        training_protocol="legacy",
        enable_contraction=True,
        opnorm_clamp_interval=100,
        capture_preinterpolation_policy_pair=False,
        evaluation_policy_mode="configured",
        policy_epsilon=0.0,
        gamma=0.99,
        value_target_clip=20.0,
    ):
        dataset = DummyPuzzleDataset(num_instances=6, seq_len=8, vocab_size=12)
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=gamma,
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
            stop_action_mode=stop_mode,
            theory_exact_mixture=theory_exact_mixture,
            enable_kl_trust_region=enable_kl_trust_region,
            distill_mixture_policy=distill_mixture_policy,
            exact_baseline_summation=exact_baseline_summation,
            exact_k_step_targets=exact_k_step_targets,
            training_protocol=training_protocol,
            enable_contraction=enable_contraction,
            opnorm_clamp_interval=opnorm_clamp_interval,
            capture_preinterpolation_policy_pair=(
                capture_preinterpolation_policy_pair
            ),
            evaluation_policy_mode=evaluation_policy_mode,
            policy_epsilon=policy_epsilon,
            gamma=gamma,
            value_target_clip=value_target_clip,
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
        transitions = list(trainer.replay.storage)
        self.assertTrue(transitions)
        self.assertTrue(bool(transitions[-1].done.item()))
        for transition in transitions:
            self.assertIsNotNone(transition.latent)
            if bool(transition.done.item()):
                self.assertIsNone(transition.next_latent)
            else:
                self.assertIsNotNone(transition.next_latent)
            self.assertIsNotNone(transition.behavior_log_prob)
            self.assertTrue(torch.isfinite(transition.behavior_log_prob))

        transition = transitions[0]
        x_batch, _, x_next_batch, *_ = trainer._stack_batch([transition])
        self.assertIn("solution", x_batch)
        self.assertIn("solution", x_next_batch)
        torch.testing.assert_close(x_batch["solution"][0], transition.x["solution"])

    def test_gamma_zero_trainer_updates_without_bootstrapping(self):
        trainer, _ = self._make_trainer(gamma=0.0)
        trainer.collect_episode()
        trainer.collect_episode()

        result = trainer.value_update()

        self.assertEqual(trainer.rl_cfg.gamma, 0.0)
        self.assertTrue(math.isfinite(result["loss_value"]))

    def test_persistent_exact_baseline_policy_update_runs(self):
        trainer, _ = self._make_trainer(
            episodic_latent=False,
            stop_mode="terminal",
            theory_exact_mixture=True,
            exact_baseline_summation=True,
            exact_k_step_targets=True,
            training_protocol="fixed_base_exact",
            enable_contraction=False,
            opnorm_clamp_interval=0,
            value_target_clip=None,
        )
        trainer.set_checker_fn(dummy_checker)
        while len(trainer.replay) < trainer.rl_cfg.batch_size:
            trainer.collect_episode()

        result = trainer.policy_update()

        self.assertIn("loss_policy", result)
        self.assertTrue(math.isfinite(result["loss_policy"]))
        self.assertLessEqual(
            result["exact_centering_defect_max"],
            result["exact_centering_tolerance"],
        )
        self.assertEqual(result["exact_centering_batches_total"], 1.0)
        state = trainer.exact_centering_checkpoint_state()
        self.assertEqual(state["batch_count"], 1)
        self.assertEqual(
            state["maximum_observed"],
            result["exact_centering_defect_max_observed"],
        )

    def test_exact_mixture_evaluation_uses_deployed_policy_callback(self):
        trainer, dataset = self._make_fixed_base_trainer()

        with patch(
            "rl.evaluator.evaluate_plan_policy_with_scores",
            return_value=(0.0, 0.0, {}),
        ) as evaluate:
            metrics = trainer.evaluate_policy_metrics(
                trainer.env_config,
                dataset,
                dummy_checker,
            )

        self.assertEqual(metrics["eval_policy_mode"], "stochastic_exact_mixture")
        self.assertEqual(metrics["eval_seed"], trainer.rl_cfg.eval_seed)
        self.assertFalse(evaluate.call_args.kwargs["greedy"])
        self.assertTrue(evaluate.call_args.kwargs["record_local_seeding"])
        self.assertEqual(
            evaluate.call_args.kwargs["additional_models"],
            (trainer.policy_model_candidate,),
        )
        callback = evaluate.call_args.kwargs["policy_dist_fn"]
        self.assertIs(callback.__self__, trainer)
        self.assertIs(callback.__func__, trainer._mixed_policy_dist.__func__)

    def test_preinterpolation_evaluator_uses_captured_pair_only(self):
        trainer, dataset = self._make_trainer(
            capture_preinterpolation_policy_pair=True,
            evaluation_policy_mode="preinterpolation_exact_mixture",
        )
        self.assertIsNotNone(trainer.preinterpolation_policy_base)
        self.assertIsNotNone(trainer.preinterpolation_policy_candidate)

        with torch.no_grad():
            candidate_parameter = next(
                trainer.policy_model_candidate.edit_policy.parameters()
            )
            candidate_parameter.add_(0.25)
        trainer._capture_current_preinterpolation_pair()
        captured_candidate = next(
            trainer.preinterpolation_policy_candidate.edit_policy.parameters()
        ).detach().clone()
        trainer._sync_policy_old_towards_candidate()
        trainer._sync_candidate_policy_from_old()
        current_candidate = next(
            trainer.policy_model_candidate.edit_policy.parameters()
        ).detach()
        self.assertFalse(torch.equal(captured_candidate, current_candidate))

        with patch(
            "rl.evaluator.evaluate_plan_policy_with_scores",
            return_value=(0.0, 0.0, {}),
        ) as evaluate:
            metrics = trainer.evaluate_policy_metrics(
                trainer.env_config,
                dataset,
                dummy_checker,
            )

        kwargs = evaluate.call_args.kwargs
        self.assertIs(kwargs["model"], trainer.preinterpolation_policy_base)
        self.assertEqual(
            kwargs["additional_models"],
            (trainer.preinterpolation_policy_candidate,),
        )
        callback = kwargs["policy_dist_fn"]
        self.assertIs(callback.__self__, trainer)
        self.assertIs(
            callback.__func__,
            trainer._preinterpolation_mixed_policy_dist.__func__,
        )
        self.assertFalse(kwargs["greedy"])
        self.assertEqual(
            metrics["eval_policy_mode"],
            "stochastic_preinterpolation_exact_mixture",
        )

    def test_bridge_control_samples_deployed_actor_with_record_local_seed(self):
        trainer, dataset = self._make_trainer(
            capture_preinterpolation_policy_pair=True,
            evaluation_policy_mode="stochastic_deployed",
        )
        with patch(
            "rl.evaluator.evaluate_plan_policy_with_scores",
            return_value=(0.0, 0.0, {}),
        ) as evaluate:
            metrics = trainer.evaluate_policy_metrics(
                trainer.env_config,
                dataset,
                dummy_checker,
            )

        kwargs = evaluate.call_args.kwargs
        self.assertIs(kwargs["model"], trainer.policy_model_old)
        self.assertIsNone(kwargs["policy_dist_fn"])
        self.assertEqual(kwargs["additional_models"], ())
        self.assertFalse(kwargs["greedy"])
        self.assertTrue(kwargs["record_local_seeding"])
        self.assertEqual(
            metrics["eval_policy_mode"],
            "stochastic_deployed_policy",
        )

    def test_preinterpolation_evaluator_requires_pair_capture(self):
        with self.assertRaisesRegex(ValueError, "requires.*capture"):
            self._make_trainer(
                evaluation_policy_mode="preinterpolation_exact_mixture",
            )

        with self.assertRaisesRegex(ValueError, "policy_epsilon=0"):
            self._make_trainer(
                capture_preinterpolation_policy_pair=True,
                evaluation_policy_mode="preinterpolation_exact_mixture",
                policy_epsilon=0.1,
            )

    def test_persistent_exact_mixture_evaluation_carries_old_latent_and_clock(self):
        trainer, dataset = self._make_fixed_base_trainer(episodic_latent=False)
        trainer.rl_cfg.eval_num_episodes = 1
        trainer.rl_cfg.mixture_alpha = 0.75

        sample = dataset.samples[0]
        sample["inputs"] = torch.ones(dataset.seq_len, dtype=torch.long)
        sample["initial_plan"] = torch.zeros(dataset.seq_len, dtype=torch.long)
        sample["solution"] = torch.full(
            (dataset.seq_len,), dataset.vocab_size - 1, dtype=torch.long
        )

        action_old = 2
        action_candidate = 3
        action_dim = trainer.policy_model_old.config.rl_num_actions
        old_probs = torch.zeros(1, action_dim)
        candidate_probs = torch.zeros(1, action_dim)
        old_probs[0, action_old] = 1.0
        candidate_probs[0, action_candidate] = 1.0

        record_sha = sample_sha256(sample["inputs"], sample["solution"])
        mixture_probs = (
            (1.0 - trainer.rl_cfg.mixture_alpha) * old_probs
            + trainer.rl_cfg.mixture_alpha * candidate_probs
        )
        for base_seed in range(10_000):
            episode_seed = record_local_evaluation_seed(base_seed, record_sha)
            with torch.random.fork_rng():
                torch.manual_seed(episode_seed)
                sampled_action = int(
                    torch.distributions.Categorical(probs=mixture_probs).sample().item()
                )
            if sampled_action == action_candidate:
                trainer.rl_cfg.eval_seed = base_seed
                break
        else:
            self.fail(
                "Could not find a deterministic seed that samples the candidate action."
            )

        def carry(tag):
            tensor = torch.tensor([[[float(tag)]]])
            return TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=tensor,
                z_L=tensor.clone(),
            )

        initial_carry = carry(0)
        old_output_carries = [carry(1), carry(2), carry(3)]
        old_calls = []
        candidate_calls = []

        def init_latent(_x_batch, _plan):
            return initial_carry

        def old_policy_dist(x_batch, _y_batch, *, n, action_mask, z):
            call_index = len(old_calls)
            old_calls.append(
                {
                    "n": n,
                    "z": z,
                    "remaining_edits": int(x_batch["remaining_edits"].item()),
                }
            )
            return (
                torch.distributions.Categorical(probs=old_probs),
                old_output_carries[call_index],
            )

        def candidate_policy_dist(x_batch, _y_batch, *, n, action_mask, z):
            candidate_calls.append(
                {
                    "n": n,
                    "z": z,
                    "remaining_edits": int(x_batch["remaining_edits"].item()),
                }
            )
            return torch.distributions.Categorical(probs=candidate_probs), z

        with patch.object(
            trainer.policy_model_old,
            "init_latent",
            side_effect=init_latent,
        ), patch.object(
            trainer.policy_model_old,
            "policy_dist",
            side_effect=old_policy_dist,
        ), patch.object(
            trainer.policy_model_candidate,
            "policy_dist",
            side_effect=candidate_policy_dist,
        ):
            metrics = trainer.evaluate_policy_metrics(
                trainer.env_config,
                dataset,
                dummy_checker,
            )

        self.assertEqual(
            [call["n"] for call in old_calls],
            [trainer.rl_cfg.inner_unroll_n] * 3,
        )
        self.assertEqual([call["n"] for call in candidate_calls], [0, 0, 0])
        self.assertEqual([call["remaining_edits"] for call in old_calls], [3, 2, 1])
        self.assertEqual(
            [call["remaining_edits"] for call in candidate_calls],
            [3, 2, 1],
        )
        self.assertIs(old_calls[0]["z"], initial_carry)
        self.assertIs(old_calls[1]["z"], old_output_carries[0])
        self.assertIs(old_calls[2]["z"], old_output_carries[1])
        for call, expected_carry in zip(candidate_calls, old_output_carries):
            self.assertIs(call["z"], expected_carry)

        record = metrics["per_instance"][0]
        actions = record["actions"]
        self.assertEqual(record["environment_interactions"], 3)
        self.assertEqual(record["invalid_action_count"], 0)
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[0], action_candidate)
        self.assertNotEqual(actions[0], action_old)

    def test_legacy_exact_actor_is_insulated_until_post_value_snapshot_sync(self):
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

    def test_legacy_exact_snapshot_copies_nonpolicy_state_and_preserves_policy_heads(self):
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

    def test_legacy_exact_clamp_precedes_target_snapshot_and_policy_update(self):
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

    def _make_fixed_base_trainer(self, *, episodic_latent=True):
        return self._make_trainer(
            episodic_latent=episodic_latent,
            stop_mode="terminal",
            theory_exact_mixture=True,
            exact_baseline_summation=True,
            exact_k_step_targets=True,
            training_protocol="fixed_base_exact",
            enable_contraction=False,
            opnorm_clamp_interval=0,
            value_target_clip=None,
        )

    def test_fixed_base_optimizer_owns_only_value_head(self):
        trainer, _ = self._make_fixed_base_trainer()
        optimizer_ids = {
            id(parameter)
            for group in trainer.value_opt.param_groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for name, parameter in trainer.model.named_parameters()
            if name.startswith("value_head.")
        }

        self.assertEqual(optimizer_ids, expected_ids)
        self.assertTrue(expected_ids)
        self.assertTrue(
            all(
                parameter.requires_grad == name.startswith("value_head.")
                for name, parameter in trainer.model.named_parameters()
            )
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in trainer.policy_model_old.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad == name.startswith("edit_policy.")
                for name, parameter in trainer.policy_model_candidate.named_parameters()
            )
        )

    def test_fixed_base_collection_uses_old_policy_not_deployed_mixture(self):
        trainer, _ = self._make_fixed_base_trainer(episodic_latent=False)
        original_old_policy_dist = trainer.policy_model_old.policy_dist
        with patch.object(
            trainer.model,
            "init_latent",
            side_effect=AssertionError("critic initialized persistent actor state"),
        ), patch.object(
            trainer.policy_model_old,
            "policy_dist",
            wraps=original_old_policy_dist,
        ) as old_policy_dist, patch.object(
            trainer,
            "_mixed_policy_dist",
            side_effect=AssertionError("deployed mixture used for collection"),
        ):
            trainer.collect_episode(max_env_steps=1)

        self.assertGreaterEqual(old_policy_dist.call_count, 1)
        transition = trainer.replay.storage[0]
        batched = trainer._state_is_batched(transition.x)
        x_batch = trainer._prepare_batch_x(transition.x, batched=batched)
        y_batch = trainer._prepare_plan(transition.y, batched=batched)
        action_mask = trainer._compute_training_action_mask(x_batch, y_batch)
        with torch.no_grad():
            old_dist, _ = original_old_policy_dist(
                x_batch,
                y_batch,
                n=trainer.rl_cfg.inner_unroll_n,
                action_mask=action_mask,
                z=transition.latent,
            )
            expected_log_prob = old_dist.log_prob(
                transition.action.to(trainer.device).reshape(1)
            ).cpu().reshape(())
        torch.testing.assert_close(transition.behavior_log_prob, expected_log_prob)

    def test_fixed_base_stays_frozen_across_outer_updates(self):
        trainer, _ = self._make_fixed_base_trainer()
        trainer.set_checker_fn(dummy_checker)
        base_before = {
            name: value.detach().clone()
            for name, value in trainer.policy_model_old.state_dict().items()
            if not name.startswith("value_head.")
        }
        critic_map_before = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
            if not name.startswith(("value_head.", "edit_policy."))
        }

        trainer.train_step()
        trainer.train_step()

        for name, expected in base_before.items():
            torch.testing.assert_close(
                trainer.policy_model_old.state_dict()[name], expected
            )
        for name, expected in critic_map_before.items():
            torch.testing.assert_close(trainer.model.state_dict()[name], expected)
        for name, base_value in trainer.policy_model_old.state_dict().items():
            if name.startswith(("value_head.", "edit_policy.")):
                continue
            torch.testing.assert_close(
                trainer.policy_model_candidate.state_dict()[name], base_value
            )
        for actor in (trainer.policy_model_old, trainer.policy_model_candidate):
            for name, value in trainer.model.value_head.state_dict().items():
                torch.testing.assert_close(actor.value_head.state_dict()[name], value)

    def test_fixed_base_rejects_scheduled_clamp_and_sparse_optimizer(self):
        with self.assertRaisesRegex(ValueError, "fixed_base_exact"):
            self._make_trainer(
                stop_mode="terminal",
                theory_exact_mixture=True,
                exact_baseline_summation=True,
                exact_k_step_targets=True,
                training_protocol="fixed_base_exact",
                enable_contraction=True,
                opnorm_clamp_interval=1,
                value_target_clip=None,
            )

        trainer, _ = self._make_fixed_base_trainer()
        with self.assertRaisesRegex(ValueError, "trainable puzzle embeddings"):
            trainer.set_puzzle_embedding_optimizer(object())

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

    def test_exact_centering_defect_rejects_nonfinite_and_excess_error(self):
        valid = _validated_statewise_centering_defect(
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([[0.5, 0.5]]),
        )
        self.assertEqual(valid, 0.0)

        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _validated_statewise_centering_defect(
                torch.tensor([[float("nan"), 0.0]]),
                torch.tensor([[0.5, 0.5]]),
            )

        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            _validated_statewise_centering_defect(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[0.5, 0.5]]),
            )

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
