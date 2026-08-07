"""
Tests for theory-exact components - unittest version.
Converts the pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch
import torch.nn as nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig, merge_rl_config_layer
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from utils.lipschitz import (
    estimate_Cdrift,
    estimate_plan_change,
    compute_value_of_memory_residual,
    compute_exact_baseline_summation,
    compute_exact_advantage,
    compute_theoretical_drift_bound,
)


def get_small_config():
    """Small TRM config for testing."""
    return dict(
        batch_size=4,
        seq_len=16,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=8,
        vocab_size=10,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=2,
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
        rl_enable_contraction=True,
        rl_target_Lz=0.9,
        rl_target_Lv=1.0,
        rl_enable_policy_head=True,
        rl_num_actions=16 * 10 + 1,
        rl_latent_projection_mode="disabled",
        rl_latent_ball_radius=None,
    )


def get_sample_batch():
    """Sample batch for testing."""
    batch_size = 4
    seq_len = 16
    return {
        "inputs": torch.randint(0, 10, (batch_size, seq_len)),
        "puzzle_identifiers": torch.arange(batch_size),
    }


class TestPlanChangeTracking(unittest.TestCase):
    """Tests for plan change tracking Δy_max."""

    def test_plan_change_integer_tokens(self):
        """Test plan change with integer tokens (Hamming distance)."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([1, 2, 9, 4, 5])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 1.0, f"Expected 1 change, got {change}")

    def test_plan_change_multiple_edits(self):
        """Test plan change with multiple edits."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([9, 2, 9, 4, 9])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 3.0, f"Expected 3 changes, got {change}")

    def test_plan_change_batch(self):
        """Test plan change with batched plans."""
        y_old = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y_new = torch.tensor([[1, 9, 3], [9, 9, 6]])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 2.0, f"Expected max change of 2, got {change}")

    def test_plan_change_no_change(self):
        """Test plan change when plans are identical."""
        y = torch.tensor([1, 2, 3, 4, 5])
        change = estimate_plan_change(y, y.clone())
        self.assertEqual(change, 0.0, f"Expected 0 change, got {change}")


class TestDriftBound(unittest.TestCase):
    """Tests for drift bound C_drift(n)."""

    def test_theoretical_drift_bound(self):
        """Test theoretical drift bound computation."""
        kappa_n = 0.5
        E_0 = 1.0
        L_zstar = 0.5
        delta_y_max = 0.1

        drift = compute_theoretical_drift_bound(kappa_n, E_0, L_zstar, delta_y_max)

        expected = 0.5 * 1.0 + (0.5 * 0.5 * 0.1) / 0.5
        self.assertAlmostEqual(drift, expected, places=6)

    def test_drift_bound_not_contractive(self):
        """Test drift bound returns inf when not contractive."""
        drift = compute_theoretical_drift_bound(1.5, 1.0, 0.5, 0.1)
        self.assertEqual(drift, float("inf"))


class TestExactBaseline(unittest.TestCase):
    """Tests exact statewise centering used by Theorem 6.7."""

    def test_exact_advantage_centering(self):
        """Test that exact advantages are centered: E_{a~π}[Â(s,a)] = 0."""
        batch_size = 4
        num_actions = 10

        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)

        exact_baseline = (probs * q_values).sum(dim=-1)
        advantages_all = q_values - exact_baseline.unsqueeze(-1)
        expected_adv = (probs * advantages_all).sum(dim=-1)

        self.assertTrue((expected_adv.abs() < 1e-5).all())

    def test_compute_exact_advantage(self):
        """Test compute_exact_advantage function."""
        batch_size = 4
        num_actions = 10

        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        exact_baseline = (probs * q_values).sum(dim=-1)

        actions = torch.randint(0, num_actions, (batch_size,))
        adv = compute_exact_advantage(q_values, exact_baseline, actions)

        self.assertEqual(adv.shape, (batch_size,))

        for i in range(batch_size):
            expected = q_values[i, actions[i]] - exact_baseline[i]
            self.assertAlmostEqual(adv[i].item(), expected.item(), places=5)


class TestRLConfigTheoryOptions(unittest.TestCase):
    """Tests for new theory-exact config options."""

    def test_default_values(self):
        """Test default values for new config options."""
        cfg = RLConfig()

        self.assertFalse(cfg.exact_baseline_summation)
        self.assertEqual(cfg.latent_projection_mode, "enabled")
        self.assertEqual(cfg.latent_ball_radius, 10.0)
        self.assertFalse(cfg.track_drift_metrics)
        self.assertFalse(cfg.track_plan_change)
        self.assertFalse(cfg.compute_value_of_memory)

    def test_enable_theory_exact(self):
        """Test enabling all theory-exact options."""
        cfg = RLConfig(
            exact_baseline_summation=True,
            latent_projection_mode="enabled",
            latent_ball_radius=5.0,
            track_drift_metrics=True,
            track_plan_change=True,
            compute_value_of_memory=True,
        )

        self.assertTrue(cfg.exact_baseline_summation)
        self.assertEqual(cfg.latent_projection_mode, "enabled")
        self.assertEqual(cfg.latent_ball_radius, 5.0)
        self.assertTrue(cfg.track_drift_metrics)
        self.assertTrue(cfg.track_plan_change)
        self.assertTrue(cfg.compute_value_of_memory)

    def test_projection_disabled_is_explicit_and_serialized_without_radius(self):
        cfg = RLConfig(latent_projection_mode="disabled")

        self.assertEqual(cfg.latent_projection_mode, "disabled")
        self.assertIsNone(cfg.latent_ball_radius)
        serialized = cfg.model_dump()
        self.assertEqual(serialized["latent_projection_mode"], "disabled")
        self.assertIsNone(serialized["latent_ball_radius"])

    def test_legacy_zero_radius_migrates_to_explicit_disabled_mode(self):
        with self.assertWarns(DeprecationWarning):
            cfg = RLConfig(latent_ball_radius=0.0)

        self.assertEqual(cfg.latent_projection_mode, "disabled")
        self.assertIsNone(cfg.latent_ball_radius)

    def test_projection_config_layers_preserve_atomic_mode_radius_contract(self):
        defaults = RLConfig().model_dump()

        disabled = RLConfig(
            **merge_rl_config_layer(
                defaults,
                {"latent_projection_mode": "disabled"},
            )
        )
        self.assertEqual(disabled.latent_projection_mode, "disabled")
        self.assertIsNone(disabled.latent_ball_radius)

        with self.assertWarns(DeprecationWarning):
            legacy_disabled = RLConfig(
                **merge_rl_config_layer(
                    defaults,
                    {"latent_ball_radius": 0.0},
                )
            )
        self.assertEqual(legacy_disabled.latent_projection_mode, "disabled")
        self.assertIsNone(legacy_disabled.latent_ball_radius)

        enabled = merge_rl_config_layer(
            defaults,
            {
                "latent_projection_mode": "enabled",
                "latent_ball_radius": 3.0,
            },
        )
        layered_disabled = RLConfig(
            **merge_rl_config_layer(
                enabled,
                {"latent_projection_mode": "disabled"},
            )
        )
        self.assertEqual(layered_disabled.latent_projection_mode, "disabled")
        self.assertIsNone(layered_disabled.latent_ball_radius)

    def test_projection_contract_rejects_invalid_mode_radius_pairs(self):
        invalid_configs = (
            {"latent_projection_mode": "enabled", "latent_ball_radius": None},
            {"latent_projection_mode": "enabled", "latent_ball_radius": 0.0},
            {"latent_projection_mode": "enabled", "latent_ball_radius": -1.0},
            {"latent_projection_mode": "disabled", "latent_ball_radius": 0.0},
            {"latent_projection_mode": "disabled", "latent_ball_radius": 1.0},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RLConfig(**kwargs)

    def test_algorithm_scalar_domains_accept_all_boundaries(self):
        for tau in (0.0, 1.0):
            for alpha in (0.0, 1.0):
                cfg = RLConfig(
                    gamma=0.0,
                    K=1,
                    inner_unroll_n=0,
                    target_ema_tau=tau,
                    mixture_alpha=alpha,
                )
                self.assertEqual(cfg.gamma, 0.0)
                self.assertEqual(cfg.K, 1)
                self.assertEqual(cfg.inner_unroll_n, 0)

    def test_algorithm_scalar_domains_reject_out_of_range_values(self):
        invalid_configs = (
            {"gamma": -0.01},
            {"gamma": 1.0},
            {"K": 0},
            {"inner_unroll_n": -1},
            {"target_ema_tau": -0.01},
            {"target_ema_tau": 1.01},
            {"mixture_alpha": -0.01},
            {"mixture_alpha": 1.01},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RLConfig(**kwargs)

    def test_fixed_base_protocol_accepts_zero_discount(self):
        cfg = RLConfig(
            gamma=0.0,
            training_protocol="fixed_base_exact",
            stop_action_mode="terminal",
            value_target_clip=None,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            theory_exact_mixture=True,
            policy_epsilon=0.0,
            distill_mixture_policy=False,
            enable_contraction=False,
        )

        self.assertTrue(cfg.is_fixed_base_proposal_exact())

    def test_fixed_base_protocol_requires_terminal_stop(self):
        cfg = RLConfig(
            training_protocol="fixed_base_exact",
            stop_action_mode="noop",
            value_target_clip=None,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            theory_exact_mixture=True,
            policy_epsilon=0.0,
            distill_mixture_policy=False,
            enable_contraction=False,
        )

        self.assertFalse(cfg.is_fixed_base_proposal_exact())
        validation = cfg.validate_theory_alignment(warn=False)
        self.assertTrue(
            any("stop_action_mode='terminal'" in issue for issue in validation["issues"])
        )

    def test_fixed_base_protocol_rejects_value_target_clipping(self):
        cfg = RLConfig(
            training_protocol="fixed_base_exact",
            stop_action_mode="terminal",
            value_target_clip=50.0,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            theory_exact_mixture=True,
            policy_epsilon=0.0,
            distill_mixture_policy=False,
            enable_contraction=False,
        )

        self.assertFalse(cfg.is_fixed_base_proposal_exact())
        validation = cfg.validate_theory_alignment(warn=False)
        self.assertTrue(
            any("value_target_clip=None" in issue for issue in validation["issues"])
        )

    def test_fixed_base_contraction_requires_immutable_value_head_norm(self):
        cfg = RLConfig(
            training_protocol="fixed_base_exact",
            stop_action_mode="terminal",
            value_target_clip=None,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            theory_exact_mixture=True,
            policy_epsilon=0.0,
            distill_mixture_policy=False,
            enable_contraction=True,
            opnorm_clamp_interval=0,
            disable_value_head_norm=False,
        )

        self.assertFalse(cfg.is_fixed_base_proposal_exact())
        validation = cfg.validate_theory_alignment(warn=False)
        self.assertTrue(
            any("disable_value_head_norm=True" in issue for issue in validation["issues"])
        )


class TestForwardInvariantProjection(unittest.TestCase):
    """Tests for forward-invariant projection (Eq. 14 in paper)."""

    def setUp(self):
        self.config = get_small_config()
        self.sample_batch = get_sample_batch()
        self.sample_plan = torch.randint(0, 10, self.sample_batch["inputs"].shape)

    def test_projection_bounds_latent_norm(self):
        """Test that projection keeps ||z|| ≤ R."""
        config = dict(self.config)
        config["rl_latent_projection_mode"] = "enabled"
        config["rl_latent_ball_radius"] = 5.0
        model = TinyRecursiveReasoningModel_ACTV1(config)
        R = model.config.rl_latent_ball_radius
        self.assertIsNotNone(R)

        z = model.init_latent(self.sample_batch, self.sample_plan)
        for _ in range(10):
            z = model.update_latent(z, self.sample_plan, self.sample_batch)

        z_H_norm = z.z_H.norm(p=2, dim=(1, 2))
        z_L_norm = z.z_L.norm(p=2, dim=(1, 2))
        joint_norm = torch.sqrt(z_H_norm.square() + z_L_norm.square())

        self.assertTrue((z_H_norm <= R + 1e-5).all())
        self.assertTrue((z_L_norm <= R + 1e-5).all())
        self.assertTrue((joint_norm <= R + 1e-5).all())

    def test_projection_is_exact_below_previous_numerical_floor(self):
        config = {
            **self.config,
            "rl_latent_projection_mode": "enabled",
            "rl_latent_ball_radius": 1e-12,
        }
        model = TinyRecursiveReasoningModel_ACTV1(config)
        z_h = torch.tensor([[[2e-12]], [[5e-13]], [[0.0]]])
        z_l = torch.zeros_like(z_h)

        projected_h, projected_l = model.inner._project_carry_to_ball(
            z_h,
            z_l,
            1e-12,
        )

        torch.testing.assert_close(
            projected_h,
            torch.tensor([[[1e-12]], [[5e-13]], [[0.0]]]),
            rtol=1e-5,
            atol=0.0,
        )
        torch.testing.assert_close(projected_l, z_l, rtol=0.0, atol=0.0)

    def test_projection_norm_avoids_float32_square_overflow(self):
        model = TinyRecursiveReasoningModel_ACTV1(
            {
                **self.config,
                "rl_latent_projection_mode": "enabled",
                "rl_latent_ball_radius": 2.0,
            }
        )
        z_h = torch.tensor([[[1e20]]], dtype=torch.float32)
        z_l = torch.zeros_like(z_h)

        norm, _, _ = model.inner._joint_carry_geometry(z_h, z_l)
        projected_h, projected_l = model.inner._project_carry_to_ball(
            z_h,
            z_l,
            2.0,
        )

        torch.testing.assert_close(norm, torch.tensor([[[1e20]]]))
        torch.testing.assert_close(projected_h, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(projected_l, z_l, atol=0.0, rtol=0.0)

    def test_projection_norm_avoids_float32_square_underflow(self):
        radius = 1e-35
        model = TinyRecursiveReasoningModel_ACTV1(
            {
                **self.config,
                "rl_latent_projection_mode": "enabled",
                "rl_latent_ball_radius": radius,
            }
        )
        z_h = torch.tensor([[[1e-30]]], dtype=torch.float32)
        z_l = torch.zeros_like(z_h)

        norm, _, _ = model.inner._joint_carry_geometry(z_h, z_l)
        projected_h, projected_l = model.inner._project_carry_to_ball(
            z_h,
            z_l,
            radius,
        )

        torch.testing.assert_close(norm, torch.tensor([[[1e-30]]]))
        torch.testing.assert_close(
            projected_h,
            torch.tensor([[[radius]]]),
            rtol=1e-5,
            atol=0.0,
        )
        torch.testing.assert_close(projected_l, z_l, atol=0.0, rtol=0.0)

    def test_disabled_projection_is_the_identity_operator(self):
        """Disabled mode returns the raw recurrent update unchanged."""
        model = TinyRecursiveReasoningModel_ACTV1(self.config)
        carry = model.init_latent(self.sample_batch, self.sample_plan)
        batch = model._standardize_latent_batch(
            self.sample_batch,
            self.sample_plan,
        )
        context = model._resolve_latent_context(batch)
        input_embeddings = context["input_embeddings_with_plan"]

        with torch.no_grad():
            expected_l = carry.z_L
            for _ in range(model.config.L_cycles):
                expected_l = model.inner.L_level(
                    expected_l,
                    carry.z_H + input_embeddings,
                    **context["seq_info"],
                )
            expected_h = model.inner.L_level(
                carry.z_H,
                expected_l,
                **context["seq_info"],
            )
            actual, _, active = model.inner.latent_step_with_projection_info(
                carry,
                input_embeddings,
                context["seq_info"],
            )

        torch.testing.assert_close(actual.z_H, expected_h)
        torch.testing.assert_close(actual.z_L, expected_l)
        self.assertFalse(bool(active.any().item()))

    def test_model_projection_contract_and_legacy_migration(self):
        with self.assertRaises(ValueError):
            TinyRecursiveReasoningModel_ACTV1Config(
                **{
                    **self.config,
                    "rl_latent_projection_mode": "enabled",
                    "rl_latent_ball_radius": 0.0,
                }
            )
        with self.assertRaises(ValueError):
            TinyRecursiveReasoningModel_ACTV1Config(
                **{
                    **self.config,
                    "rl_latent_projection_mode": "disabled",
                    "rl_latent_ball_radius": 1.0,
                }
            )

        legacy = dict(self.config)
        legacy.pop("rl_latent_projection_mode")
        legacy["rl_latent_ball_radius"] = 0.0
        with self.assertWarns(DeprecationWarning):
            migrated = TinyRecursiveReasoningModel_ACTV1Config(**legacy)
        self.assertEqual(migrated.rl_latent_projection_mode, "disabled")
        self.assertIsNone(migrated.rl_latent_ball_radius)

    def test_projection_radius_must_be_representable_in_latent_dtype(self):
        invalid = (
            ("float16", 1e-12),
            ("bfloat16", 1e-45),
            ("float32", 4e38),
        )
        for forward_dtype, radius in invalid:
            with self.subTest(forward_dtype=forward_dtype, radius=radius):
                with self.assertRaisesRegex(
                    ValueError,
                    "represented in forward_dtype",
                ):
                    TinyRecursiveReasoningModel_ACTV1Config(
                        **{
                            **self.config,
                            "forward_dtype": forward_dtype,
                            "rl_latent_projection_mode": "enabled",
                            "rl_latent_ball_radius": radius,
                        }
                    )

        for forward_dtype in ("float16", "bfloat16", "float32"):
            limits = torch.finfo(getattr(torch, forward_dtype))
            with self.subTest(forward_dtype=forward_dtype, radius="tiny"):
                config = TinyRecursiveReasoningModel_ACTV1Config(
                    **{
                        **self.config,
                        "forward_dtype": forward_dtype,
                        "rl_latent_projection_mode": "enabled",
                        "rl_latent_ball_radius": limits.tiny,
                    }
                )
                self.assertEqual(config.rl_latent_ball_radius, limits.tiny)


if __name__ == "__main__":
    unittest.main()
