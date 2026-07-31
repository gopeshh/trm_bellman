"""
Tests for Lipschitz spectral normalization - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def _make_small_config():
    return dict(
        batch_size=2,
        seq_len=4,
        puzzle_emb_ndim=8,
        num_puzzle_identifiers=3,
        vocab_size=16,
        H_cycles=2,
        L_cycles=2,
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
        puzzle_emb_len=4,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=True,
    )


class TestLipschitzSpectralNorm(unittest.TestCase):
    """Tests for contraction-oriented normalization components."""

    def test_clamping_scaling_and_value_head_norm_are_applied(self):
        """The recurrent path uses scaling; the value head retains spectral norm."""
        config = _make_small_config()
        model = TinyRecursiveReasoningModel_ACTV1(config)

        reasoning_modules = [
            module
            for name, module in model.inner.named_modules()
            if name.startswith("L_level") and hasattr(module, "weight")
        ]
        self.assertTrue(reasoning_modules)
        self.assertTrue(
            any(hasattr(module, "_inner_lip_scale") for module in reasoning_modules),
            "Expected contraction scaling on the recurrent reasoning path",
        )
        self.assertFalse(
            any(hasattr(module, "weight_u") for module in reasoning_modules),
            "The recurrent path uses direct clamping, not spectral-norm wrappers",
        )

        self.assertIsNotNone(model.value_head)
        has_sn_value = False
        for module in model.value_head.modules():
            if getattr(module, "weight_u", None) is not None and getattr(module, "weight_v", None) is not None:
                has_sn_value = True
                break
        self.assertTrue(has_sn_value, "Expected spectral norm on at least one linear in value head")


if __name__ == "__main__":
    unittest.main()
