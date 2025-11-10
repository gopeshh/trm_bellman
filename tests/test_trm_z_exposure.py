"""
Unit tests for TRM z_n exposure (Step 10).

Verify that:
1. Existing supervised training is unchanged by default (return_z=False)
2. When return_z=True, we can access z_n from the inner recursion
"""
import pytest

torch = pytest.importorskip("torch")

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1  # noqa: E402


def test_trm_default_behavior_unchanged():
    """Test that default behavior (return_z=False) is unchanged."""
    config_dict = {
        "batch_size": 2,
        "seq_len": 9,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 100,
        "vocab_size": 10,
        "H_cycles": 2,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2.0,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    model.eval()

    # Create a batch
    batch = {
        "inputs": torch.randint(0, 10, (2, 9)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    # Initialize carry
    carry = model.initial_carry(batch)

    # Forward without return_z (default)
    new_carry, outputs = model(carry, batch)

    # Check that outputs contain expected keys
    assert "logits" in outputs
    assert "q_halt_logits" in outputs
    assert "q_continue_logits" in outputs

    # Check that z_n is NOT in outputs by default
    assert "z_n" not in outputs

    # Check that logits have the correct shape
    assert outputs["logits"].shape == (2, 9, 10)  # (batch, seq, vocab)


def test_trm_return_z_enabled():
    """Test that return_z=True exposes z_n in outputs."""
    config_dict = {
        "batch_size": 2,
        "seq_len": 9,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 100,
        "vocab_size": 10,
        "H_cycles": 2,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2.0,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    model.eval()

    # Create a batch
    batch = {
        "inputs": torch.randint(0, 10, (2, 9)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    # Initialize carry
    carry = model.initial_carry(batch)

    # Forward WITH return_z=True
    new_carry, outputs = model(carry, batch, return_z=True)

    # Check that z_n IS in outputs
    assert "z_n" in outputs

    # z_n should be a tuple of (z_H, z_L)
    z_n = outputs["z_n"]
    assert isinstance(z_n, tuple)
    assert len(z_n) == 2

    z_H, z_L = z_n

    # Check shapes (should be [batch, seq_len, hidden_size])
    assert z_H.shape == (2, 9, 32)
    assert z_L.shape == (2, 9, 32)

    # Check that other outputs are still present
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 9, 10)


def test_trm_z_different_across_batches():
    """Test that z_n varies with different inputs."""
    config_dict = {
        "batch_size": 2,
        "seq_len": 9,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 100,
        "vocab_size": 10,
        "H_cycles": 2,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2.0,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    model.eval()

    # Create two different batches
    batch1 = {
        "inputs": torch.zeros(2, 9, dtype=torch.long),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    batch2 = {
        "inputs": torch.ones(2, 9, dtype=torch.long),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    # Get z_n for both batches
    carry1 = model.initial_carry(batch1)
    _, outputs1 = model(carry1, batch1, return_z=True)
    z_H1, z_L1 = outputs1["z_n"]

    carry2 = model.initial_carry(batch2)
    _, outputs2 = model(carry2, batch2, return_z=True)
    z_H2, z_L2 = outputs2["z_n"]

    # z_n should be different for different inputs
    assert not torch.allclose(z_H1, z_H2, atol=1e-5)
    assert not torch.allclose(z_L1, z_L2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
