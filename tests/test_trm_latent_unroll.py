import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def _make_config():
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
        expansion=2,
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
    )


def _dummy_batch(config):
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    vocab_size = config["vocab_size"]
    num_puzzle_identifiers = config["num_puzzle_identifiers"]
    return {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, num_puzzle_identifiers, (batch_size,)),
    }


def test_latent_unroll_helpers_shapes_and_determinism():
    torch.manual_seed(0)
    config = _make_config()
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.eval()

    batch = _dummy_batch(config)
    plan = torch.randint_like(batch["inputs"], low=0, high=config["vocab_size"])

    z0 = model.init_latent(batch, plan)
    z_n, zs = model.unroll_latent(batch, plan, n=3)
    z_n_eval = model.eval_latent(batch, plan, n=3)

    expected_shape = (
        config["batch_size"],
        config["seq_len"] + config["puzzle_emb_len"],
        config["hidden_size"],
    )
    assert z0.z_H.shape == expected_shape
    assert z0.z_L.shape == expected_shape
    assert len(zs) == 4  # n + 1
    assert torch.allclose(zs[0].z_H, z0.z_H)
    assert torch.allclose(zs[0].z_L, z0.z_L)

    # Consistency between helper APIs
    torch.testing.assert_close(z_n.z_H, z_n_eval.z_H)
    torch.testing.assert_close(z_n.z_L, z_n_eval.z_L)

    # Determinism for fixed weights and inputs
    z_n_eval_2 = model.eval_latent(batch, plan, n=3)
    torch.testing.assert_close(z_n_eval.z_H, z_n_eval_2.z_H)
    torch.testing.assert_close(z_n_eval.z_L, z_n_eval_2.z_L)

    # Check that every stored latent shares the same shape
    for latent_state in zs:
        assert latent_state.z_H.shape == expected_shape
        assert latent_state.z_L.shape == expected_shape

